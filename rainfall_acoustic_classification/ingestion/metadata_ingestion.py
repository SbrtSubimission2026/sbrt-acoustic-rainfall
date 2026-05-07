#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Metadata Ingestion 
=========================
A clean, functional pipeline to scan audio files, extract metadata via regex, 
and merge ground-truth data.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-06
:Version: 1.0.0
:License: Private (Internal Use Only)
:Status: Development
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union
import pandas as pd
import numpy as np

def scan_files(root_dir: Union[str, Path], extensions: Tuple[str, ...] = ('.wav',), location: Optional[str] = None) -> pd.DataFrame:
    """
    Scans a directory recursively for audio files, ignoring hidden folders.

    Traverses the specified root directory and constructs a DataFrame containing
    the basic metadata (filename, absolute path, extension, and inferred location)
    for all files matching the specified extensions.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        The root directory to start the recursive scan.
    extensions : tuple of str, default=('.wav',)
        A tuple containing the allowed file extensions. Extensions must be 
        lowercase and include the leading dot.
    location : str, optional
        An explicit location label to assign to all scanned files. If None, 
        the name of the `root_dir` is used by default.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the inventory of found files with columns:
        ['file_name', 'file_path', 'extension', 'location'].
        Returns an empty DataFrame if no matching files are found.

    Raises
    ------
    FileNotFoundError
        If the specified `root_dir` does not exist or is not a directory.
    """
    root_path = Path(root_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_path}")

    found_paths = []
    
    # os.walk is extremely fast for deep directory trees
    for root, dirs, files in os.walk(root_path):
        # Modifying dirs in-place to skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith(extensions):
                found_paths.append(Path(root) / file)

    # Deterministic sort
    found_paths.sort()

    if not found_paths:
        return pd.DataFrame()
    
    final_location = location if location is not None else root_path.name

    # Build initial dataframe
    records = [{
        'file_name': p.name,
        'file_path': str(p.resolve()),
        'extension': p.suffix.lower(),
        # Simple heuristic for location (the parent folder name)
        'location': final_location
    } for p in found_paths]

    return pd.DataFrame(records)


def extract_filename_metadata(df: pd.DataFrame, regex_pattern: str) -> pd.DataFrame:
    """
    Applies a regular expression pattern to filenames to extract metadata into columns.

    This function performs vectorized extraction of meteorological and temporal 
    metadata from structured filenames, cleans specific fields (e.g., recorder ID), 
    casts timestamps to datetime objects, and infers periods of the day.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing at least a 'file_name' column.
    regex_pattern : str
        The regular expression pattern with named capture groups to extract 
        metadata from the 'file_name' column.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the original columns plus the newly extracted 
        and processed metadata columns (e.g., 'timestamp', 'period', 'mm_hr').
        Returns the original DataFrame unmodified if it is empty or lacks a 
        'file_name' column.
    """
    if df.empty or 'file_name' not in df.columns:
        return df

    # 1. Vectorized Regex Extraction (Pandas native)
    extracted = df['file_name'].str.extract(regex_pattern)
    df = pd.concat([df, extracted], axis=1)

    # 2. Clean Recorder ID (Removes trailing letters/numbers after SMM if needed)
    if 'recorder' in df.columns:
        df['recorder'] = df['recorder'].str.extract(r'(SMM\d+)', expand=False).fillna(df['recorder'])

    # 3. Process Rain Data (Replace '-' with '.' and calculate mm/hr)
    if 'rain_mm_str' in df.columns:
        df['mm_5min'] = pd.to_numeric(df['rain_mm_str'].str.replace('-', '.', regex=False), errors='coerce')
        df['mm_hr'] = (df['mm_5min'] * 12).round(2)
        df.drop(columns=['rain_mm_str'], inplace=True)

    # 4. Cast Timestamps (Vectorized datetime conversion)
    date_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']
    if set(date_cols).issubset(df.columns):
        # Creates a single string "YYYY-MM-DD HH:MM:SS" and converts safely
        datetime_str = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + \
                       df['hour'] + ':' + df['minute'] + ':' + df['second']
        df['timestamp'] = pd.to_datetime(datetime_str, errors='coerce')

    # 5. Time Labels (Infer periods of the day)
    if 'hour' in df.columns:
        bins = [0, 6, 12, 18, 24]
        labels = ['overnight', 'morning', 'afternoon', 'night']
        df['period'] = pd.cut(
            pd.to_numeric(df['hour'], errors='coerce'), 
            bins=bins, labels=labels, right=False
        )

    if 'category' in df.columns:
        df['category'] = df['category'].fillna('unlabeled')
        
    return df


def merge_ground_truth(df_raw: pd.DataFrame, audit_path: Union[str, Path], join_key: str = 'file_name', gt_mapping: Optional[Dict[str, str]] = None, no_overwrite: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merges human-annotated Ground Truth data with the automatically extracted metadata.

    Reads a Ground Truth file (CSV or Parquet), standardizes its column names, 
    and performs a left join with the raw metadata DataFrame. In case of 
    conflicts, Ground Truth values take precedence and overwrite the extracted ones.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        The DataFrame containing the automatically extracted metadata.
    audit_path : str or pathlib.Path
        The path to the Ground Truth annotation file (.csv or .parquet).
    join_key : str, default='file_name'
        The column name used as the index to join both DataFrames.
    gt_mapping : dict, optional
        A dictionary mapping the Ground Truth column names to the standard schema.
        If None, a default mapping is applied to ensure backward compatibility.
    no_overwrite : list of str, optional
        A list of column names that should NOT be overwritten by the Ground Truth 
        data in case of conflicts (e.g., ['timestamp']). If None, all conflicting 
        columns are overwritten by the Ground Truth.

    Returns
    -------
    pandas.DataFrame
        A merged and reordered DataFrame containing both the extracted metadata 
        and the aligned Ground Truth annotations.

    Raises
    ------
    FileNotFoundError
        If the Ground Truth file specified by `audit_path` does not exist.
    ValueError
        If the Ground Truth file is not a supported format (.csv or .parquet).
    KeyError
        If the specified `join_key` is not found in the Ground Truth DataFrame.
    """
    if df_raw.empty:
        return df_raw

    path = Path(audit_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground Truth file not found: {path}")

    # Load Ground Truth
    if path.suffix == '.csv':
        df_gt = pd.read_csv(path)
    elif path.suffix == '.parquet':
        df_gt = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported format. Use .csv or .parquet")
    
    # Rename GT columns to match our standard schema
    if gt_mapping is None:
        gt_mapping = {
            'TIMESTAMP': 'timestamp', 
            'filename': 'file_name', 
            'total_rain': 'mm_5min', 
            'rain_class': 'category',
            'rain': 'is_rain'
        }
    df_gt.rename(columns=gt_mapping, inplace=True)
    
    # Cast boolean robustly if 'is_rain' exists
    if 'is_rain' in df_gt.columns:
        df_gt['is_rain'] = pd.to_numeric(df_gt['is_rain'], errors='coerce').astype('boolean')

    # Ensure join key exists and is clean
    if join_key in df_gt.columns:
        df_gt[join_key] = df_gt[join_key].apply(lambda x: Path(str(x)).name)
        df_gt.drop_duplicates(subset=[join_key], inplace=True)
    else:
        raise KeyError(f"Join key '{join_key}' not found in Ground Truth.")

    # Left Join (Keep all audio files, add GT data where it exists)
    df_merged = pd.merge(df_raw, df_gt, on=join_key, how='left', suffixes=('', '_gt'))

    protected_cols = no_overwrite if no_overwrite is not None else []

    # Conflict Resolution: Ground Truth overwrites extracted metadata
    for col in df_raw.columns:
        audit_col = f"{col}_gt"
        if audit_col in df_merged.columns:
            if col not in protected_cols:
                # Overwrite original with GT if GT is not NaN and column is not protected
                df_merged[col] = df_merged[audit_col].combine_first(df_merged[col])
            
            # Clean up the audit column to maintain the standard schema
            df_merged.drop(columns=[audit_col], inplace=True)

    # Reorder columns for readability
    desired_order = [
        'file_name', 'timestamp', 'period', 'is_rain', 'mm_5min', 'mm_hr', 
        'category', 'recorder', 'location', 'file_path'
    ]
    final_cols = [c for c in desired_order if c in df_merged.columns]
    # Append any extra columns from GT that were not in the desired order
    final_cols += [c for c in df_merged.columns if c not in final_cols]

    return df_merged[final_cols]


def sanitize_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes the DataFrame immediately after merging the ground truth audit data.

    This function resolves nomenclature inconsistencies introduced by human auditors,
    removes unnecessary columns, and recalculates dependent meteorological variables 
    to ensure data consistency before entering the filtering and splitting stages.

    Parameters
    ----------
    df : pandas.DataFrame
        The merged DataFrame containing both the raw sensor metadata and the 
        injected ground truth labels.

    Returns
    -------
    pandas.DataFrame
        A sanitized copy of the input DataFrame with standardized categories 
        and mathematically accurate precipitation metrics.
    """
    if df.empty:
        return df
        
    df_clean = df.copy()

    # =======================================================
    # 1. PURGE REDUNDANT COLUMNS
    # Remove the 'rain' column introduced by the auditor
    # =======================================================
    df_clean = df_clean.drop(columns=['rain'], errors='ignore')

    # =======================================================
    # 2. NOMENCLATURE STANDARDIZATION
    # Force the auditor's 'no rain' to match the 'no-rain' standard
    # =======================================================
    if 'category' in df_clean.columns:
        # Cast to string safely in case of purely null columns, then standardize
        df_clean['category'] = (
            df_clean['category']
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({'no rain': 'no-rain'})
        )
        # Restore actual NaNs that became the string 'nan' during astype(str)
        df_clean['category'] = df_clean['category'].replace({'nan': float('nan')})

    # =======================================================
    # 3. METRIC RECALCULATION
    # The hourly rate is strictly derived from the 5-minute volume.
    # This automatically overwrites any human errors or NaNs in 'mm_hr'
    # =======================================================
    if 'mm_5min' in df_clean.columns:
        # 12 periods of 5 minutes = 1 hour
        df_clean['mm_hr'] = (df_clean['mm_5min'] * 12).round(2)

    return df_clean


if __name__ == '__main__':
    pass