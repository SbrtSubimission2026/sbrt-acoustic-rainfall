#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Dataset Builder
=======================
Handles data filtering, undersampling, and strict zero-contamination splitting.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-06
:Version: 1.0.2
:License: Private (Internal Use Only)
:Status: Beta
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import logging
from rainfall_acoustic_classification.utils import get_standard_logger


def discard_files(
    df: pd.DataFrame, 
    discard_rules: List[str], 
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Applies a list of pandas query strings to discad invalid records.

    Evaluates each rule against the DataFrame. Records that evaluate to 
    True for any given rule are discarded.

    Parameters
    ----------
    df : pandas.DataFrame
        The raw input dataset.
    discard_rules : List[str]
        A list of string queries valid for `pandas.DataFrame.eval()`.
        Example: ["snr_db < 5.0", "duration < 9.5"].
    logger : Optional[logging.Logger], default=None
        An optional logger instance. If None, a standard logger is created.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only the records that passed all rules.

    Raises
    ------
    Exception
        If a provided query string is invalid or fails during evaluation.
    """
    log = logger or get_standard_logger("DataCurator")
    df_clean = df.copy()

    if not discard_rules:
        log.info("No discard rules provided. Bypassing curation.")
        return df_clean

    log.info(f"Applying {len(discard_rules)} discard rule(s)...")
    
    for rule in discard_rules:
        try:
            # '~' inverts the mask: we keep records that DO NOT match the discard rule
            mask = df_clean.eval(rule)
            df_clean = df_clean[~mask]
        except Exception as e:
            log.error(f"Rule evaluation failed for '{rule}'. Error: {e}")
            raise e

    log.info(f"Curation complete. Remaining records: {len(df_clean)} (Original: {len(df)}).")
    return df_clean


def undersample_1_to_1(
    df: pd.DataFrame,
    target_col: str,
    anchor_classes: List[str],
    sample_classes: List[str],
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Performs exact 1:1 class undersamplig between an anchor group and a sampling pool.

    The anchor group dictates the maximum allowable size. The sampling pool 
    is randomly undersampled to match the exact size of the anchor group.

    Parameters
    ----------
    df : pandas.DataFrame
        The curated input dataset.
    target_col : str
        The name of the column containing the class labels.
    anchor_classes : List[str]
        The class labels that form the baseline group (e.g., ['light', 'heavy']).
    sample_classes : List[str]
        The class labels that will be undersampled to match the anchor's size 
        (e.g., ['no-rain']).
    random_state : int, default=42
        Seed for the random number generator to ensure reproducibility.
    logger : Optional[logging.Logger], default=None
        An optional logger instance. If None, a standard logger is created.

    Returns
    -------
    pandas.DataFrame
        A shuffled, balanced DataFrame containing an equal amount of records 
        from both the anchor and the sampled classes.
    """
    log = logger or get_standard_logger("DataBalancer")
    log.info(f"Initiating 1:1 undersamplig on column '{target_col}'.")

    # Isolate the two groups
    df_anchor = df[df[target_col].isin(anchor_classes)]
    df_pool = df[df[target_col].isin(sample_classes)]

    anchor_count = len(df_anchor)
    pool_count = len(df_pool)

    if anchor_count == 0:
        log.warning("No anchor data found! Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns)

    log.info(f"Anchor group size: {anchor_count} | Pool group size: {pool_count}")

    # Enforce strict 1:1 parity
    if pool_count < anchor_count:
        log.warning(
            f"Pool size ({pool_count}) is smaller than Anchor size ({anchor_count}). "
            "Using all available pool records without matching 1:1."
        )
        df_sampled = df_pool
    else:
        df_sampled = df_pool.sample(n=anchor_count, random_state=random_state)

    # Concatenate and shuffle
    df_balanced = pd.concat([df_anchor, df_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    log.info(f"Balanced Dataset generated: {len(df_balanced)} total records.")
    return df_balanced


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Splits a DataFrame into Train, Validation, and Test sets with exact global proportions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The fully cleaned and preprocessed dataset.
    test_size : float
        The global proportion of the dataset to include in the test split.
    val_size : float
        The global proportion of the dataset to include in the validation split.
    stratify_cols : Optional[List[str]]
        List of column names to use for stratified sampling.
    random_state : int
        Seed for reproducibility.
        
    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]
        The train, validation (or None), and test DataFrames.
    """
    logger = logging.getLogger("DatasetSplitter")
    
    if df.empty:
        raise ValueError("Serious Quack. Input DataFrame is empty.")
        
    if test_size + val_size >= 1.0:
        raise ValueError("Test and Validation sizes combined must be less than 1.0.")

    # 1. Stratification Setup (Scikit-learn handles multiple columns directly)
    stratify_data = df[stratify_cols] if stratify_cols else None

    # 2. First Split: Extract the Test Set
    train_temp, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_data
    )

    # 3. Second Split: Extract the Validation Set
    if val_size > 0.0:
        # Mathematical adjustment to ensure global proportions are respected
        relative_val_size = val_size / (1.0 - test_size)
        
        strat_val_data = train_temp[stratify_cols] if stratify_cols else None
        
        train, val = train_test_split(
            train_temp,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=strat_val_data
        )
        logger.info(f"Split sizes -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return train, val, test

    logger.info(f"Split sizes -> Train: {len(train_temp)} | Test: {len(test)}")
    return train_temp, None, test

if __name__ == "__main__":
    pass