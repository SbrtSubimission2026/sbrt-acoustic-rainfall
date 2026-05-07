#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Audio Core and Catalog
==============================
Provides foundational data structures, robust input/output (I/O) operations 
for audio files, and a memory-efficient cataloging system to manage 
acoustic metadata across the pipeline.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-09
:Version: 1.1.0
:License: Private (Internal Use Only)
:Status: Development
"""
import csv
import hashlib
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Iterator

from rainfall_acoustic_classification.utils import get_standard_logger


def load_audio_sample(
    file_path: Union[str, Path],
    sample_rate: int = 24000,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Optional['AudioSample']:
    """
    Safely loads an audio file from disk into an AudioSample Data Transfer Object.

    Utilizes `librosa` for robust audio decoding and resampling. If the file is 
    corrupted, empty, or missing, it logs the error and returns None instead 
    of halting the execution pipeline.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The absolute or relative path to the audio file.
    sample_rate : int, default=24000
        The target sampling rate. The audio will be resampled if necessary.
    duration : float, optional
        Only load up to this much audio (in seconds). Loads the entire file if None.
    metadata : dict, optional
        A dictionary of contextual metadata to attach to the resulting AudioSample.
    logger : logging.Logger, optional
        An optional logger instance. If None, a standard logger is created.

    Returns
    -------
    AudioSample or None
        An instantiated AudioSample object containing the waveform and metadata.
        Returns None if the file could not be loaded or is empty.
    """
    if logger is None:
        from rainfall_acoustic_classification.utils import get_standard_logger
        log = get_standard_logger("AudioLoader")
    else:
        log = logger
    path_obj = Path(file_path)

    if not path_obj.exists():
        log.error(f"File not found: {path_obj.absolute()}")
        return None

    try:
        # Catching UserWarnings from librosa (e.g., empty files or PySoundFile issues)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path_obj, sr=sample_rate, duration=duration)
            
        if y.size == 0:
            log.warning(f"Audio file is empty after loading: {path_obj.name}")
            return None

        sample = AudioSample(
            audio_data=y,
            sample_rate=sr,
            file_path=path_obj,
            metadata=metadata
        )
        
        log.debug(f"Successfully loaded: {sample}")
        return sample

    except Exception as e:
        log.error(f"Failed to load audio {path_obj.name}. Error: {e}")
        return None


def save_audio_sample(
    sample: 'AudioSample',
    output_dir: Union[str, Path],
    extension: str = ".wav",
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Saves the audio waveform from an AudioSample object back to disk.

    Creates the target directory if it does not exist and writes the 
    underlying NumPy array to an audio file using the `soundfile` library 
    for high-fidelity encoding.

    Parameters
    ----------
    sample : AudioSample
        The data transfer object containing the audio array and sample rate.
    output_dir : str or pathlib.Path
        The destination directory where the file should be saved.
    extension : str, default=".wav"
        The desired audio format extension (e.g., '.wav', '.flac'). 
        If it lacks a leading dot, one will be added automatically.
    logger : logging.Logger, optional
        An optional logger instance. If None, a standard logger is created.

    Returns
    -------
    pathlib.Path or None
        The absolute path to the successfully saved audio file, 
        or None if the operation fails.
    """
    log = logger or get_standard_logger("AudioSaver")
    out_dir = Path(output_dir)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not extension.startswith("."):
            extension = f".{extension}"
            
        out_filename = sample.file_path.with_suffix(extension).name
        out_path = out_dir / out_filename

        sf.write(
            file=str(out_path), 
            data=sample.audio_data, 
            samplerate=sample.sample_rate
        )
        
        log.info(f"Audio successfully saved to: {out_path.name}")
        return out_path

    except Exception as e:
        log.error(f"Failed to save audio '{sample.file_name}' to '{out_dir}'. Error: {e}")
        return None


class AudioSample:
    """
    Data Transfer Object (DTO) representing a single audio file or segment.
    
    This class acts as the single source of truth for an acoustic sample throughout 
    the processing pipeline. It encapsulates the raw waveform, sample rate, 
    file identity, and dynamically stores metadata and extracted acoustic features.

    Parameters
    ----------
    audio_data : numpy.ndarray
        The 1D array representing the audio time-series.
    sample_rate : int
        The sampling rate of the audio data.
    file_path : str or pathlib.Path
        The absolute or relative path to the original audio file.
    metadata : dict, optional
        A dictionary containing contextual information (e.g., category, location, recorder).

    Attributes
    ----------
    audio_data : numpy.ndarray
        The raw waveform array.
    sample_rate : int
        The sampling rate used.
    file_path : pathlib.Path
        The resolved path to the associated physical file.
    file_name : str
        The name of the file (extracted from `file_path`).
    metadata : dict
        Dictionary of contextual data attached to the sample.
    features : dict
        Dictionary storing extracted acoustic metrics dynamically.
    """

    def __init__(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        file_path: Union[str, Path], 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.metadata = metadata or {}
        self.features: Dict[str, float] = {}

    def add_features(self, new_features: Dict[str, float]) -> None:
        """
        Updates the internal features dictionary with new acoustic metrics.

        Parameters
        ----------
        new_features : dict
            A dictionary mapping feature names (str) to their values (float) 
            to append or update in the current sample.
            
        Returns
        -------
        None
        """
        self.features.update(new_features)

    def __repr__(self) -> str:
        return (
            f"<AudioSample(file_name='{self.file_name}', "
            f"sr={self.sample_rate}, duration={len(self.audio_data)/self.sample_rate:.2f}s, "
            f"features_count={len(self.features)})>"
        )


class AudioCatalog:
    """
    Manager class for a collection of AudioSample objects using a CSV backend.
    
    Handles the persistence of metadata, generates unique cryptographic 
    IDs for each audio file to prevent duplication, and acts as a lazy-loading 
    factory to stream AudioSample objects without exhausting system RAM.

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        The file path to the CSV file serving as the catalog database.
    logger : logging.Logger, optional
        An optional logger instance.

    Attributes
    ----------
    catalog_path : pathlib.Path
        The absolute path to the CSV database.
    logger : logging.Logger
        The logger instance bound to this catalog.
    df : pandas.DataFrame
        The in-memory representation of the catalog's metadata.
    """

    def __init__(
        self, 
        catalog_path: Union[str, Path], 
        logger: Optional[logging.Logger] = None
    ):
        self.catalog_path = Path(catalog_path)
        self.logger = logger or logging.getLogger("AudioCatalog")
        self.df = self._initialize_catalog()

    def _initialize_catalog(self) -> pd.DataFrame:
        """
        Loads the existing CSV catalog or creates an empty DataFrame if none exists.

        Returns
        -------
        pandas.DataFrame
            The loaded or empty catalog DataFrame with required base columns.
        """
        if self.catalog_path.exists():
            self.logger.info(f"Loading existing catalog from {self.catalog_path.name}")
            return pd.read_csv(self.catalog_path)
        else:
            self.logger.info("Creating a new empty catalog.")
            return pd.DataFrame(columns=["audio_id", "file_name", "sample_rate", "file_path"])

    @staticmethod
    def generate_hash_id(file_name: str, metadata: Dict[str, Any]) -> str:
        """
        Generates a unique SHA-256 hash for an audio sample based on its core identity.

        Parameters
        ----------
        file_name : str
            The name of the audio file.
        metadata : dict
            The metadata dictionary containing defining traits (e.g., device, timestamp).

        Returns
        -------
        str
            A 16-character hexadecimal string representing the unique cryptographic ID.
        """
        signature = f"{file_name}_{metadata.get('device', '')}_{metadata.get('timestamp', '')}"
        return hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]

    def register_sample(self, sample: 'AudioSample') -> str:
        """
        Registers an AudioSample's metadata into the catalog. 
        
        Does not save the audio array, only the structural and contextual data.
        Silently skips registration if the derived hash ID already exists.

        Parameters
        ----------
        sample : AudioSample
            The AudioSample DTO whose metadata will be registered.

        Returns
        -------
        str
            The generated or retrieved unique `audio_id` for the sample.
        """
        audio_id = self.generate_hash_id(sample.file_name, sample.metadata)
        
        if audio_id in self.df['audio_id'].values:
            self.logger.debug(f"Sample {audio_id} already exists in catalog. Skipping.")
            return audio_id

        row_data = {
            "audio_id": audio_id,
            "file_name": sample.file_name,
            "sample_rate": sample.sample_rate,
            "file_path": str(sample.file_path.absolute()),
        }
        row_data.update(sample.metadata)

        new_row_df = pd.DataFrame([row_data])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        
        self.logger.debug(f"Registered new sample: {audio_id}")
        return audio_id

    def save_catalog(self) -> None:
        """
        Flushes the current in-memory DataFrame to the underlying CSV file.

        Returns
        -------
        None
        """
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.catalog_path, index=False)
        self.logger.info(f"Catalog saved with {len(self.df)} records.")

    def stream_samples(self, target_sr: int = 24000) -> Iterator['AudioSample']:
        """
        A memory-efficient generator that yields instantiated AudioSample objects.
        
        It retrieves items one by one directly from the physical files listed 
        in the CSV catalog, avoiding out-of-memory errors on large datasets.

        Parameters
        ----------
        target_sr : int, default=24000
            The sample rate to force upon loading the audio.

        Yields
        ------
        AudioSample
            An instantiated AudioSample ready for processing.
        """
        for _, row in self.df.iterrows():
            file_path = row['file_path']
            
            base_cols = {"audio_id", "file_name", "sample_rate", "file_path"}
            recovered_metadata = {k: v for k, v in row.items() if k not in base_cols and pd.notna(v)}
            recovered_metadata['audio_id'] = row['audio_id'] # Inject the hash ID

            sample = load_audio_sample(
                file_path=file_path,
                sample_rate=target_sr,
                metadata=recovered_metadata,
                logger=self.logger
            )
            
            if sample:
                yield sample

if __name__ == '__main__':
    pass