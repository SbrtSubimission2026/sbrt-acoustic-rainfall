#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rainfall Acoustic Classification
================================
A comprehensive pipeline for environmental audio processing, dataset curation, 
data augmentation, and acoustic metric extraction for rainfall classification.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 1.0.0
:License: Private (Internal Use Only)
"""

__version__ = "1.0.0"

# 1. Utilities
from .utils import get_standard_logger

# 2. Core Data Structures
from .core import AudioCatalog, AudioSample

# 3. Ingestion & Curation
from .ingestion import (
    scan_files,
    extract_filename_metadata,
    merge_ground_truth,
    discard_files,
    undersample_1_to_1,
    split_dataset
)

# 4. Processing Workers & Configs
from .processing import (
    AcousticMetrics,
    AcousticMetricsConfig,
    AudioAugmenter,
    AugmenterConfig,
    AudioSegmenter,
    SegmenterConfig
)

# 5. Public API Declaration
__all__ = [
    "get_standard_logger",
    "AudioCatalog",
    "AudioSample",
    "scan_files",
    "extract_filename_metadata",
    "merge_ground_truth",
    "discard_files",
    "undersample_1_to_1",
    "split_dataset",
    "AcousticMetrics",
    "AcousticMetricsConfig",
    "AudioAugmenter",
    "AugmenterConfig",
    "AudioSegmenter",
    "SegmenterConfig"
]