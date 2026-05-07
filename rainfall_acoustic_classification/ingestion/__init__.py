#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Ingestion
=================
Handles scanning, metadata extraction, ground-truth merging, curation, 
and dataset splitting for the acoustic pipeline.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 1.0.0
:License: Private (Internal Use Only)
"""

from .metadata_ingestion import (
    scan_files,
    extract_filename_metadata,
    merge_ground_truth,
    sanitize_merged_data
)

from .dataset_builder import (
    discard_files,
    undersample_1_to_1,
    split_dataset
)

__all__ = [
    "scan_files",
    "extract_filename_metadata",
    "merge_ground_truth",
    "sanitize_merged_data",
    "discard_files",
    "undersample_1_to_1",
    "split_dataset"
]