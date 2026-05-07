#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Processing
==================
Provides isolated workers and configurations for audio augmentation, 
segmentation, and acoustic metric extraction.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 1.0.0
:License: Private (Internal Use Only)
"""

from .augmenter import AudioAugmenter, AugmenterConfig
from .segmenter import AudioSegmenter, SegmenterConfig
from .acoustic_metrics import AcousticMetrics, AcousticMetricsConfig

__all__ = [
    "AudioAugmenter",
    "AugmenterConfig",
    "AudioSegmenter",
    "SegmenterConfig",
    "AcousticMetrics",
    "AcousticMetricsConfig"
]