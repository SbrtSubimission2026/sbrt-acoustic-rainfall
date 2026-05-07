#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Core
============
Provides foundational data structures, I/O operations, and cataloging 
for acoustic samples.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 1.0.0
:License: Private (Internal Use Only)
"""

from .audio_catalog import (
    load_audio_sample,
    save_audio_sample,
    AudioSample,
    AudioCatalog
)

__all__ = [
    "load_audio_sample",
    "save_audio_sample",
    "AudioSample",
    "AudioCatalog"
]