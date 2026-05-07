#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Visualization
=====================
Provides a standardized engine and plotting strategies for generating 
scientific figures with consistent IEEE styling and high-fidelity exports.

Metadata
--------
:Author: Giovanni G. R. Milan
:Version: 1.0.0
:License: Private (Internal Use Only)
"""

from .engine import (
    VisualizationEngine,
    VisualizationConfig
)

from .plots import (
    plot_styled_donut, 
    plot_styled_stacked_bar, 
    plot_temporal_log_distribution, 
    plot_custom_matrix
)
__all__ = [
    "VisualizationEngine",
    "VisualizationConfig",
    "plot_styled_donut", 
    "plot_styled_stacked_bar", 
    "plot_temporal_log_distribution", 
    "plot_custom_matrix"
]