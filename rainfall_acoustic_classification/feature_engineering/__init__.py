#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Feature Engineering
===========================
Public API for the feature engineering pipeline. Exposes data curation,
statistical feature selection, and validation plotting utilities.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-17
:Version: 1.0.0
"""

from sklearn import set_config

# Ensures all scikit-learn transformers and selectors return Pandas DataFrames by default
set_config(transform_output="pandas")

# 1. Import Domain and Experiment Logic
from .experiment import (
    ExperimentConfig, 
    ExperimentCreator
)

# 2. Import Statistical Selectors and Builders
from .selectors import (
    VectorSelectorConfig,
    SingleSelectorConfig,
    SingleFeatureSelector,
    CollinearityFilter,
    build_single_selector,
    VectorSelector,
)

# 3. Import Validation Plots
from .plots import (
    plot_correlation_heatmap,
    plot_fisher_scores,
    plot_selection_rationale,
    plot_feature_importance,
    plot_scaled_distributions,
    plot_survival_scatter_matrix,
    plot_correlation_comparison,
    plot_feature_stability,
    plot_individual_boxplots,
)

# Explicitly define the public API of this module (Capsulation)
__all__ = [
    # Experiment Boundaries
    "ExperimentConfig",
    "ExperimentCreator",
    
    # Selection Engine
    "VectorSelectorConfig",
    "SingleSelectorConfig",
    "SingleFeatureSelector",
    "CollinearityFilter",
    "build_single_selector",
    "VectorSelector",
    
    # Visualization Suite
    "plot_correlation_heatmap",
    "plot_fisher_scores",
    "plot_selection_rationale",
    "plot_feature_importance",
    "plot_scaled_distributions",
    "plot_survival_scatter_matrix",
    "plot_correlation_comparison",
    "plot_feature_stability",
    "plot_individual_boxplots",
]