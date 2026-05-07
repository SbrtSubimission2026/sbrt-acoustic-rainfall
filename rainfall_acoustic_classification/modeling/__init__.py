#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Modeling
================
Public API for machine learning classification, hyperparameter tuning, 
robust validation (including cross-domain evaluation), experiment tracking, 
and publication-ready plotting.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

from .classifiers import (
    ClassifierConfig,
    ClassifierFactory
)

from .tuning import (
    TuningConfig,
    ModelOptimizer
)

from .validation import (
    ValidationConfig,
    ModelEvaluator
)

from .plots import (
    plot_confusion_matrix_grid,
    plot_multiclass_pr_curve,
    plot_experiment_performance_heatmap,
    plot_domain_shift_bar
)

from .tracking import (
    ExperimentTracker
)

__all__ = [
    # Classifiers Factory
    "ClassifierConfig",
    "ClassifierFactory",
    
    # Hyperparameter Tuning
    "TuningConfig",
    "ModelOptimizer",
    
    # Stress Testing & Validation
    "ValidationConfig",
    "ModelEvaluator",
    
    # Visualization Engine
    "plot_confusion_matrix_grid",
    "plot_multiclass_pr_curve",
    "plot_experiment_performance_heatmap",
    "plot_domain_shift_bar",
    
    # MLOps & Tracking
    "ExperimentTracker"
]