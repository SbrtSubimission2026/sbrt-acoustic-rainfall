#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Model Validation
========================
Provides rigorous evaluation metrics for highly imbalanced acoustic data.
Calculates Macro F1-Score, Multiclass Precision-Recall AUC (PR-AUC), 
and orchestrates Cross-Domain stress testing.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from dataclasses import dataclass, fields
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score, 
    average_precision_score, 
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("ModelEvaluator")

@dataclass(frozen=True)
class ValidationConfig:
    """
    Configuration DTO for the Validation and Evaluation process.

    Parameters
    ----------
    average_method : str, default='macro'
        The averaging strategy for multiclass metrics. 'macro' is enforced
        to treat minority classes (e.g., Violent Rain) equally to majority ones.
    return_report_dict : bool, default=True
        Whether to return the classification report as a parsed dictionary 
        rather than a formatted string (ideal for programmatic extraction).
    """
    average_method: str = 'macro'
    return_report_dict: bool = True

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'ValidationConfig':
        """Instantiates the configuration safely from kwargs."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """Enforces domain boundaries for evaluation metrics."""
        valid_averages = {'macro', 'micro', 'weighted'}
        if self.average_method not in valid_averages:
            raise ValueError(f"average_method must be one of {valid_averages}")


class ModelEvaluator:
    """
    Handles the rigorous extraction of metrics from champion models, 
    specifically tailored to survive extreme class imbalances and 
    Cross-Domain acoustic shifts.
    """

    @staticmethod
    def _calculate_multiclass_pr_auc(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        classes: np.ndarray, 
        average: str
    ) -> float:
        """
        Safely calculates the Precision-Recall AUC for both binary and 
        multiclass vectors using One-vs-Rest (OVR) binarization.
        """
        # Binary Classification Edge Case
        if len(classes) == 2:
            # Assumes y_proba has shape (n_samples, 2), we extract the positive class
            y_true_binary = (y_true == classes[1]).astype(int)
            return average_precision_score(y_true_binary, y_proba[:, 1], average=average)
            
        # Multiclass Classification (OVR)
        y_bin = label_binarize(y_true, classes=classes)
        return average_precision_score(y_bin, y_proba, average=average)

    @staticmethod
    def evaluate(
        model: BaseEstimator, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        config: Optional[ValidationConfig] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive evaluation of a fitted model against an 
        unseen dataset (e.g., Target Domain / Cross-Domain).

        Parameters
        ----------
        model : BaseEstimator
            A fitted model architecture.
        X_test : pd.DataFrame
            The unseen feature matrix (e.g., IDSM data or Holdout Test Set).
        y_test : pd.Series
            The unseen target labels.
        config : ValidationConfig, optional
            The evaluation configuration.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing F1-Score, PR-AUC, Classification Report, 
            and Confusion Matrices (Raw and Normalized).
        """
        if config is None:
            config = ValidationConfig.from_kwargs(**kwargs)
            
        logger.info("Executing Rigorous Stress Test (Model Evaluation)...")
        
        # 1. Hard Predictions (For F1 and Confusion Matrix)
        y_pred = model.predict(X_test)
        classes = model.classes_
        
        # 2. Soft Predictions (For PR-AUC)
        y_proba = None
        pr_auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            pr_auc = ModelEvaluator._calculate_multiclass_pr_auc(
                y_test.values, y_proba, classes, config.average_method
            )
        else:
            logger.warning("Champion model lacks 'predict_proba'. PR-AUC computation skipped.")

        # 3. Core Metrics Compilation
        f1_val = f1_score(y_test, y_pred, average=config.average_method)
        
        conf_matrix_raw = confusion_matrix(y_test, y_pred, labels=classes)
        conf_matrix_norm = confusion_matrix(y_test, y_pred, labels=classes, normalize='true')
        
        class_report = classification_report(
            y_test, y_pred, labels=classes, output_dict=config.return_report_dict, zero_division=0
        )
        
        logger.info(f"Test Results -> F1-{config.average_method.capitalize()}: {f1_val:.4f} | "
                    f"PR-AUC: {pr_auc if pr_auc else 'N/A'}")

        return {
            'classes': classes,
            'y_pred': y_pred,
            'y_proba': y_proba,
            f'f1_{config.average_method}': f1_val,
            f'pr_auc_{config.average_method}': pr_auc,
            'confusion_matrix_raw': conf_matrix_raw,
            'confusion_matrix_normalized': conf_matrix_norm,
            'classification_report': class_report
        }

if __name__ == '__main__':
    pass