#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Hyperparameter Tuning
=============================
Handles the grid search optimization process. Enforces Macro F1-Score 
optimization and mandates explicit validation sets (Holdout) to evaluate
hyperparameters without cross-contamination.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict
from dataclasses import dataclass, fields
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("HyperparameterTuner")

@dataclass(frozen=True)
class TuningConfig:
    """
    Configuration DTO for the Model Optimizer.

    Parameters
    ----------
    param_grid : Dict[str, list]
        Dictionary with parameters names as keys and lists of parameter 
        settings to try as values.
    scoring_metric : str, default='f1_macro'
        The rigorous metric used to evaluate the predictions on the validation set.
    n_jobs : int, default=-1
        Number of jobs to run in parallel during the grid search.
    """
    param_grid: Dict[str, list]
    scoring_metric: str = 'f1_macro'
    n_jobs: int = -1

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'TuningConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """
        Validates the fundamental integrity of the parameters.
        """
        if not isinstance(self.param_grid, dict) or not self.param_grid:
            raise ValueError("A valid, non-empty param_grid dictionary must be provided.")
            
        if not isinstance(self.scoring_metric, str) or not self.scoring_metric.strip():
            raise ValueError("scoring_metric must be a valid string (e.g., 'f1_macro').")

        # CPU Scaling Protection
        max_cores = os.cpu_count() or 2
        if self.n_jobs == -1:
            reserved_cores = 2 if max_cores >= 16 else 1
            safe_cores = max(1, max_cores - reserved_cores)
        else:
            safe_cores = min(max(1, self.n_jobs), max_cores)
            
        object.__setattr__(self, 'n_jobs', safe_cores)


class ModelOptimizer:
    """
    Orchestrates the optimization of estimator hyperparameters.
    Strictly mandates an explicit Validation Set to preserve data curation 
    (e.g., Data Augmentation on train set only).
    """

    @staticmethod
    def optimize(
        estimator: BaseEstimator, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: TuningConfig = None,
        **kwargs: Any
    ) -> BaseEstimator:
        """
        Executes Grid Search optimization strictly using the validation set.

        Parameters
        ----------
        estimator : BaseEstimator
            The un-fitted model architecture (e.g., from ClassifierFactory).
        X_train : pd.DataFrame
            The training feature matrix (potentially augmented).
        y_train : pd.Series
            The training target labels.
        X_val : pd.DataFrame
            The explicit validation feature matrix (original distribution).
        y_val : pd.Series
            The explicit validation target labels.
        config : TuningConfig, optional
            The tuning configuration containing the param grid and metric.
        **kwargs : Any
            Implicit parameters to build the configuration on-the-fly.

        Returns
        -------
        BaseEstimator
            A new model instance trained EXCLUSIVELY on X_train using the 
            best hyperparameters found during the grid search.
        """
        if config is None:
            config = TuningConfig.from_kwargs(**kwargs)
            
        logger.info(f"Starting Optimization Engine. Target Metric: {config.scoring_metric}")
        
        # 1. The Scikit-Learn API PredefinedSplit workaround
        # We concatenate temporarily ONLY to satisfy the GridSearchCV API.
        X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        # test_fold: -1 = Train strictly on this; 0 = Validate strictly on this.
        test_fold = np.concatenate([
            np.full(X_train.shape[0], -1), 
            np.zeros(X_val.shape[0])
        ])
        cv_strategy = PredefinedSplit(test_fold)
        
        # 2. Setup early stopping for XGBoost (if applicable)
        fit_params = {}
        if type(estimator).__name__ == 'XGBClassifier':
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        # 3. Executing the Grid Search
        search = GridSearchCV(
            estimator=estimator,
            param_grid=config.param_grid,
            scoring=config.scoring_metric,
            cv=cv_strategy,
            n_jobs=config.n_jobs,
            verbose=1,
            refit=False  # CRITICAL: Do NOT retrain on X_combined. Respect data augmentation.
        )
        
        search.fit(X_combined, y_combined, **fit_params)
        
        logger.info(f"Optimization complete. Champion Score ({config.scoring_metric}) on Val Set: {search.best_score_:.4f}")
        logger.info(f"Champion Hyperparameters: {search.best_params_}")
        
        # 4. The Senior Move: Safe Manual Refit
        # We extract the best parameters and train a fresh clone ONLY on the augmented Train Set.
        logger.info("Refitting the champion model exclusively on the Training Set...")
        champion_model = clone(estimator).set_params(**search.best_params_)
        champion_model.fit(X_train, y_train)
        
        return champion_model

if __name__ == '__main__':
    pass