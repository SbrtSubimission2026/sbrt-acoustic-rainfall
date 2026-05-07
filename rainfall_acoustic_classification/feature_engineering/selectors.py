#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Feature Selectors
=========================
Contains scikit-learn compatible pipelines for statistical feature reduction,
normalization, Fisher Scoring (ANOVA), and baseline isolation.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-20
:Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, fields
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("FeatureSelector")

# ==========================================
# CONFIGURATION DTOs
# ==========================================

@dataclass(frozen=True)
class VectorSelectorConfig:
    """
    Configuration DTO for the comprehensive Feature Selection pipeline.
    
    Parameters
    ----------
    corr_threshold : float, default=0.85
        Pearson threshold for collinearity removal.
    fisher_percentile: int, default=60

    rf_n_estimators : int, default=100
        Number of trees for the Gini Impurity evaluation.
    rf_max_depth : int, optional, default=None
        Maximum depth of the trees. Limits overfitting during selection.
    rf_min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    rf_class_weight : str or dict, optional, default='balanced'
        Weights associated with classes to handle severe imbalance.
    random_state : int, default=42
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of CPU cores to use.
    """
    corr_threshold: float = 0.85
    fisher_percentile: int = 60
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_class_weight: Optional[str] = 'balanced'
    random_state: int = 42
    n_jobs: int = -1

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'VectorSelectorConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        if not (0.0 <= self.corr_threshold <= 1.0):
            raise ValueError(f"corr_threshold must be in [0.0, 1.0]. Got: {self.corr_threshold}")
        if self.rf_n_estimators < 1:
            raise ValueError(f"rf_n_estimators must be >= 1. Got: {self.rf_n_estimators}")
        if self.rf_min_samples_split < 2:
            raise ValueError(f"rf_min_samples_split must be >= 2. Got: {self.rf_min_samples_split}")
        if not (1 <= self.fisher_percentile <= 100):
            raise ValueError("fisher_percentile must be between 1 and 100.")

@dataclass(frozen=True)
class SingleSelectorConfig:
    """
    Configuration DTO for the baseline Single Feature pipeline.
    """
    target_feature: str = 'psd_mean'
    fisher_percentile: int = 60

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'SingleSelectorConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        if not isinstance(self.target_feature, str) or not self.target_feature.strip():
            raise ValueError("target_feature must be a valid, non-empty string.")
        if not (1 <= self.fisher_percentile <= 100):
            raise ValueError("fisher_percentile must be between 1 and 100.")

# ==========================================
# SCIKIT-LEARN TRANSFORMERS
# ==========================================

class SafeImputer(BaseEstimator, TransformerMixin):
    """
    Imputer blindado. Força dados para numérico float64 (C-level), 
    higieniza infinitos, preenche com a mediana e evita bugs de tipagem do Scikit-Learn.
    """
    def fit(self, X: pd.DataFrame, y=None) -> 'SafeImputer':
        self.medians_ = {}
        self.valid_cols_ = []
        X_work = X.copy()

        for col in X_work.columns:
            if X_work[col].isna().all():
                continue # Ejeta sumariamente colunas 100% vazias
            self.valid_cols_.append(col)
            self.medians_[col] = X_work[col].median()
            
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X[self.valid_cols_].copy()
        
        X_out = X_out.replace([np.inf, -np.inf], np.nan)

        for col in self.valid_cols_:
            X_out[col] = X_out[col].fillna(self.medians_[col])
            
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.valid_cols_)

class SingleFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer designed for baseline evaluations. 
    Strips the entire feature space down to a single specified metric.
    """
    def __init__(self, target_feature: str):
        self.target_feature = target_feature

    def fit(self, X: pd.DataFrame, y=None) -> 'SingleFeatureSelector':
        # Rigorous check during fit to prevent silent downstream crashes
        if self.target_feature not in X.columns:
            logger.error(f"Feature '{self.target_feature}' is missing. Available: {X.columns.tolist()}")
            raise ValueError(f"Target feature '{self.target_feature}' not found in DataFrame.")

        if y is not None:
            try:
                # O Pulo do Gato: Calcula o Fisher Score apenas para logar no terminal
                f_val, _ = f_classif(X[[self.target_feature]], y)
                logger.info(f"Isolated '{self.target_feature}' | Fisher Score (F-Value): {f_val[0]:.2f}")
            except Exception as e:
                logger.warning(f"Isolated '{self.target_feature}' | Fisher calculation failed: {e}")
        else:
            logger.info(f"SingleFeatureSelector locked onto: {self.target_feature}")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.target_feature not in X.columns:
            raise KeyError(f"Feature '{self.target_feature}' vanished during transform step.")
        return X[[self.target_feature]].copy()


class CollinearityFilter(BaseEstimator, TransformerMixin):
    """
    Eliminates highly correlated acoustic metrics.
    Operates on scaled or unscaled data seamlessly (Pearson is scale-invariant).
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X: pd.DataFrame, y=None) -> 'CollinearityFilter':
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        logger.info(f"Collinearity filter established. Dropping {len(self.to_drop_)} features.")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.drop(columns=[col for col in self.to_drop_ if col in X.columns])

# ==========================================
# PIPELINE BUILDERS
# ==========================================

def build_single_selector(config: Optional[SingleSelectorConfig] = None, **kwargs: Any) -> Pipeline:
    """
    Builds a standardized pipeline that isolates a single baseline metric.
    Includes robust scaling for downstream linear models.
    """
    if config is None:
        config = SingleSelectorConfig.from_kwargs(**kwargs)
        
    return Pipeline([
        ('safe_imputer', SafeImputer()),
        ('isolate_feature', SingleFeatureSelector(target_feature=config.target_feature)),
        ('scaler', StandardScaler())  # Crucial for linear models even on single features
    ])


class VectorSelector(BaseEstimator, TransformerMixin):
    """
    Encapsula o pipeline de seleção de features de ponta a ponta.
    Garante a extração de métricas internas (Fisher, Gini) e a rastreabilidade 
    dos nomes das colunas de forma autônoma e orientada a objetos.
    """
    def __init__(self, config: Optional['VectorSelectorConfig'] = None, **kwargs: Any):
        self.config = config if config is not None else VectorSelectorConfig.from_kwargs(**kwargs)
        self.pipeline_ = None
        self.original_features_ = None
        self.feature_metrics_df_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.original_features_ = X.columns.tolist()
        '''
        rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight=self.config.rf_class_weight,
            random_state=self.config.random_state, 
            n_jobs=self.config.n_jobs
        )
        '''
        self.pipeline_ = Pipeline(steps=[
            ('safe_imputer', SafeImputer()),
            ('fisher_score', SelectPercentile(score_func=f_classif, percentile=self.config.fisher_percentile)),
            # ('zero_variance', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler()),
            # ('multicollinearity', CollinearityFilter(threshold=self.config.corr_threshold)),
            # ('rf_gini', SelectFromModel(rf, threshold='mean'))
        ])
        
        self.pipeline_.fit(X, y)
        
        self._compile_metrics()
        
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        features = self.original_features_
        for name, step in self.pipeline_.steps:
            if hasattr(step, 'get_feature_names_out'):
                features = list(step.get_feature_names_out(features))
        return np.array(features)

    def _compile_metrics(self) -> None:
        current_features = self.original_features_
        fisher_dict = {}
        gini_dict = {}

        for name, step in self.pipeline_.steps:
            if hasattr(step, 'scores_'):
                fisher_dict = dict(zip(current_features, np.nan_to_num(step.scores_)))
            
            if hasattr(step, 'estimator_') and hasattr(step.estimator_, 'feature_importances_'):
                gini_dict = dict(zip(current_features, step.estimator_.feature_importances_))

            if hasattr(step, 'get_feature_names_out'):
                current_features = list(step.get_feature_names_out(current_features))

        df = pd.DataFrame({'Feature_Name': self.original_features_})
        df['Fisher_Score'] = df['Feature_Name'].map(fisher_dict)
        df['Gini_Importance'] = df['Feature_Name'].map(gini_dict)
        df['Survived_Pipeline'] = df['Feature_Name'].isin(current_features).astype(int)

        self.feature_metrics_df_ = df.sort_values(by='Fisher_Score', ascending=False)

    def get_metrics_report(self) -> pd.DataFrame:
        if self.feature_metrics_df_ is None:
            raise ValueError("O seletor ainda não foi treinado. Chame .fit() primeiro.")
        return self.feature_metrics_df_

if __name__ == '__main__':
    pass