#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Classifiers Factory
===========================
Provides a centralized, extensible factory using the Registry Pattern for 
instantiating machine learning classifiers. Enforces strict handling of 
class imbalance and computational resource limits.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

import os
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, fields
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("ClassifierFactory")

@dataclass(frozen=True)
class ClassifierConfig:
    """
    Configuration DTO for Classifier instantiation.

    Parameters
    ----------
    model_type : str, default='rf'
        The type of model to instantiate (e.g., 'lr', 'sgd', 'rf', 'xgb').
    random_state : int, default=42
        Seed for reproducibility across identical runs.
    n_jobs : int, default=-1
        Number of CPU cores to utilize. Will be scaled automatically.
    model_hyperparams : Dict[str, Any], optional
        Additional hyper-parameters to pass directly to the model's constructor.
    """
    model_type: str = 'rf'
    random_state: int = 42
    n_jobs: int = -1
    model_hyperparams: Optional[Dict[str, Any]] = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'ClassifierConfig':
        """
        Instantiates the configuration safely from a pool of arbitrary kwargs.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary keyword arguments.

        Returns
        -------
        ClassifierConfig
            An immutable instance of the classifier configuration.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """
        Normalizes mutable defaults, scales CPU usage safely, and enforces 
        domain boundaries after initialization.

        Raises
        ------
        ValueError
            If `n_jobs` is zero or less than -1.
        """
        # 1. Normalize Hyperparameters
        if self.model_hyperparams is None:
            object.__setattr__(self, 'model_hyperparams', {})
            
        # 2. CPU Scaling Protection (Reserve 1 core for the OS minimum)
        max_cores = os.cpu_count() or 2
        if self.n_jobs == -1:
            reserved_cores = 2 if max_cores >= 16 else 1
            safe_cores = max(1, max_cores - reserved_cores)
        else:
            safe_cores = min(max(1, self.n_jobs), max_cores)
            
        logger.info(f"CPU Scaling: Requested {self.n_jobs} -> Allocated {safe_cores} cores.")
        object.__setattr__(self, 'n_jobs', safe_cores)


class ClassifierFactory:
    """
    Registry for Machine Learning models. 
    Implements the Open/Closed Principle via Decorators.
    """
    _registry: Dict[str, Callable[[ClassifierConfig], BaseEstimator]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """
        Decorator to register a new model builder function.

        Parameters
        ----------
        model_name : str
            The string identifier for the model (e.g., 'rf').

        Returns
        -------
        Callable
            The decorator function.
        """
        def wrapper(builder_func: Callable[[ClassifierConfig], BaseEstimator]) -> Callable:
            cls._registry[model_name.lower()] = builder_func
            return builder_func
        return wrapper

    @classmethod
    def build(cls, config: Optional[ClassifierConfig] = None, **kwargs: Any) -> BaseEstimator:
        """
        Builds the requested classifier by routing to the registered builder.

        Parameters
        ----------
        config : ClassifierConfig, optional
            The configuration object. If None, it is built from kwargs.
        **kwargs : Any
            Keyword arguments to construct the configuration on-the-fly.

        Returns
        -------
        BaseEstimator
            An un-fitted, scikit-learn compatible model instance.

        Raises
        ------
        ValueError
            If the requested `model_type` is not found in the registry.
        """
        if config is None:
            config = ClassifierConfig.from_kwargs(**kwargs)
            
        m_type = config.model_type.lower()
        if m_type not in cls._registry:
            raise ValueError(
                f"Model type '{m_type}' is not registered. "
                f"Available models: {list(cls._registry.keys())}"
            )
            
        logger.info(f"Building registered model architecture: {m_type.upper()}")
        return cls._registry[m_type](config)


# =============================================================================
# MODEL BUILDERS (Registered dynamically)
# =============================================================================

@ClassifierFactory.register('lr')
def _build_logistic_regression(config: ClassifierConfig) -> BaseEstimator:
    """
    Builds a Logistic Regression classifier.

    Parameters
    ----------
    config : ClassifierConfig
        The configuration containing hyper-parameters and CPU limits.

    Returns
    -------
    BaseEstimator
        The configured LogisticRegression instance.
    """
    params = {
        'random_state': config.random_state, 
        'max_iter': 2000,
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    
    return LogisticRegression(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('sgd')
def _build_sgd(config: ClassifierConfig) -> BaseEstimator:
    """
    Builds a Stochastic Gradient Descent classifier.

    Parameters
    ----------
    config : ClassifierConfig
        The configuration containing hyper-parameters and CPU limits.

    Returns
    -------
    BaseEstimator
        The configured SGDClassifier instance.
    """
    params = {
        'random_state': config.random_state,
        'loss': 'log_loss',
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    
    return SGDClassifier(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('rf')
def _build_random_forest(config: ClassifierConfig) -> BaseEstimator:
    """
    Builds a Random Forest classifier.

    Parameters
    ----------
    config : ClassifierConfig
        The configuration containing hyper-parameters and CPU limits.

    Returns
    -------
    BaseEstimator
        The configured RandomForestClassifier instance.
    """
    params = {
        'random_state': config.random_state,
        'class_weight': 'balanced'
    }
    params.update(config.model_hyperparams)
    
    return RandomForestClassifier(n_jobs=config.n_jobs, **params)


@ClassifierFactory.register('xgb')
def _build_xgboost(config: ClassifierConfig) -> BaseEstimator:
    """
    Builds an XGBoost classifier, handling missing library dependencies gracefully.

    Parameters
    ----------
    config : ClassifierConfig
        The configuration containing hyper-parameters and CPU limits.

    Returns
    -------
    BaseEstimator
        The configured XGBClassifier instance.

    Raises
    ------
    ImportError
        If the xgboost package is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost is not installed. Run: pip install xgboost")
        raise ImportError("Missing required dependency: xgboost")

    params = {
        'random_state': config.random_state,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss'
    }
    params.update(config.model_hyperparams)
    
    return xgb.XGBClassifier(n_jobs=config.n_jobs, **params)

@ClassifierFactory.register("nusvc")
def build_nusvc(config: ClassifierConfig) -> NuSVC:
    """
    Constrói o classificador NuSVC seguindo as configurações do projeto.
    """
    return NuSVC(
        nu=getattr(config, "nu", 0.5),              
        kernel=getattr(config, "kernel", "linear"),    # 'linear', 'poly', 'rbf', 'sigmoid'
        gamma=getattr(config, "gamma", "scale"),    # 'scale' or 'auto'
        probability=True,
        class_weight=getattr(config, "class_weight", "balanced"),
        random_state=config.random_state,
        cache_size=getattr(config, "cache_size", 500)
    )


if __name__ == '__main__':
    pass