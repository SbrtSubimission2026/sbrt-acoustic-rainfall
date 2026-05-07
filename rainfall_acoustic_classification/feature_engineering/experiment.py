#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Experiment Creator
==========================
Handles deterministic mutations for acoustic datasets based on 
content filtering and target granularity mapping. Implements Smart Mapping 
to intelligently handle subsets ('wet', 'dry') without rigid string constraints.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-20
:Version: 2.0.0
"""

import pandas as pd
from typing import Tuple, List, Dict, Any, Union, Optional
from dataclasses import dataclass, field, fields

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("ExperimentCreator")

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration DTO for the Experiment Creator process.

    Parameters
    ----------
    content : str, default='dry/wet'
        The data subset to process. Must be 'dry/wet', 'wet', or 'dry'.
    granularity : int, default=5
        The target number of classes. Ignored if content='dry'.
    label_col : str, default='category'
        The column name containing the target labels.
    metadata_cols : Union[List[str], Tuple[str, ...], str]
        List of standard metadata columns to be detached from the feature matrix.
    custom_mapping : Dict[str, str], optional
        An explicit mapping dictionary to override the Smart Mapping engine.
    """
    content: str = 'dry/wet'
    granularity: int = 5
    label_col: str = 'category'
    
    metadata_cols: Union[List[str], Tuple[str, ...], str] = field(default_factory=lambda: [
        'file_name', 'timestamp', 'period', 'mm_5min', 'mm_hr', 'category',
        'recorder', 'location', 'file_path', 'extension', 'year', 'month',
        'day', 'hour', 'minute', 'second', 'wet', 'split', 'should_augment',
        'segment_idx', 'offset_sec', 'aug_params'
    ])
    
    custom_mapping: Optional[Dict[str, str]] = None

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'ExperimentConfig':
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        if isinstance(self.metadata_cols, str):
            object.__setattr__(self, 'metadata_cols', [self.metadata_cols])
        elif isinstance(self.metadata_cols, tuple):
            object.__setattr__(self, 'metadata_cols', list(self.metadata_cols))
            
        valid_contents = ['dry/wet', 'wet', 'dry']
        if self.content.lower() not in valid_contents:
            raise ValueError(f"Content must be one of {valid_contents}. Got: {self.content}")


class ExperimentCreator:
    """
    Data engineering boundary class. 
    Mutates DataFrames deterministically to prepare them for statistical feature selection.
    """

    @staticmethod
    def _normalize_labels(y: pd.Series) -> pd.Series:
        """
        Kills 'string picuinhas'. Converts all labels to lowercase, 
        replaces hyphens/underscores with spaces, and strips whitespace.
        E.g., 'No-Rain', 'no_rain', ' NO RAIN ' all become 'no rain'.
        """
        return y.astype(str).str.lower().str.replace('-', ' ').str.replace('_', ' ').str.strip()

    @staticmethod
    def _get_smart_mapping(content: str, granularity: int) -> Dict[str, str]:
        """
        The intelligence engine. Understands how to group rain classes 
        based strictly on the acoustic content subset and requested granularity.
        """
        c = content.lower()
        
        # Scenario 1: Only Background Noise (Anomaly Detection preparation)
        if c == 'dry':
            return {'no rain': 'No Rain'}

        # Scenario 2: Only Rain (Classification of Intensities)
        if c == 'wet':
            if granularity == 2:
                return {'light': 'Light/Moderate', 'moderate': 'Light/Moderate', 
                        'heavy': 'Heavy/Violent', 'violent': 'Heavy/Violent'}
            if granularity == 3:
                return {'light': 'Light', 'moderate': 'Moderate', 
                        'heavy': 'Heavy/Violent', 'violent': 'Heavy/Violent'}
            # For 4 or 5 granularity on 'wet', keep them separated
            return {'light': 'Light', 'moderate': 'Moderate', 'heavy': 'Heavy', 'violent': 'Violent'}

        # Scenario 3: Full Dataset (Dry vs Wet spectrum)
        if granularity == 2:
            return {'no rain': 'Dry', 'light': 'Wet', 'moderate': 'Wet', 
                    'heavy': 'Wet', 'violent': 'Wet'}
        if granularity == 3:
            return {'no rain': 'No Rain', 'light': 'Light/Moderate', 'moderate': 'Light/Moderate', 
                    'heavy': 'Heavy/Violent', 'violent': 'Heavy/Violent'}
        if granularity == 4:
            return {'no rain': 'No Rain', 'light': 'Light', 'moderate': 'Moderate', 
                    'heavy': 'Heavy/Violent', 'violent': 'Heavy/Violent'}
            
        # Default: 5 classes
        return {'no rain': 'No Rain', 'light': 'Light', 'moderate': 'Moderate', 
                'heavy': 'Heavy', 'violent': 'Violent'}

    @staticmethod
    def extract_X_y_meta(
        df: pd.DataFrame, 
        config: Optional[ExperimentConfig] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Separates the monolithic DataFrame into Features (X), Target (y), and Metadata."""
        if config is None:
            config = ExperimentConfig.from_kwargs(**kwargs)

        actual_meta_cols = [c for c in config.metadata_cols if c in df.columns]
        meta_df = df[actual_meta_cols].copy()
        y = df[config.label_col].copy()
        X = df.drop(columns=actual_meta_cols).copy()
        
        return X, y, meta_df

    @staticmethod
    def apply_experiment_rules(
        X: pd.DataFrame, 
        y: pd.Series, 
        config: Optional[ExperimentConfig] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Applies Content filtering and Granularity mapping safely.
        """
        if config is None:
            config = ExperimentConfig.from_kwargs(**kwargs)

        logger.info(f"Applying Smart Rules -> Content: '{config.content}' | Granularity: {config.granularity}")
        
        # 1. Normalize String Variations
        y_norm = ExperimentCreator._normalize_labels(y)

        # 2. Content Mutation Filter
        if config.content == 'wet':
            mask = y_norm != 'no rain'
        elif config.content == 'dry':
            mask = y_norm == 'no rain'
        else:  # 'dry/wet'
            mask = pd.Series(True, index=y_norm.index)

        X_mut = X[mask].copy()
        y_mut = y_norm[mask].copy()

        # 3. Resolve Mapping Strategy
        if config.custom_mapping:
            # Normalize custom mapping keys to match our normalized y_mut
            active_map = {k.lower().replace('-', ' ').replace('_', ' ').strip(): v 
                          for k, v in config.custom_mapping.items()}
        else:
            active_map = ExperimentCreator._get_smart_mapping(config.content, config.granularity)

        # 4. Apply Mapping
        y_mapped = y_mut.map(active_map)

        if y_mapped.isnull().any():
            missing_keys = y_mut[y_mapped.isnull()].unique()
            logger.warning(f"Unmapped acoustic classes detected: {missing_keys}. They will become NaN.")

        # 5. Sanity Check (Warnings instead of crashes)
        max_classes = y_mapped.nunique()
        if config.content != 'dry' and config.granularity > max_classes:
            logger.warning(
                f"Requested granularity ({config.granularity}) is larger than the unique "
                f"classes physically available in this subset ({max_classes})."
            )

        return X_mut, y_mapped

if __name__ == '__main__':
    pass