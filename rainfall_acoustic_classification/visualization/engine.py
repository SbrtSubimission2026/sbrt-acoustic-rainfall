#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Visualization Engine
============================
Handles scientific data visualization adhering to IEEE publication standards.
Provides a foundational engine for consistent styling, color mapping, 
and high-fidelity file export.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-20
:Version: 2.0.0
:License: Private (Internal Use Only)
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional, Union, List
import logging

from rainfall_acoustic_classification.utils import get_standard_logger


@dataclass(frozen=True)
class VisualizationConfig:
    """
    Configuration DTO for the Visualization Engine.

    Parameters
    ----------
    output_dir : str or pathlib.Path, default="reports/figures"
        The base directory where generated figures will be saved.
    style : dict, optional
        A dictionary containing Matplotlib configuration parameters 
        compliant with publication standards. Default = IEEE.
    """
    output_dir: Union[str, Path] = "reports/figures"
    style: Dict[str, Any] = field(default_factory=lambda: {
        'pdf.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': False
    })

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'VisualizationConfig':
        """Instantiates the config by filtering only valid fields from kwargs."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, 'output_dir', Path(self.output_dir))
        except TypeError:
            pass


class VisualizationEngine:
    """
    Core engine for rendering and exporting scientific figures.

    Manages the global state of Matplotlib styles to ensure consistency 
    across all exported plots. Provides robust file handling for multiple formats.
    """
    def __init__(
        self, 
        config: Optional[VisualizationConfig] = None, 
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.config = config if config is not None else VisualizationConfig.from_kwargs(**kwargs)
        self.logger = logger or get_standard_logger("VizEngine")
        
        self.output_path = self.config.output_dir.resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Aplica o estilo globalmente no momento da instanciação
        self._apply_style()

    def _apply_style(self) -> None:
        """Applies the IEEE style configuration to Matplotlib's global rcParams."""
        plt.rcParams.update(self.config.style)
        self.logger.debug("IEEE Matplotlib style applied.")

    def get_rain_color_palette(self) -> Dict[str, str]:
        """
        Provides a standardized color mapping for rainfall categories.
        Includes base classes and dynamic mapped granularities (e.g. Dry, Wet).

        Returns
        -------
        dict
            A dictionary mapping category names to their HEX color codes.
        """
        class_all = ['No Rain', 'Light', 'Moderate', 'Heavy', 'Violent']
        plasma_palette = sns.color_palette("plasma", n_colors=5).as_hex()
        
        color_map = {cls: col for cls, col in zip(class_all, plasma_palette)}
        
        # --- Aggregated maped class support (granularity 2, 3 and 4) ---
        color_map['Dry'] = color_map['No Rain']
        color_map['Wet'] = color_map['Moderate']
        
        color_map['Light/Moderate'] = color_map['Moderate']
        color_map['Heavy/Violent'] = color_map['Violent']
        
        
        color_map['Unlabeled'] = '#999999'
        
        return color_map

    def save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = ['pdf', 'png']) -> None:
        """
        Exports a Matplotlib figure to the configured output directory.
        """
        for fmt in formats:
            clean_fmt = fmt.lstrip('.')
            path = self.output_path / f"{filename}.{clean_fmt}"
            try:
                fig.savefig(path, format=clean_fmt, dpi=self.config.style.get('figure.dpi', 300))
                self.logger.info(f"Figure saved: {path.name}")
            except Exception as e:
                self.logger.error(f"Failed to save figure {filename} as {clean_fmt}. Error: {e}")

if __name__ == '__main__':
    pass