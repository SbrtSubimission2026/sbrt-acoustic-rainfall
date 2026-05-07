#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Model Evaluation Plots
==============================
Provides static visualizations for the classification pipeline. 
Tailored for multiclass imbalance, Precision-Recall evaluation, 
and Cross-Domain experiment matrices.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-19
:Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from rainfall_acoustic_classification.utils import get_standard_logger

logger = get_standard_logger("ModelingPlotss")


def plot_confusion_matrix_grid(
    cm_raw: np.ndarray,
    cm_norm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Versão Final para Tese: Compacta, Cores Navy e Darkred.
    Eixo X: Nomes em BAIXO, Label (Predicted) no TOPO.
    Eixo Y: Nomes na ESQUERDA, Label (True) na DIREITA.
    Sem negrito, sem símbolo de %.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Paletas solicitadas
    cmap_raw = sns.light_palette("navy", as_cmap=True)
    cmap_norm = sns.light_palette("darkred", as_cmap=True)
    
    def _style_axis(ax, label_x, label_y, current_title):
        # Eixo X (Predicted): Ticks (nomes das classes) em BAIXO
        ax.xaxis.tick_bottom()
        # Rótulo ("Predicted Label") no TOPO
        ax.xaxis.set_label_position('top')
        
        # Eixo Y (True): Nomes na ESQUERDA, Rótulo (Label) na DIREITA
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position('right')
        
        ax.set_xlabel(label_x, fontsize=12, weight='normal', labelpad=15)
        ax.set_ylabel(label_y, fontsize=12, weight='normal', labelpad=15, rotation=270)
        
        # Título um pouco mais alto (pad=30) para não colidir com o "Predicted Label"
        ax.set_title(current_title, fontsize=13, weight='normal', pad=30)
        
        # Ajuste dos Ticks
        ax.set_xticklabels(classes, fontsize=11, rotation=45, ha='right', weight='normal')
        ax.set_yticklabels(classes, fontsize=11, rotation=0, weight='normal')

    # --- PLOT 1: RAW COUNTS (NAVY BLUE) ---
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap=cmap_raw, 
                cbar=False, square=True, 
                annot_kws={"size": 13, "weight": "normal"}, ax=ax1)
    
    _style_axis(ax1, "Predicted Label", "True Label", f"{title}\n(Absolute Counts)")
    plt.tight_layout(pad=0.5)
    
    if save_path:
        path_raw = save_path.replace(".png", "_raw.png")
        fig1.savefig(path_raw, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # --- PLOT 2: NORMALIZED (DARK RED) ---
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    annot_norm = np.array([[f"{val:.2f}" for val in row] for row in cm_norm])
    
    sns.heatmap(cm_norm, annot=annot_norm, fmt='', cmap=cmap_norm, 
                cbar=False, square=True, 
                annot_kws={"size": 11, "weight": "normal"}, ax=ax2)
    
    _style_axis(ax2, "Predicted Label", "True Label", f"{title}\n(Row Normalized)")
    plt.tight_layout(pad=0.5)
    
    if save_path:
        path_norm = save_path.replace(".png", "_norm.png")
        fig2.savefig(path_norm, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)


def plot_multiclass_pr_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    color_map: Dict[str, str],
    class_order: List[str],
    title: str = "Multiclass Precision-Recall Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plots the Precision-Recall curve for each class using One-vs-Rest (OVR), 
    essential for evaluating models on highly imbalanced target spaces.

    Parameters
    ----------
    y_true : pd.Series
        The actual target labels.
    y_proba : np.ndarray
        The probability predictions from the model (from `.predict_proba()`).
    color_map : Dict[str, str]
        Mapping of class names to hexadecimal color codes (from engine.py).
    class_order : List[str]
        The authoritative order of classes (from engine.py).
    title : str, default="Multiclass Precision-Recall Curves"
        The title of the plot.
    save_path : str, optional
        If provided, saves the figure to this filepath.
    """
    logger.info("Generating Multiclass Precision-Recall Curves...")
    
    n_classes = len(class_order)
    y_bin = label_binarize(y_true, classes=class_order)
    
    # Handle the binary case shape discrepancy
    if n_classes == 2:
        y_bin = np.hstack((1 - y_bin, y_bin))
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each class curve
    for i, class_name in enumerate(class_order):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = average_precision_score(y_bin[:, i], y_proba[:, i])
        
        color = color_map.get(class_name, 'black')
        ax.plot(
            recall, precision, color=color, lw=2,
            label=f'PR curve of class {class_name} (area = {pr_auc:0.3f})'
        )
        
    # Baseline for random guessing is the fraction of positives (can't plot universally for multiclass easily)
    # So we plot the Macro-average manually if desired, but individual curves are stronger.
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR-Curve saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_experiment_performance_heatmap(
    results_df: pd.DataFrame,
    metric_col: str = 'f1_macro',
    title: str = "Algorithm vs. Granularity Performance",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a heatmap showing performance across algorithms and target granularities.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: 'Algorithm' (e.g., 'RF', 'SGD'), 
        'Granularity' (e.g., '2-Class', '3-Class'), and the `metric_col`.
    metric_col : str, default='f1_macro'
        The metric to plot as color intensity.
    title : str, default="Algorithm vs. Granularity Performance"
        The title of the plot.
    save_path : str, optional
        If provided, saves the figure to this filepath.
    """
    logger.info("Generating Experiment Matrix Heatmap...")
    
    pivot_df = results_df.pivot(index="Algorithm", columns="Granularity", values=metric_col)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", 
        cbar_kws={'label': metric_col.upper()}, ax=ax
    )
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Target Granularity", fontsize=12)
    ax.set_ylabel("Algorithm Architecture", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_domain_shift_bar(
    shift_df: pd.DataFrame,
    metric_col: str = 'f1_macro',
    title: str = "Cross-Domain Shift Resilience",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a grouped bar chart comparing In-Domain vs Cross-Domain performance.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Must contain columns: 'Algorithm', 'Domain' (e.g., 'UECE', 'IDSM'), 
        and the `metric_col`.
    metric_col : str, default='f1_macro'
        The metric to plot on the Y-axis.
    title : str, default="Cross-Domain Shift Resilience"
        The title of the plot.
    save_path : str, optional
        If provided, saves the figure to this filepath.
    """
    logger.info("Generating Domain Shift Resilience Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=shift_df, x='Algorithm', y=metric_col, hue='Domain', 
        palette="muted", ax=ax
    )
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(metric_col.upper(), fontsize=12)
    ax.set_xlabel("Algorithm Architecture", fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add numerical labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
        
    ax.legend(title='Evaluation Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Domain shift plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    pass