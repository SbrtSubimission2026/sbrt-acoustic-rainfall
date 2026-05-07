#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Specialized Plots
=========================
Collection of standardized plotting functions for rainfall acoustic analysis.
Designed to be used in conjunction with the VisualizationEngine.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-15
:Version: 1.0.0
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from typing import Optional, List, Dict

from .engine import VisualizationEngine


def _get_normalized_palette(engine: VisualizationEngine, categories: List[str]) -> List[str]:
    """Helper to fetch colors robustly, regardless of title case or kebab-case."""
    color_map = engine.get_rain_color_palette()
    return [color_map.get(c, '#999999') for c in categories]


def plot_styled_donut(
    df: pd.DataFrame, 
    engine: VisualizationEngine, 
    categories: List[str], 
    title: str
) -> plt.Figure:
    """Generates a donut chart with a strict monospace tabular legend."""
    counts = df['category'].value_counts().reindex(categories).fillna(0)
    total_n = counts.sum()
    percentages = (counts / total_n) * 100
    colors = _get_normalized_palette(engine, categories)
    
    # Construção da Legenda estilo Tabela
    w_cat, w_per, w_rec = 10, 8, 8
    legend_labels = []
    for label, p, c in zip(categories, percentages, counts):
        row = f"{label.ljust(w_cat)} | {f'{p:.2f}%'.center(w_per)} | {str(int(c)).rjust(w_rec)}"
        legend_labels.append(row)

    # Renderização (Estilo já herdado globalmente pela Engine)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    wedges, _ = ax.pie(
        counts, 
        colors=colors, 
        startangle=140,
        wedgeprops={'width': 0.35, 'edgecolor': 'white', 'linewidth': 1.2},
        pctdistance=0.85
    )

    ax.text(0, 0, f"Total Data\n{int(total_n):,} rec", 
            ha='center', va='center', fontsize=18)

    ax.legend(
        wedges, 
        legend_labels,
        title=f"{'Category'.ljust(w_cat)} | {'Percent'.center(w_per)} | {'Recordings'.rjust(w_rec)}",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05), 
        frameon=True,
        prop={'family': 'monospace', 'size': 10}, 
        ncol=1,
        labelspacing=0.6
    )

    ax.set_title(title, fontsize=18, pad=5)
    plt.subplots_adjust(top=0.7, bottom=0.08)
    
    return fig


def plot_styled_stacked_bar(
    df: pd.DataFrame, 
    engine: VisualizationEngine, 
    categories: List[str], 
    title: str
) -> plt.Figure:
    """Generates an ultra-compact stacked horizontal bar with inline percentages."""
    counts = df['category'].value_counts().reindex(categories).fillna(0)
    total_n = counts.sum()
    percentages = (counts / total_n) * 100
    colors = _get_normalized_palette(engine, categories)
    
    w_cat, w_per, w_rec = 11, 7, 8
    legend_labels = []
    for label, p, c in zip(categories, percentages, counts):
        row = f"{label.ljust(w_cat)} | {f'{p:.1f}%'.rjust(w_per)} | {str(int(c)).rjust(w_rec)}"
        legend_labels.append(row)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    
    left_acumulado = 0
    bars = []
    for i, (cat, prop, color) in enumerate(zip(categories, percentages, colors)):
        b = ax.barh(y=0, width=prop, left=left_acumulado, color=color, 
                    edgecolor='white', height=0.5, linewidth=1.2)
        bars.append(b[0])
        
        if prop > 5:
            text_color = 'white' if i < 2 else 'black'
            if cat == 'Unlabeled': text_color = 'black'
            ax.text(left_acumulado + prop/2, 0, f"{prop:.1f}%", 
                    ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
        left_acumulado += prop

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    leg = ax.legend(
        bars, 
        legend_labels,
        title=f"{'Category'.ljust(w_cat)} | {'Percent'.center(w_per)} | {'Records'.rjust(w_rec)}",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15), 
        frameon=True,
        prop={'family': 'monospace', 'size': 9}, 
        ncol=2, 
        labelspacing=0.5,
        columnspacing=2.0
    )
    leg.get_title().set_family('monospace')
    leg.get_title().set_fontsize(9)

    ax.set_title(title, fontsize=14, pad=20)
    ax.text(0.5, 1.05, f"Total Data: {int(total_n):,} recordings", 
            ha='center', va='bottom', transform=ax.transAxes, fontsize=10, style='italic')
    
    plt.subplots_adjust(top=0.8, bottom=0.3)
    return fig


def plot_temporal_log_distribution(
    df: pd.DataFrame, 
    engine: VisualizationEngine,
    title: str = "Temporal Rainfall Distribution (Log Scale)"
) -> plt.Figure:
    """Generates a horizontal bar chart of temporal distribution on a logarithmic scale."""
    # Garante a extração exata das categorias presentes nas colunas
    monthly_data = pd.crosstab(df['month_label'], df['category'], dropna=False)
    colors = _get_normalized_palette(engine, monthly_data.columns.tolist())
    
    fig, ax = plt.subplots(figsize=(7, 6))
    monthly_data.iloc[::-1].plot(kind='barh', ax=ax, width=0.8, color=colors)
    
    ax.set_xscale('log')
    ax.set_title(title, pad=15, fontsize=18)
    ax.set_xlabel("Quantity of Audios")
    ax.set_ylabel("Month")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    return fig


def plot_custom_matrix(
    data: pd.DataFrame, 
    title: str, 
    fmt: str = "d"
) -> plt.Figure:
    """Generates a dual-layer heatmap masking zeroes with white background/grey text."""
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = (data == 0)
    
    # Plot principal (dados > 0)
    sns.heatmap(data, annot=True, fmt=fmt, cmap="YlGnBu", mask=mask,
                linewidths=1, linecolor='#F0F0F0', cbar=False, ax=ax)
    
    # Plot dos zeros (Fundo branco, texto cinza claro)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=ListedColormap(['white']), 
                mask=~mask, cbar=False, ax=ax, 
                annot_kws={"color": "#D3D3D3"}, linewidths=1, linecolor='#F0F0F0')
    
    # Destaque das linhas de Total
    ax.axhline(data.shape[0]-1, color='black', linewidth=2)
    ax.axvline(data.shape[1]-1, color='black', linewidth=2)
    
    ax.set_title(title, pad=20)
    return fig

def plot_compact_confusion_matrix(
    cm: np.ndarray, 
    classes: List[str], 
    title: str
) -> plt.Figure:
    """
    Generates an ultra-compact confusion matrix optimized for thesis space.
    Maximizes data-ink ratio by removing colorbars, enforcing square cells,
    and enlarging font sizes.
    """
    # 1. Tamanho super reduzido da figura
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # 2. Plotagem otimizada no eixo atual (ax)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=False,           # Remove barra lateral
                square=True,          # Força quadrados perfeitos
                annot_kws={"size": 14, "weight": "bold"}, # Números gigantes
                xticklabels=classes, 
                yticklabels=classes,
                ax=ax)
    
    # 3. Maximização do espaço das Labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
    
    ax.set_ylabel('Rótulo Real', fontsize=13, weight='bold')
    ax.set_xlabel('Predição da Cascata', fontsize=13, weight='bold')
    ax.set_title(title, fontsize=12, pad=10)
    
    # 4. Corte de margens absolutas
    plt.tight_layout(pad=0.5)
    
    return fig
if __name__ == '__main__':
    pass