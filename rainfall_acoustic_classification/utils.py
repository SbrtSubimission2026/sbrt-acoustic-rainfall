#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Utilities
=================
Global utility functions and cross-cutting concerns for the 
rainfall_acoustic_classification project.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-04-10
:Version: 2.0.0
:License: Private (Internal Use Only)
:Status: Stable
"""

import sys
import logging
import multiprocessing
import pandas as pd
import math
import gc
from pathlib import Path
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Dict, Any, Optional, Union
from tqdm.auto import tqdm

def get_standard_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard library logger with uniform formatting.

    Parameters
    ----------
    name : str
        The name of the logger instance (typically the module or process name).

    Returns
    -------
    logging.Logger
        A configured standard library Logger instance outputting to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def _safe_worker_wrapper(
    task: Dict[str, Any], 
    worker_func: Callable[[Dict[str, Any]], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Executes a worker function with exception handling to prevent pool crashes.

    This internal wrapper ensures that if a single task fails (e.g., due to a 
    corrupted file), the exception is caught and an empty list is returned, 
    allowing the rest of the multiprocessing pool to continue execution.

    Parameters
    ----------
    task : dict
        A dictionary containing the data and metadata for a single task.
    worker_func : callable
        The function to process the task. It must accept a dictionary and 
        return a list of dictionaries.

    Returns
    -------
    list of dict
        A list of processed records. Returns an empty list if an exception occurs.
    """
    try:
        return worker_func(task)
    except Exception as e:
        return []


def parallel_pipe(
    df: pd.DataFrame,
    worker_func: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    n_jobs: int = -1,
    desc: str = "Parallel Processing",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Distributes a worker function across multiple CPU cores using a Map-Reduce approach.

    Employs dynamic chunking and heuristic core reservation to optimize Inter-Process 
    Communication (IPC) overhead and prevent operating system resource exhaustion. 
    Designed for native integration with `pandas.DataFrame.pipe()`. Uses Joblib (Loky) 
    backend for maximum stability on Windows memory mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset where each row represents a task to be processed.
    worker_func : callable
        A pure, top-level function that accepts a single row dictionary and returns 
        a list of output dictionaries.
    n_jobs : int, default=-1
        The number of CPU cores to allocate. If -1, applies a reservation heuristic 
        (leaves 1 core free on small machines, 2 or more on heavily-threaded servers).
    desc : str, default="Parallel Processing"
        The text description displayed on the TQDM progress bar.
    logger : logging.Logger, optional
        An optional logger instance. If None, a standard logger is created.

    Returns
    -------
    pandas.DataFrame
        A flattened DataFrame constructed from the aggregated results of all tasks.
    """
    # Fallback to print if get_standard_logger is not imported in this scope
    # Adjust this line back to your get_standard_logger if it's available.
    log = logger or logging.getLogger("ParallelPipe")
    
    if df.empty:
        log.warning("Input DataFrame is empty. Bypassing parallel execution.")
        return df

    # Core allocation heuristic to ensure OS stability
    max_cores = multiprocessing.cpu_count() or 2
    if n_jobs == -1:
        reserved_cores = 2 if max_cores >= 16 else 1
        safe_cores = max(1, max_cores - reserved_cores)
    else:
        safe_cores = min(max(1, n_jobs), max_cores)

    total_tasks = len(df)

    # Calculate optimal chunksize to minimize IPC overhead and stabilize RAM    
    optimal_chunksize = max(1, math.ceil(total_tasks / (safe_cores * 4)))
    
    log.info(f"Hardware: {max_cores} cores detected. Allocating {safe_cores} cores.")
    log.info(f"Memory Guard: Processing {total_tasks} tasks with chunksize={optimal_chunksize} (Loky Backend).")
    
    tasks = df.to_dict('records')
    
    # =======================================================
    # 1. EXECUTE MAP-REDUCE (JOBLIB)
    # batch_size substitui o chunksize para o motor loky
    # =======================================================
    raw_results = Parallel(n_jobs=safe_cores, backend='loky', batch_size=optimal_chunksize)(
        delayed(worker_func)(task) for task in tqdm(tasks, desc=desc, unit="file")
    )
    
    # =======================================================
    # 2. FLATTEN & GC FASE 1 (Limpa resultados brutos)
    # =======================================================
    all_results = []
    for result_list in raw_results:
        if result_list:
            all_results.extend(result_list)
            
    del raw_results
    gc.collect()
    
    # =======================================================
    # 3. DATAFRAME BUILD & GC FASE 2 (Limpa lista achatada)
    # =======================================================
    if not all_results:
        log.warning("Parallel pipe completed, but workers returned no data.")
        return pd.DataFrame()
        
    total_generated = len(all_results)
    
    # Cria a matriz tabular
    df_out = pd.DataFrame(all_results)
    
    # Deleta a lista de memória IMEDIATAMENTE após criar o DataFrame
    del all_results
    gc.collect()
                
    log.info(f"Parallel pipe completed. Generated {total_generated} output records.")
    return df_out


def save_checkpoint(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Salva o DataFrame no disco e devolve-o intacto para não quebrar o pipeline.
    """
    df.to_csv(file_path, **kwargs)
    return df

import pandas as pd
from typing import List

def prepare_balanced_training_set(
    df_train: pd.DataFrame, 
    rain_classes: List[str], 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Prepares the training dataset by applying intra-class balancing via data augmentation.

    This function ensures that all underrepresented rain classes are synthetically
    augmented to match the sample count of the majority rain class (N_max). The
    original data remains unaugmented, and copies are flagged for augmentation to
    achieve an equiprobable distribution across all target classes.

    Parameters
    ----------
    df_train : pandas.DataFrame
        The original training metadata containing at least the 'category' column.
    rain_classes : list of str
        A list specifying the labels of the rain classes to be balanced.
    random_state : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        The consolidated training dataset containing both the original unaugmented
        samples and the newly flagged augmented samples.
    """
    # 1. Original data always passes through "clean" (without augmentation)
    df_train['should_augment'] = False

    # 2. Isolate only the rain classes to determine the mathematical distribution
    df_rain_only = df_train[df_train['category'].isin(rain_classes)]
    counts = df_rain_only['category'].value_counts()

    # 3. Calculate N_max (the sample count of the majority class)
    n_max = counts.max()
    majority_class = counts.idxmax()

    print(" Intra-Class Balancing Strategy:")
    print(f"   -> Majority Class: '{majority_class}' with N_max = {n_max} samples.")

    augmented_dfs = []

    # 4. Generate synthetic augmentation flags for minority classes
    for cls in rain_classes:
        n_current = counts.get(cls, 0)
        deficit = n_max - n_current
        
        if deficit > 0:
            # Isolate the original samples of the minority class
            df_minority = df_train[df_train['category'] == cls]
            
            # Sample the exact amount needed to cover the deficit
            # (replace=True is essential if deficit > current original samples)
            df_aug = df_minority.sample(n=deficit, replace=True, random_state=random_state).copy()
            df_aug['should_augment'] = True
            
            augmented_dfs.append(df_aug)
            print(f"   -> [{cls}] Originals: {n_current} | Augmented generated: +{deficit} | Final Total: {n_current + deficit}")
        else:
            print(f"   -> [{cls}] Originals: {n_current} | Already at N_max. No augmentation applied.")

    # 5. Consolidate the final training dataset
    if augmented_dfs:
        df_final = pd.concat([df_train] + augmented_dfs, ignore_index=True)
    else:
        df_final = df_train.copy()

    return df_final

if __name__ == "__main__":
    pass