from __future__ import annotations
 
from typing import List, Optional
 
import numpy as np
import pandas as pd
 
def classify_error(
    correct: bool,
    predicted: Optional[str],
    categories: List[str],
) -> Optional[str]:
    """_summary_

    Args:
        correct (bool): _description_
        predicted (Optional[str]): _description_
        categories (List[str]): _description_

    Returns:
        Optional[str]: _description_
    """
    if correct:
        return None
    if predicted is None:
        return "abstention"
    if predicted not in categories:
        return "hallucination"
    return "misclassification"
 

def tag_collapse(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        threshold (float, optional): _description_. Defaults to 0.90.

    Returns:
        pd.DataFrame: _description_
    """
    valid = df["predicted"].dropna()
    if valid.empty:
        return df
 
    dominant_label = valid.mode().iloc[0]
    dominant_frac  = (valid == dominant_label).sum() / len(valid)
 
    if dominant_frac >= threshold:
        mask = df["predicted"] == dominant_label
        df.loc[mask, "error_type"] = "collapse"
 
    return df
 
 

def bootstrap_accuracy(
    correct: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """_summary_

    Args:
        correct (np.ndarray): _description_
        n_bootstrap (int, optional): _description_. Defaults to 1000.
        alpha (float, optional): _description_. Defaults to 0.05.
        rng_seed (int, optional): _description_. Defaults to 42.

    Returns:
        tuple[float, float, float]: _description_
    """
    rng = np.random.default_rng(rng_seed)
    n   = len(correct)
    boot_means = np.array([
        rng.choice(correct, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(correct.mean()), lo, hi
 
 
def _accuracy_row(subset: pd.DataFrame, group_label: str) -> dict:
    """Compute one row of an accuracy summary table for a given subset."""
    correct = subset["correct"].values.astype(float)
    mean, lo, hi = bootstrap_accuracy(correct)
    return {
        "group":    group_label,
        "n":        len(subset),
        "accuracy": round(mean, 4),
        "ci_lower": round(lo,   4),
        "ci_upper": round(hi,   4),
    }
 
 
def compute_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        dict[str, pd.DataFrame]: _description_
    """
    summaries: dict[str, pd.DataFrame] = {}
    summaries["overall"] = pd.DataFrame([_accuracy_row(df, "all")])
    summaries["by_region"] = pd.DataFrame([
        _accuracy_row(group, region)
        for region, group in df.groupby("region")
    ])
    summaries["by_quintile"] = pd.DataFrame([
        _accuracy_row(group, quintile)
        for quintile, group in df.groupby("income_quintile")
    ])
    summaries["by_category"] = pd.DataFrame([
        _accuracy_row(group, category)
        for category, group in df.groupby("ground_truth")
    ])
    summaries["error_counts"] = (
        df["error_type"]
        .value_counts(dropna=False)
        .rename_axis("error_type")
        .reset_index(name="count")
    )
 
    return summaries