from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from ..config.config import (
    SEM_SIM_MODEL_NAME,
    CTX_SIM_MODEL_NAME,
    SEM_SIM_THRESHOLD,
    CTX_SIM_THRESHOLD,
)


class SimilarityEvaluator:
    """
    Computes semantic and contextual similarity between two short text labels.
    """

    def __init__(
        self,
        sem_model_name: str   = SEM_SIM_MODEL_NAME,
        ctx_model_name: str   = CTX_SIM_MODEL_NAME,
    ):
        self._sem_model_name = sem_model_name
        self._ctx_model_name = ctx_model_name
        self._sem_model: Optional[SentenceTransformer] = None
        self._ctx_model: Optional[CrossEncoder]        = None

    def load(self) -> None:
        """Download and initialise both similarity models. Idempotent."""
        if self._sem_model is not None:
            return
        self._sem_model = SentenceTransformer(self._sem_model_name)
        self._ctx_model = CrossEncoder(self._ctx_model_name)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity of bi-encoder embeddings, in [-1, 1]."""
        emb1 = self._sem_model.encode(text1, convert_to_tensor=True)
        emb2 = self._sem_model.encode(text2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())

    def contextual_similarity(self, text1: str, text2: str) -> float:
        """Cross-encoder STS score, sigmoid-normalised to (0, 1)."""
        raw = float(self._ctx_model.predict([(text1, text2)]))
        return float(1.0 / (1.0 + np.exp(-raw)))

def classify_error(
    ground_truth: str,
    predicted:    str,
    evaluator:    SimilarityEvaluator,
    sem_threshold: float = SEM_SIM_THRESHOLD,
    ctx_threshold: float = CTX_SIM_THRESHOLD,
) -> tuple[str, float, float]:
    """
    Classify one prediction and return its error type plus raw similarity scores.
    """
    if predicted == "abstention":
        return "abstention", 0.0, 0.0

    sem_sim = evaluator.semantic_similarity(ground_truth, predicted)
    ctx_sim = evaluator.contextual_similarity(ground_truth, predicted)

    is_sem_match = sem_sim >= sem_threshold
    is_ctx_match = ctx_sim >= ctx_threshold

    # Fix: Ensure error_type is always associated with a value
    if is_sem_match and is_ctx_match:
        error_type = "correct"
    elif not is_sem_match and not is_ctx_match:
        error_type = "hallucination"
    else:
        # Catch-all for when only one of the two similarities meets the threshold
        error_type = "misclassification"

    return error_type, sem_sim, ctx_sim


def bootstrap_accuracy(
    correct:     np.ndarray,
    n_bootstrap: int   = 1000,
    alpha:       float = 0.05,
    rng_seed:    int   = 42,
) -> tuple[float, float, float]:
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
    correct = (subset["error_type"] == "correct").values.astype(float)
    mean, lo, hi = bootstrap_accuracy(correct)
    return {
        "group":         group_label,
        "n":             len(subset),
        "accuracy":      round(mean, 4),
        "ci_lower":      round(lo,   4),
        "ci_upper":      round(hi,   4),
        "mean_sem_sim":  round(subset["sem_similarity"].mean(), 4),
        "mean_ctx_sim":  round(subset["ctx_similarity"].mean(), 4),
    }


def compute_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
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