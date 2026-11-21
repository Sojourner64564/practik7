"""Метрики качества ранжирования, покрытия, разнообразия и новизны."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def precision_at_k(rec: Sequence[int], rel: set, k: int) -> float:
    return len(set(list(rec)[:k]) & rel) / k


def recall_at_k(rec: Sequence[int], rel: set, k: int) -> float:
    return len(set(list(rec)[:k]) & rel) / len(rel) if rel else 0.0


def average_precision_at_k(rec: Sequence[int], rel: set, k: int) -> float:
    hits, s = 0, 0.0
    for i, item in enumerate(list(rec)[:k], 1):
        if item in rel:
            hits += 1
            s += hits / i
    return s / min(len(rel), k) if rel else 0.0


def ndcg_at_k(rec: Sequence[int], rel: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, it in enumerate(list(rec)[:k], 1) if it in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def catalog_coverage(all_recs: Iterable[Sequence[int]], n_items: int) -> float:
    seen = set()
    for r in all_recs:
        seen.update(map(int, r))
    return len(seen) / n_items


def intra_list_diversity(rec: Sequence[int], X) -> float:
    """1 - среднее попарное косинусное сходство (X - L2-нормированная матрица объектов)."""
    rec = list(rec)
    n = len(rec)
    if n < 2:
        return 0.0
    S = X[rec] @ X[rec].T
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    mean_sim = (S.sum() - np.trace(S)) / (n * (n - 1))
    return float(1.0 - mean_sim)


def novelty(rec: Sequence[int], pop_prob: np.ndarray) -> float:
    """Средняя self-information: -log2 p(item), p = доля пользователей, знающих объект."""
    p = np.clip(pop_prob[list(rec)], 1e-12, 1.0)
    return float(np.mean(-np.log2(p)))


def genre_count(rec: Sequence[int], genres: Sequence[Sequence[str]]) -> int:
    return len({g for i in rec for g in genres[i]})


def mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v)))
