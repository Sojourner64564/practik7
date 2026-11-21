"""Единая оценка: ранжирование по всему каталогу за вычетом train, MMR, сегменты, CI."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .data import Split


def recommend(model, split: Split, users: np.ndarray, k: int, X=None,
              mmr_lambda: float | None = None, pool: int = 50, batch: int = 256) -> dict[int, np.ndarray]:
    """Top-k по всему каталогу без train-объектов. При mmr_lambda - MMR-переранжирование пула."""
    depth = min(max(pool, k), split.n_items - 1) if mmr_lambda is not None else min(k, split.n_items - 1)
    by_user = split.train.groupby("user")["item"].apply(lambda s: s.values).to_dict()
    out = {}
    for b in range(0, len(users), batch):
        u = users[b:b + batch]
        s = np.array(model.score(u), dtype=np.float64)
        for row, uu in enumerate(u):
            s[row, by_user.get(int(uu), [])] = -np.inf
        top = np.argpartition(-s, depth - 1, axis=1)[:, :depth]
        vals = np.take_along_axis(s, top, 1)
        o = np.argsort(-vals, axis=1, kind="stable")
        top, vals = np.take_along_axis(top, o, 1), np.take_along_axis(vals, o, 1)
        for row, uu in enumerate(u):
            if mmr_lambda is None:
                out[int(uu)] = top[row]
            else:
                out[int(uu)] = mmr_rerank(top[row], vals[row], X, mmr_lambda, k)
    return out


def mmr_rerank(ids: np.ndarray, scores: np.ndarray, X, lam: float, k: int) -> np.ndarray:
    """Maximal Marginal Relevance: lam * rel - (1 - lam) * max_sim_to_selected."""
    s = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
    S = X[ids] @ X[ids].T
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    selected, cand = [], list(range(len(ids)))
    while len(selected) < k and cand:
        if not selected:
            val = s[cand]
        else:
            val = lam * s[cand] - (1 - lam) * S[np.ix_(cand, selected)].max(axis=1)
        best = cand[int(np.argmax(val))]
        selected.append(best)
        cand.remove(best)
    return ids[selected]


def make_segments(split: Split, users: np.ndarray, threshold: float | None = None):
    counts = split.train_counts[users]
    thr = float(np.median(counts)) if threshold is None else threshold
    return thr, {"all": users, "light": users[counts <= thr], "heavy": users[counts > thr]}


def evaluate(recs: dict[int, np.ndarray], model_name: str, split: Split, ks, X,
             segments: dict[str, np.ndarray]) -> pd.DataFrame:
    rel = split.test_rel
    pop_prob = np.bincount(split.train["item"], minlength=split.n_items) / split.n_users
    genres = split.meta["genres"].tolist()
    rows = []
    for seg, users in segments.items():
        if len(users) == 0:
            continue
        for k in ks:
            per = {m: [] for m in ["precision", "recall", "map", "ndcg", "diversity", "novelty", "n_genres"]}
            for u in users:
                r = recs[int(u)][:k]
                per["precision"].append(M.precision_at_k(r, rel[int(u)], k))
                per["recall"].append(M.recall_at_k(r, rel[int(u)], k))
                per["map"].append(M.average_precision_at_k(r, rel[int(u)], k))
                per["ndcg"].append(M.ndcg_at_k(r, rel[int(u)], k))
                per["diversity"].append(M.intra_list_diversity(r, X))
                per["novelty"].append(M.novelty(r, pop_prob))
                per["n_genres"].append(M.genre_count(r, genres))
            for m, vals in per.items():
                mean, ci = M.mean_ci95(vals)
                rows.append(dict(model=model_name, k=k, segment=seg, metric=m, value=mean,
                                 ci95=ci, n_users=len(users)))
            cov = M.catalog_coverage((recs[int(u)][:k] for u in users), split.n_items)
            rows.append(dict(model=model_name, k=k, segment=seg, metric="coverage", value=cov,
                             ci95=np.nan, n_users=len(users)))
    return pd.DataFrame(rows)


def significance_table(metrics: pd.DataFrame, baseline: str = "popular", metric: str = "map",
                       segment: str = "all") -> pd.DataFrame:
    """Консервативный критерий: интервалы value±ci95 пересекаются -> различие незначимо."""
    d = metrics[(metrics.metric == metric) & (metrics.segment == segment)]
    rows = []
    for k, g in d.groupby("k"):
        b = g[g.model == baseline]
        if b.empty:
            continue
        b = b.iloc[0]
        for _, r in g[g.model != baseline].iterrows():
            lo, hi = r.value - r.ci95, r.value + r.ci95
            overlap = not (lo > b.value + b.ci95 or hi < b.value - b.ci95)
            verdict = "незначимо" if overlap else ("лучше" if r.value > b.value else "хуже")
            rows.append(dict(model=r.model, k=k, metric=metric, value=r.value,
                             baseline=b.value, verdict=verdict))
    return pd.DataFrame(rows)
