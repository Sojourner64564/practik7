"""Разведочный анализ: статистики (eda.json) и графики (eda.png)."""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import Split


def run_eda(split: Split, out_dir: str) -> dict:
    both = np.concatenate([split.train["user"].values, split.test["user"].values])
    items = np.concatenate([split.train["item"].values, split.test["item"].values])
    ratings = np.concatenate([split.train["rating"].values, split.test["rating"].values])
    n_inter = len(both)
    u_cnt = np.bincount(both, minlength=split.n_users)
    i_cnt = np.bincount(items, minlength=split.n_items)
    genres = Counter(g for gl in split.meta["genres"] for g in gl)
    top_n = max(1, split.n_items // 10)
    stats = dict(
        n_users=int(split.n_users), n_items=int(split.n_items), n_interactions=int(n_inter),
        sparsity=float(1 - n_inter / (split.n_users * split.n_items)),
        user_activity_median=float(np.median(u_cnt)), user_activity_max=int(u_cnt.max()),
        item_popularity_median=float(np.median(i_cnt)), item_popularity_max=int(i_cnt.max()),
        top10pct_items_share=float(np.sort(i_cnt)[::-1][:top_n].sum() / n_inter),
        top_genres=genres.most_common(10),
    )
    if split.explicit:
        vals, cnts = np.unique(ratings, return_counts=True)
        stats["rating_distribution"] = {str(v): int(c) for v, c in zip(vals, cnts)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eda.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(u_cnt, bins=50, color="#4c72b0"); ax[0, 0].set_yscale("log")
    ax[0, 0].set(title="Активность пользователей", xlabel="взаимодействий", ylabel="пользователей")
    ax[0, 1].plot(np.sort(i_cnt)[::-1], color="#dd8452"); ax[0, 1].set_yscale("log")
    ax[0, 1].set(title="Популярность объектов (длинный хвост)", xlabel="ранг", ylabel="взаимодействий")
    if split.explicit:
        ax[1, 0].bar([str(v) for v in vals], cnts, color="#55a868")
        ax[1, 0].set(title="Распределение оценок", xlabel="оценка")
    else:
        ax[1, 0].hist(np.log10(np.clip(ratings, 1, None)), bins=40, color="#55a868")
        ax[1, 0].set(title="Число прослушиваний", xlabel="log10(weight)")
    tg = genres.most_common(10)
    if tg:
        ax[1, 1].barh([g for g, _ in tg][::-1], [c for _, c in tg][::-1], color="#c44e52")
    ax[1, 1].set(title="Топ жанров/тегов")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "eda.png"), dpi=130); plt.close(fig)
    return stats
