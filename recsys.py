#!/usr/bin/env python3
"""Главный CLI-скрипт: загрузка данных, обучение моделей, оценка, EDA, MMR."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from recsys_lib.data import load_dataset, make_split
from recsys_lib.eda import run_eda
from recsys_lib.evaluate import evaluate, make_segments, recommend, significance_table
from recsys_lib.models import build_model, build_tfidf

ALL_MODELS = ["popular", "content", "svd", "als", "bpr"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Рекомендательная система для медиакаталога")
    p.add_argument("--dataset", required=True, choices=["movielens", "lastfm", "amazon"])
    p.add_argument("--path", required=True, help="папка с датасетом")
    p.add_argument("--model", default="all", help="popular,content,svd,als,bpr | all")
    p.add_argument("--k", type=int, nargs="+", default=[10])
    p.add_argument("--evaluate", action="store_true", help="рассчитать метрики")
    p.add_argument("--eda", action="store_true", help="построить графики EDA")
    p.add_argument("--mmr", type=float, nargs="*", default=[], help="значения lambda для MMR")
    p.add_argument("--segment-threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results")
    return p.parse_args(argv)


def write_cases(path, split, recs, users, n=4, seed=42):
    """Материал для ручного разбора: история, тест и топ-5 каждой модели для n пользователей."""
    rng = np.random.default_rng(seed)
    rel = split.test_rel
    chosen = rng.choice(users, size=min(n, len(users)), replace=False)
    title = split.meta["title"].tolist()
    lines = ["# Материал для ручного разбора\n"]
    for u in chosen:
        hist = split.train_pos[split.train_pos["user"] == u].sort_values("rating", ascending=False)
        lines.append(f"## Пользователь {split.user_ids[u]}\n")
        lines.append("**Любимое в train:** " + "; ".join(title[i] for i in hist["item"].head(5)) + "\n")
        lines.append("**Тест (релевантное):** " + "; ".join(title[i] for i in sorted(rel[int(u)])) + "\n")
        for name, r in recs.items():
            items = [("✅ " if int(i) in rel[int(u)] else "") + title[i] for i in r[int(u)][:5]]
            lines.append(f"- **{name}:** " + "; ".join(items))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_coverage(metrics, kmax, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = metrics[(metrics.metric == "coverage") & (metrics.segment == "all") & (metrics.k == kmax)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["model"], d["value"] * 100, color="#4c72b0")
    ax.set(title=f"Покрытие каталога @{kmax}", ylabel="% каталога")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.output, exist_ok=True)
    np.random.seed(a.seed)
    ds = load_dataset(a.dataset, a.path)
    split = make_split(ds, seed=a.seed)
    print(f"{a.dataset}: users={split.n_users} items={split.n_items} "
          f"train={len(split.train)} test={len(split.test)}")
    if a.eda:
        run_eda(split, a.output)

    names = ALL_MODELS if a.model == "all" else [m.strip() for m in a.model.split(",")]
    if "svd" in names and not split.explicit:
        print("SVD требует явных оценок - пропущен для этого датасета")
        names = [n for n in names if n != "svd"]

    users = split.eval_users
    kmax = max(a.k + [10])
    X = build_tfidf(split.meta)
    recs, fit_time = {}, {}
    for n in names:
        t = time.time()
        model = build_model(n, seed=a.seed).fit(split)
        fit_time[n] = round(time.time() - t, 2)
        recs[n] = recommend(model, split, users, kmax)
        if n == "content":
            for lam in a.mmr:
                recs[f"content+mmr{lam}"] = recommend(model, split, users, kmax, X=X, mmr_lambda=lam)
        print(f"  {n}: fit {fit_time[n]}s")

    dump_users = users[:50]
    with open(os.path.join(a.output, "recs.json"), "w", encoding="utf-8") as f:
        json.dump({m: {str(split.user_ids[u]): [int(split.item_ids[i]) for i in r[int(u)]]
                       for u in dump_users} for m, r in recs.items()}, f)

    if a.evaluate:
        thr, segs = make_segments(split, users, a.segment_threshold)
        metrics = pd.concat([evaluate(r, m, split, a.k, X, segs) for m, r in recs.items()],
                            ignore_index=True)
        metrics.to_csv(os.path.join(a.output, "metrics.csv"), index=False)
        plot_coverage(metrics, max(a.k), os.path.join(a.output, "coverage.png"))
        if "popular" in names:
            sig = significance_table(metrics)
            sig.to_csv(os.path.join(a.output, "significance.csv"), index=False)
            print(sig.to_string(index=False))
        write_cases(os.path.join(a.output, "cases.md"), split, recs, users, seed=a.seed)
        show = metrics[(metrics.segment == "all") & (metrics.k == max(a.k)) &
                       metrics.metric.isin(["map", "ndcg", "coverage", "diversity"])]
        print(show.pivot(index="model", columns="metric", values="value").round(4))
        print(f"segment threshold (train interactions): {thr}")
    else:
        thr = None

    import sklearn, scipy
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for lib in ("surprise", "implicit"):
        try:
            versions[lib] = __import__(lib).__version__
        except Exception:
            pass
    with open(os.path.join(a.output, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(a), seed=a.seed, versions=versions, n_users=split.n_users,
                       n_items=split.n_items, n_train=len(split.train), n_test=len(split.test),
                       n_eval_users=len(users), segment_threshold=thr, fit_seconds=fit_time),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
