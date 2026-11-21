"""Сквозной прогон на синтетическом MovieLens-подобном датасете (без surprise/implicit)."""
import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recsys

rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
G = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance"]
movies = pd.DataFrame({"movieId": range(1, 301), "title": [f"Movie {i}" for i in range(1, 301)],
                       "genres": ["|".join(rng.choice(G, 2, replace=False)) for _ in range(300)]})
rows = []
for u in range(1, 101):
    fav = rng.choice(G, 2, replace=False)
    for m in rng.choice(300, 40, replace=False):
        liked = any(g in movies.genres[m] for g in fav)
        rows.append((u, m + 1, rng.choice([4, 4.5, 5]) if liked else rng.choice([1, 2, 3]), 1000 + int(rng.integers(1e6))))
pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(f"{d}/ratings.csv", index=False)
movies.to_csv(f"{d}/movies.csv", index=False)
out = os.path.join(d, "out")
recsys.main(["--dataset", "movielens", "--path", d, "--model", "popular,content", "--k", "5", "10",
             "--eda", "--evaluate", "--mmr", "0.5", "0.9", "--output", out])
print(sorted(os.listdir(out)))
