"""Загрузка датасетов, дедупликация, k-core фильтр, поюзерный сплит."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    interactions: pd.DataFrame   # колонки: user, item, rating, timestamp
    item_meta: pd.DataFrame      # индекс = item; колонки: title, text, genres (list[str])
    explicit: bool               # есть ли явные оценки
    rel_threshold: float         # rating >= threshold => релевантно
    chronological: bool          # есть ли время для хронологического сплита


@dataclass
class Split:
    train: pd.DataFrame          # user, item, rating (индексы 0..n-1)
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_ids: np.ndarray         # индекс -> исходный id
    item_ids: np.ndarray
    meta: pd.DataFrame           # строки выровнены по индексу item (0..n_items-1)
    explicit: bool
    rel_threshold: float

    @property
    def train_pos(self) -> pd.DataFrame:
        return self.train[self.train["rating"] >= self.rel_threshold]

    @property
    def test_rel(self) -> dict[int, set[int]]:
        pos = self.test[self.test["rating"] >= self.rel_threshold]
        return {int(u): set(map(int, g["item"])) for u, g in pos.groupby("user")}

    @property
    def eval_users(self) -> np.ndarray:
        """Пользователи, попавшие в оценку: есть тест и хотя бы 1 релевантный объект."""
        return np.array(sorted(self.test_rel.keys()), dtype=int)

    @property
    def train_counts(self) -> np.ndarray:
        return np.bincount(self.train["user"], minlength=self.n_users)


def _tok(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def load_movielens(path: str) -> Dataset:
    r = pd.read_csv(os.path.join(path, "ratings.csv")).rename(
        columns={"userId": "user", "movieId": "item"})
    movies = pd.read_csv(os.path.join(path, "movies.csv"))
    tags_path = os.path.join(path, "tags.csv")
    tag_text = pd.Series(dtype=str)
    if os.path.exists(tags_path):
        t = pd.read_csv(tags_path)
        tag_text = t.groupby("movieId")["tag"].apply(
            lambda s: " ".join(s.astype(str).str.lower()))
    genres = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|"))
    meta = pd.DataFrame({"title": movies["title"].values, "genres": genres.values},
                        index=pd.Index(movies["movieId"].values, name="item"))
    tags = tag_text.reindex(meta.index).fillna("")
    # у MovieLens нет описаний: жанры удваиваем как основной сигнал, добавляем теги и название
    meta["text"] = [" ".join(" ".join(_tok(g) for g in gl) for _ in range(2)) + " " + tg + " " + ti
                    for gl, tg, ti in zip(meta["genres"], tags, meta["title"].str.lower())]
    return Dataset("movielens", r[["user", "item", "rating", "timestamp"]], meta,
                   explicit=True, rel_threshold=4.0, chronological=True)


def load_lastfm(path: str) -> Dataset:
    kw = dict(sep="\t", encoding="latin-1")
    ua = pd.read_csv(os.path.join(path, "user_artists.dat"), **kw).rename(
        columns={"userID": "user", "artistID": "item", "weight": "rating"})
    ua["timestamp"] = np.nan
    artists = pd.read_csv(os.path.join(path, "artists.dat"), usecols=["id", "name"], **kw)
    tags = pd.read_csv(os.path.join(path, "tags.dat"), **kw)
    uta = pd.read_csv(os.path.join(path, "user_taggedartists.dat"), **kw)
    cnt = (uta.groupby(["artistID", "tagID"]).size().reset_index(name="n")
              .merge(tags, on="tagID"))
    cnt["tok"] = cnt["tagValue"].astype(str).map(_tok)
    cnt["rep"] = cnt.apply(lambda r: " ".join([r["tok"]] * min(int(r["n"]), 10)), axis=1)
    text = cnt.groupby("artistID")["rep"].apply(" ".join)
    top = cnt.sort_values("n", ascending=False).groupby("artistID")["tok"].apply(
        lambda s: list(s.head(3)))
    meta = pd.DataFrame({"title": artists["name"].values},
                        index=pd.Index(artists["id"].values, name="item"))
    meta["text"] = text.reindex(meta.index).fillna("")
    meta["genres"] = top.reindex(meta.index).apply(lambda x: x if isinstance(x, list) else [])
    return Dataset("lastfm", ua[["user", "item", "rating", "timestamp"]], meta,
                   explicit=False, rel_threshold=-np.inf, chronological=False)


def load_amazon(path: str) -> Dataset:
    """Универсальный загрузчик: ratings.csv (user,item,rating,timestamp)
    и необязательный items.csv (item,title,description,category; категории через |)."""
    r = pd.read_csv(os.path.join(path, "ratings.csv"))
    if "timestamp" not in r:
        r["timestamp"] = np.nan
    ip = os.path.join(path, "items.csv")
    if os.path.exists(ip):
        it = pd.read_csv(ip).fillna("")
        for c in ("title", "description", "category"):
            if c not in it:
                it[c] = ""
        meta = pd.DataFrame({"title": it["title"].astype(str).values,
                             "genres": it["category"].apply(
                                 lambda c: [x for x in str(c).split("|") if x]).values},
                            index=pd.Index(it["item"].values, name="item"))
        meta["text"] = (it["title"].astype(str) + " " + it["description"].astype(str)
                        + " " + it["category"].astype(str)).values
    else:
        ids = sorted(r["item"].unique())
        meta = pd.DataFrame({"title": [""] * len(ids), "genres": [[] for _ in ids],
                             "text": [""] * len(ids)}, index=pd.Index(ids, name="item"))
    return Dataset("amazon", r[["user", "item", "rating", "timestamp"]], meta,
                   explicit=True, rel_threshold=4.0, chronological=bool(r["timestamp"].notna().any()))


LOADERS = {"movielens": load_movielens, "lastfm": load_lastfm, "amazon": load_amazon}


def load_dataset(name: str, path: str) -> Dataset:
    return LOADERS[name](path)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Схлопывает дубликаты (user, item): последняя по времени / максимальная по весу."""
    key = "timestamp" if df["timestamp"].notna().any() else "rating"
    return (df.sort_values(key, kind="stable")
              .drop_duplicates(["user", "item"], keep="last").reset_index(drop=True))


def kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Итеративно удаляет пользователей и объекты с < k взаимодействий."""
    while True:
        n = len(df)
        df = df[df.groupby("user")["item"].transform("size") >= k]
        df = df[df.groupby("item")["user"].transform("size") >= k]
        if len(df) == n:
            return df.reset_index(drop=True)


def make_split(ds: Dataset, k_core: int = 5, n_test: int = 5, min_hist: int = 10,
               seed: int = 42) -> Split:
    df = kcore(dedup(ds.interactions), k_core)
    user_ids = np.sort(df["user"].unique())
    item_ids = np.sort(df["item"].unique())
    u_map = {u: i for i, u in enumerate(user_ids)}
    i_map = {v: i for i, v in enumerate(item_ids)}
    df = df.assign(user=df["user"].map(u_map).astype(int), item=df["item"].map(i_map).astype(int))
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for _, g in df.groupby("user", sort=True):
        if len(g) < min_hist:
            tr.append(g)
            continue
        if ds.chronological:
            g = g.sort_values("timestamp", kind="stable")
            tr.append(g.iloc[:-n_test]); te.append(g.iloc[-n_test:])
        else:
            p = rng.permutation(len(g))
            tr.append(g.iloc[p[n_test:]]); te.append(g.iloc[p[:n_test]])
    train = pd.concat(tr, ignore_index=True)[["user", "item", "rating"]]
    test = pd.concat(te, ignore_index=True)[["user", "item", "rating"]]
    meta = ds.item_meta.reindex(item_ids).reset_index(drop=True)
    meta["title"] = meta["title"].fillna("").astype(str)
    meta["text"] = meta["text"].fillna("").astype(str)
    meta["genres"] = meta["genres"].apply(lambda x: x if isinstance(x, list) else [])
    return Split(train, test, len(user_ids), len(item_ids), user_ids, item_ids, meta,
                 ds.explicit, ds.rel_threshold)
