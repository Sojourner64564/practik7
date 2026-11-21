"""Модели: Popular, Content (TF-IDF), SVD (surprise), ALS/BPR (implicit).
Единый интерфейс: fit(split) и score(users) -> матрица (len(users) x n_items)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Split


def build_tfidf(meta) -> sp.csr_matrix:
    """TF-IDF матрица объектов (L2-нормирована, значит X @ X.T = косинусное сходство)."""
    texts = meta["text"].replace("", "unknown_item")
    return TfidfVectorizer(sublinear_tf=True).fit_transform(texts).tocsr()


def _user_item_csr(split: Split, values=None) -> sp.csr_matrix:
    pos = split.train_pos
    data = np.ones(len(pos), dtype=np.float32) if values is None else values(pos)
    return sp.csr_matrix((data, (pos["user"].values, pos["item"].values)),
                         shape=(split.n_users, split.n_items))


class Popular:
    name = "popular"

    def fit(self, split: Split):
        self.pop = np.bincount(split.train_pos["item"], minlength=split.n_items).astype(float)
        return self

    def score(self, users):
        return np.tile(self.pop, (len(users), 1))


class Content:
    name = "content"

    def fit(self, split: Split):
        self.X = build_tfidf(split.meta)
        U = _user_item_csr(split)
        deg = np.asarray(U.sum(1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
        self.profile = (sp.diags(inv) @ U @ self.X).tocsr()   # средний вектор понравившихся объектов
        return self

    def score(self, users):
        return (self.profile[users] @ self.X.T).toarray()


class SVDModel:
    name = "svd"

    def __init__(self, factors=50, epochs=20, seed=42):
        self.factors, self.epochs, self.seed = factors, epochs, seed

    def fit(self, split: Split):
        try:
            from surprise import SVD, Dataset as SDS, Reader
        except ImportError as e:
            raise ImportError("Для модели svd нужен scikit-surprise (см. README)") from e
        df = split.train[["user", "item", "rating"]]
        reader = Reader(rating_scale=(float(df["rating"].min()), float(df["rating"].max())))
        ts = SDS.load_from_df(df, reader).build_full_trainset()
        algo = SVD(n_factors=self.factors, n_epochs=self.epochs, random_state=self.seed)
        algo.fit(ts)
        f = self.factors
        self.mu = ts.global_mean
        self.pu = np.zeros((split.n_users, f)); self.bu = np.zeros(split.n_users)
        self.qi = np.zeros((split.n_items, f)); self.bi = np.full(split.n_items, -1e9)
        for inner in ts.all_users():
            raw = int(ts.to_raw_uid(inner))
            self.pu[raw], self.bu[raw] = algo.pu[inner], algo.bu[inner]
        for inner in ts.all_items():
            raw = int(ts.to_raw_iid(inner))
            self.qi[raw], self.bi[raw] = algo.qi[inner], algo.bi[inner]
        return self

    def score(self, users):
        return self.mu + self.bu[users, None] + self.bi[None, :] + self.pu[users] @ self.qi.T


class ImplicitModel:
    def __init__(self, kind="als", factors=50, iterations=None, seed=42):
        assert kind in ("als", "bpr")
        self.kind, self.factors, self.seed = kind, factors, seed
        self.iterations = iterations or (15 if kind == "als" else 100)
        self.name = kind

    def fit(self, split: Split):
        try:
            from implicit.als import AlternatingLeastSquares
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as e:
            raise ImportError("Для моделей als/bpr нужна библиотека implicit (см. README)") from e
        if self.kind == "als" and not split.explicit:
            # уверенность растет с числом прослушиваний
            ui = _user_item_csr(split, lambda p: (1 + np.log1p(p["rating"].values)).astype(np.float32))
        else:
            ui = _user_item_csr(split)
        if self.kind == "als":
            m = AlternatingLeastSquares(factors=self.factors, iterations=self.iterations,
                                        random_state=self.seed, use_gpu=False)
        else:
            m = BayesianPersonalizedRanking(factors=self.factors, iterations=self.iterations,
                                            random_state=self.seed, use_gpu=False)
        m.fit(ui, show_progress=False)
        self.U, self.V = np.asarray(m.user_factors), np.asarray(m.item_factors)
        return self

    def score(self, users):
        return self.U[users] @ self.V.T


def build_model(name: str, seed: int = 42, factors: int = 50):
    if name == "popular":
        return Popular()
    if name == "content":
        return Content()
    if name == "svd":
        return SVDModel(factors=factors, seed=seed)
    if name in ("als", "bpr"):
        return ImplicitModel(name, factors=factors, seed=seed)
    raise ValueError(f"Неизвестная модель: {name}")
