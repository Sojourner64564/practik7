import math
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recsys_lib import metrics as M
from recsys_lib.evaluate import mmr_rerank


def test_precision_recall():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert M.precision_at_k(rec, rel, 4) == 0.5
    assert math.isclose(M.recall_at_k(rec, rel, 4), 2 / 3)
    assert M.recall_at_k(rec, set(), 4) == 0.0


def test_average_precision():
    # попадания на позициях 1 и 3: (1/1 + 2/3) / min(2, 3)
    assert math.isclose(M.average_precision_at_k([1, 2, 3], {1, 3}, 3), (1 + 2 / 3) / 2)
    assert M.average_precision_at_k([5, 6], {1}, 2) == 0.0


def test_ndcg():
    assert math.isclose(M.ndcg_at_k([1, 2], {1, 2}, 2), 1.0)
    dcg = 1 / math.log2(3)
    idcg = 1.0
    assert math.isclose(M.ndcg_at_k([9, 1], {1}, 2), dcg / idcg)


def test_coverage():
    assert M.catalog_coverage([[0, 1], [1, 2]], 10) == 0.3


def test_diversity():
    X = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
    assert math.isclose(M.intra_list_diversity([0, 1], X), 0.0)
    assert math.isclose(M.intra_list_diversity([0, 2], X), 1.0)
    assert M.intra_list_diversity([0], X) == 0.0


def test_novelty_and_genres():
    p = np.array([0.5, 0.25])
    assert math.isclose(M.novelty([0, 1], p), (1 + 2) / 2)
    assert M.genre_count([0, 1], [["a", "b"], ["b", "c"]]) == 3


def test_ci():
    m, ci = M.mean_ci95([1.0, 1.0, 1.0])
    assert m == 1.0 and ci == 0.0


def test_mmr_increases_diversity():
    X = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
    ids, scores = np.array([0, 1, 2]), np.array([1.0, 0.95, 0.5])
    assert list(mmr_rerank(ids, scores, X, 1.0, 2)) == [0, 1]   # чистая релевантность
    assert list(mmr_rerank(ids, scores, X, 0.3, 2)) == [0, 2]   # штраф за схожесть
