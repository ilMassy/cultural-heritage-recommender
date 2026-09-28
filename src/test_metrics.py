"""
test_metrics.py — Test unitari per metrics.py (punto 11 della roadmap).

Uso: pytest src/test_metrics.py -v
"""

import math
import pytest

from metrics import precision_at_k, recall_at_k, ndcg_at_k


# --------------------------------------------------------------- precision_at_k

def test_precision_perfect():
    assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_precision_none_relevant():
    assert precision_at_k([1, 2, 3], {9, 10}, 3) == 0.0


def test_precision_partial():
    # 2 rilevanti su 4 raccomandati (a k=4)
    assert precision_at_k([1, 2, 3, 4], {1, 3, 99}, 4) == pytest.approx(0.5)


def test_precision_empty_recommended():
    assert precision_at_k([], {1, 2}, 5) == 0.0


def test_precision_k_larger_than_list():
    # solo 2 item raccomandati anche se k=10: precision calcolata sui 2 disponibili
    assert precision_at_k([1, 2], {1, 2}, 10) == 1.0


def test_precision_ignores_items_beyond_k():
    # un item rilevante c'è ma oltre k: non deve contare
    assert precision_at_k([1, 2, 3], {3}, 2) == 0.0


# ------------------------------------------------------------------ recall_at_k

def test_recall_empty_relevant_set():
    # nessun item rilevante per l'utente: definito come 0.0, non un'eccezione
    assert recall_at_k([1, 2, 3], [], 3) == 0.0


def test_recall_all_found():
    assert recall_at_k([1, 2, 3], [1, 2], 3) == 1.0


def test_recall_partial():
    assert recall_at_k([1, 2, 3], [1, 2, 3, 4], 3) == pytest.approx(0.75)


def test_recall_none_found():
    assert recall_at_k([1, 2, 3], [4, 5], 3) == 0.0


def test_recall_unaffected_by_k_beyond_recommended_length():
    assert recall_at_k([1, 2], [1, 2, 3], 10) == pytest.approx(2 / 3)


# -------------------------------------------------------------------- ndcg_at_k

def test_ndcg_perfect_ranking():
    # tutti i rilevanti nelle prime posizioni: NDCG = 1.0 per costruzione
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)


def test_ndcg_no_hits():
    assert ndcg_at_k([1, 2, 3], {9, 10}, 3) == 0.0


def test_ndcg_empty_relevant_set():
    # idcg = 0 -> ritorna 0.0, non ZeroDivisionError
    assert ndcg_at_k([1, 2, 3], [], 3) == 0.0


def test_ndcg_order_matters():
    # stesso singolo hit, ma in posizione 1 vs posizione 3: il primo deve avere NDCG più alto
    ndcg_top = ndcg_at_k([1, 2, 3], {1}, 3)
    ndcg_bottom = ndcg_at_k([2, 3, 1], {1}, 3)
    assert ndcg_top > ndcg_bottom
    assert ndcg_top == pytest.approx(1.0)  # unico rilevante, in cima: ideale


def test_ndcg_caps_idcg_at_k():
    # 5 item rilevanti ma k=2: l'ideale possibile è avere 2 hit su 2, non 5
    recommended = [1, 2, 99, 98, 97]
    relevant = {1, 2, 3, 4, 5}
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(1.0)


def test_ndcg_matches_manual_calculation():
    # calcolo a mano: rilevante in posizione 2 (indice 1) su k=3
    # dcg = 1/log2(2+1) = 1/log2(3); idcg (1 solo rilevante) = 1/log2(1+1) = 1/log2(2) = 1
    recommended = [99, 1, 98]
    relevant = {1}
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(recommended, relevant, 3) == pytest.approx(expected)


# --------------------------------------------------- coerenza tra le tre metriche

@pytest.mark.parametrize("recommended,relevant,k", [
    ([1, 2, 3, 4, 5], {1, 3}, 3),
    ([], {1, 2}, 5),
    ([1, 2, 3], set(), 3),
    (list(range(100)), {5, 15, 25, 200}, 10),
])
def test_metrics_always_in_unit_interval(recommended, relevant, k):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    n = ndcg_at_k(recommended, relevant, k)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= n <= 1.0
