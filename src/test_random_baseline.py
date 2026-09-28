"""
test_random_baseline.py — Test unitari per random_baseline.py (punto 11 della roadmap).

Uso: pytest src/test_random_baseline.py -v
"""

import json
import math

import pytest

from random_baseline import (
    load_relevance_sets, evaluate, run_random, run_popularity_tags, _tag_names,
)


# ---------------------------------------------------------------- load_relevance_sets

def test_load_relevance_sets_dict_of_lists(tmp_path):
    # formato {"0": [1,2,3], "1": [4,5]}
    p = tmp_path / "rel.json"
    p.write_text(json.dumps({"0": [1, 2, 3], "1": [4, 5]}), encoding="utf-8")
    result = load_relevance_sets(str(p))
    assert result == {"0": {1, 2, 3}, "1": {4, 5}}
    assert all(isinstance(v, set) for v in result.values())


def test_load_relevance_sets_dict_with_relevant_ids_key(tmp_path):
    # formato {"0": {"relevant_ids": [1,2]}}
    p = tmp_path / "rel.json"
    p.write_text(json.dumps({"0": {"relevant_ids": [1, 2]}}), encoding="utf-8")
    result = load_relevance_sets(str(p))
    assert result == {"0": {1, 2}}


def test_load_relevance_sets_dict_with_alternate_keys(tmp_path):
    # le chiavi alternative "relevant"/"relevance_set"/"objectIDs" devono funzionare tutte
    for key in ("relevant", "relevance_set", "objectIDs"):
        p = tmp_path / f"rel_{key}.json"
        p.write_text(json.dumps({"0": {key: [7, 8]}}), encoding="utf-8")
        result = load_relevance_sets(str(p))
        assert result == {"0": {7, 8}}, f"fallito con chiave {key!r}"


def test_load_relevance_sets_list_format(tmp_path):
    # formato lista di dict [{"user_id": 0, "relevant_ids": [...]}]
    p = tmp_path / "rel.json"
    p.write_text(json.dumps([
        {"user_id": 0, "relevant_ids": [1, 2]},
        {"user_id": 1, "relevant_ids": [3]},
    ]), encoding="utf-8")
    result = load_relevance_sets(str(p))
    assert result == {"0": {1, 2}, "1": {3}}


def test_load_relevance_sets_entries_as_dicts_with_objectid(tmp_path):
    # entry tipo {"objectID": 5, "score": 0.9} invece di un intero nudo
    p = tmp_path / "rel.json"
    p.write_text(json.dumps({"0": [{"objectID": 5, "score": 0.9}, {"objectID": 6}]}), encoding="utf-8")
    result = load_relevance_sets(str(p))
    assert result == {"0": {5, 6}}


def test_load_relevance_sets_unrecognized_format_raises(tmp_path):
    p = tmp_path / "rel.json"
    p.write_text(json.dumps("non un dict né una lista"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_relevance_sets(str(p))


def test_load_relevance_sets_unrecognized_dict_keys_raise(tmp_path):
    p = tmp_path / "rel.json"
    p.write_text(json.dumps({"0": {"chiave_a_caso": [1, 2]}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_relevance_sets(str(p))


# ----------------------------------------------------------------------- _tag_names

def test_tag_names_handles_dicts_and_strings():
    tags = [{"term": "Men"}, "Women", {"term": "Portraits"}, {"no_term": "x"}]
    assert _tag_names(tags) == ["Men", "Women", "Portraits"]


def test_tag_names_handles_none():
    assert _tag_names(None) == []


def test_tag_names_handles_empty_list():
    assert _tag_names([]) == []


# --------------------------------------------------------------------------- evaluate

def test_evaluate_perfect_recommendations():
    recs = {"0": [1, 2, 3], "1": [4, 5]}
    rel = {"0": {1, 2, 3}, "1": {4, 5}}
    m = evaluate(recs, rel, k=3)
    assert m["precision_at_k"] == pytest.approx(1.0)
    assert m["recall_at_k"] == pytest.approx(1.0)
    assert m["ndcg_at_k"] == pytest.approx(1.0)
    assert m["std_users"]["precision_at_k"] == pytest.approx(0.0)


def test_evaluate_no_hits():
    recs = {"0": [1, 2, 3]}
    rel = {"0": {99, 100}}
    m = evaluate(recs, rel, k=3)
    assert m["precision_at_k"] == 0.0
    assert m["recall_at_k"] == 0.0
    assert m["ndcg_at_k"] == 0.0


def test_evaluate_std_users_reflects_variance_across_users():
    # utente 0 perfetto, utente 1 a zero: la std tra utenti deve essere > 0
    recs = {"0": [1, 2], "1": [1, 2]}
    rel = {"0": {1, 2}, "1": {99, 100}}
    m = evaluate(recs, rel, k=2)
    assert m["std_users"]["precision_at_k"] > 0.0
    assert m["precision_at_k"] == pytest.approx(0.5)  # media tra 1.0 e 0.0


# ------------------------------------------------------------------------ run_random

def test_run_random_reproducible_with_fixed_seed():
    import random
    all_ids = list(range(50))
    rel_sets = {"0": {1, 2, 3}, "1": {10, 11}}
    r1 = run_random(all_ids, rel_sets, k=5, n_runs=20, rng=random.Random(42))
    r2 = run_random(all_ids, rel_sets, k=5, n_runs=20, rng=random.Random(42))
    assert r1["precision_at_k"] == r2["precision_at_k"]
    assert r1["std_runs"]["precision_at_k"] == r2["std_runs"]["precision_at_k"]


def test_run_random_respects_pool_restriction():
    import random
    # pool ristretto a soli item rilevanti: la precisione media deve salire parecchio
    # rispetto a campionare dall'intero catalogo (sanity check di direzione, non di valore esatto)
    all_ids = list(range(1000))
    rel_sets = {"0": {1, 2, 3, 4, 5}}
    small_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    r_full = run_random(all_ids, rel_sets, k=5, n_runs=30, rng=random.Random(0), pool=all_ids)
    r_pool = run_random(all_ids, rel_sets, k=5, n_runs=30, rng=random.Random(0), pool=small_pool)
    assert r_pool["precision_at_k"] > r_full["precision_at_k"]


def test_run_random_output_keys():
    import random
    all_ids = list(range(20))
    rel_sets = {"0": {1, 2}}
    r = run_random(all_ids, rel_sets, k=3, n_runs=5, rng=random.Random(1))
    assert set(r) >= {"n_runs", "precision_at_k", "recall_at_k", "ndcg_at_k", "std_runs"}
    assert r["n_runs"] == 5


# ------------------------------------------------------------------ run_popularity_tags

def test_run_popularity_tags_deterministic():
    items = [
        {"objectID": 1, "tags": [{"term": "A"}, {"term": "B"}]},
        {"objectID": 2, "tags": [{"term": "A"}]},
        {"objectID": 3, "tags": [{"term": "B"}, {"term": "C"}]},
        {"objectID": 4, "tags": []},
    ]
    rel_sets = {"0": {1}}
    m1 = run_popularity_tags(items, rel_sets, k=2)
    m2 = run_popularity_tags(items, rel_sets, k=2)
    assert m1 == m2  # nessuna casualità: stesso input, stesso output


def test_run_popularity_tags_tie_break_by_objectid():
    # item 1 e 2 hanno esattamente lo stesso punteggio (stesso singolo tag "A"):
    # il tie-break deterministico su objectID deve mettere prima l'ID più basso
    items = [
        {"objectID": 2, "tags": [{"term": "A"}]},
        {"objectID": 1, "tags": [{"term": "A"}]},
    ]
    rel_sets = {"0": {1}}
    m = run_popularity_tags(items, rel_sets, k=1)
    # se il tie-break funziona, l'item scelto per il k=1 è l'ID 1 (rilevante) -> precision 1.0
    assert m["precision_at_k"] == pytest.approx(1.0)


def test_run_popularity_tags_items_without_tags_score_zero():
    items = [
        {"objectID": 1, "tags": [{"term": "A"}, {"term": "A"}]},  # tag molto frequente
        {"objectID": 2, "tags": []},  # nessun tag -> punteggio 0.0, ultimo in classifica
    ]
    rel_sets = {"0": {2}}
    m = run_popularity_tags(items, rel_sets, k=1)
    # item 2 (senza tag) non deve mai finire primo: precision@1 deve essere 0
    assert m["precision_at_k"] == 0.0
