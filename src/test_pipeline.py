"""
Test di sanita' end-to-end su un mini-dataset sintetico (non il dataset MET reale).

Obiettivo: verificare che generate_user_profiles.py e recommender_baseline.py girino
correttamente insieme (nessun crash, metriche nei range attesi) PRIMA di lanciare la
pipeline completa sulle 2000 opere reali — utile anche come evidenza di test nel report
("codice validato con un dataset di controllo, non solo eseguito una volta a caso").

Uso:
    python test_pipeline.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from generate_user_profiles import (
    build_vocabularies,
    generate_profile,
    score_item,
)
from recommender_baseline import (
    build_idf,
    build_item_document,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recommend_for_profile,
)


def make_synthetic_records(n=40):
    """Genera un mini-dataset con la stessa struttura di met_objects.jsonl, ma con
    distribuzioni note e controllate (utile per verificare che il codice si comporti
    in modo plausibile, non solo che non vada in errore)."""
    nationalities = ["French", "Italian", "Dutch", "Spanish"]
    classifications = ["Paintings", "Miniatures", "Drawings"]
    tag_pool = ["portrait", "landscape", "religious", "still life", "royalty", "mythology"]

    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        n_tags = rng.integers(1, 4)
        record = {
            "objectID": i,
            "title": f"Synthetic Work {i}",
            "classification": rng.choice(classifications),
            "artistNationality": rng.choice(nationalities),
            "tags": [{"term": t} for t in rng.choice(tag_pool, size=n_tags, replace=False)],
            # ~25% senza anno noto, per testare il ramo period_score == 0.5 / boost 0.5
            "objectBeginDate": None if rng.random() < 0.25 else int(rng.integers(1300, 1900)),
        }
        records.append(record)
    return records


def run():
    print("=== Test 1: caricamento vocabolari da dataset sintetico ===")
    records = make_synthetic_records(40)
    nat_counter, cls_counter, tag_counter, years = build_vocabularies(records)
    assert len(nat_counter) > 0, "nessuna nazionalita' trovata"
    assert len(tag_counter) > 0, "nessun tag trovato"
    assert len(years) > 0, "nessun anno noto (atteso: ~75% dei 40 record)"
    print(f"OK — nazionalita': {len(nat_counter)}, classificazioni: {len(cls_counter)}, "
          f"tag: {len(tag_counter)}, opere con anno noto: {len(years)}/{len(records)}")

    print("\n=== Test 2: generazione profili + score_item in [0,1] ===")
    rng = np.random.default_rng(42)
    profiles = [generate_profile(uid, nat_counter, cls_counter, tag_counter, years, rng) for uid in range(5)]
    for p in profiles:
        for r in records:
            s = score_item(p, r, rng)
            assert 0.0 <= s <= 1.0, f"score fuori range: {s}"
    print(f"OK — {len(profiles)} profili generati, tutti gli score_item in [0,1]")

    print("\n=== Test 3: relevance set (ground truth) coerente ===")
    top_pct = 0.2
    n_relevant = max(1, int(len(records) * top_pct))
    relevance_sets = {}
    for p in profiles:
        scored = [(r["objectID"], score_item(p, r, rng)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        relevance_sets[str(p["user_id"])] = [oid for oid, _ in scored[:n_relevant]]
        assert len(relevance_sets[str(p["user_id"])]) == n_relevant
    print(f"OK — relevance set da {n_relevant} opere/utente su {len(records)} totali")

    print("\n=== Test 4: recommender baseline (TF-IDF + cosine) indipendente dal ground truth ===")
    records_by_id = {r["objectID"]: r for r in records}
    item_ids = list(records_by_id.keys())
    item_documents = {oid: build_item_document(records_by_id[oid]) for oid in item_ids}
    idf = build_idf(list(item_documents.values()))

    from recommender_baseline import vectorize
    item_vectors = {oid: vectorize(doc, idf) for oid, doc in item_documents.items()}

    top_k = 5
    all_precisions, all_recalls, all_ndcgs = [], [], []
    for p in profiles:
        uid = str(p["user_id"])
        top = recommend_for_profile(p, item_ids, item_vectors, records_by_id, idf, top_k)
        assert len(top) == top_k
        recommended_ids = [oid for oid, _ in top]
        relevant_ids = relevance_sets[uid]

        prec = precision_at_k(recommended_ids, relevant_ids, top_k)
        rec = recall_at_k(recommended_ids, relevant_ids, top_k)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, top_k)
        for m in (prec, rec, ndcg):
            assert 0.0 <= m <= 1.0
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_ndcgs.append(ndcg)

    print(f"OK — precision@{top_k} medio: {np.mean(all_precisions):.3f}, "
          f"recall@{top_k} medio: {np.mean(all_recalls):.3f}, "
          f"ndcg@{top_k} medio: {np.mean(all_ndcgs):.3f}")
    print("  (su dati sintetici casuali questi valori NON sono indicativi delle prestazioni "
          "reali — servono solo a verificare che il codice sia corretto e che le metriche "
          "restino nel range [0,1] senza errori.)")

    print("\n=== Test 5: round-trip su file temporanei (I/O) ===")
    with tempfile.TemporaryDirectory() as tmp:
        objects_path = Path(tmp) / "objects.jsonl"
        with open(objects_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loaded = [json.loads(l) for l in open(objects_path, encoding="utf-8")]
        assert len(loaded) == len(records)
    print("OK — scrittura/lettura JSONL corretta")

    print("\n=== Tutti i test passati ===")


if __name__ == "__main__":
    run()
