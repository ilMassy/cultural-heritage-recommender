"""
Recommender content-based classico (baseline) per il dataset MET.

PERCHE' TF-IDF + COSINE SIMILARITY E NON LA FORMULA PESATA DEI PROFILI
--------------------------------------------------------------------------------------
Il ground truth di rilevanza (generate_user_profiles.py) e' costruito con una formula
pesata esplicita (nationality/period/classification/tags, top 5% = "rilevante"). Se il
recommender da valutare usasse la STESSA formula per raccomandare, la valutazione
sarebbe circolare: precision@k risulterebbe artificialmente vicina a 1 perche' il
recommender starebbe solo ripetendo la funzione che ha etichettato le risposte giuste,
non dimostrando nessuna capacita' di generalizzazione.

Per questo il baseline qui implementato e' un algoritmo indipendente e standard in
letteratura per content-based filtering: rappresentazione bag-of-words delle opere
(nazionalita' + classificazione + tag) pesata con TF-IDF, profilo utente rappresentato
come "documento" costruito dalle sue preferenze dichiarate, similarita' coseno tra
profilo e opere. La fascia temporale (period_center) non e' inclusa nel bag-of-words
testuale (non e' un token discreto) ed e' trattata come filtro/boost separato, blando,
per non riprodurre la stessa logica esatta della formula di ground truth.

Uso:
    python recommender_baseline.py \
        --objects data/met_objects.jsonl \
        --profiles data/synthetic_users.json \
        --relevance results/user_relevance_sets.json \
        --top_k 10
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from generate_user_profiles import extract_nationalities, extract_tags, extract_year


# Boost blando per items entro la fascia temporale di interesse del profilo — separato
# dal bag-of-words testuale apposta, per non replicare la formula pesata del ground truth.
PERIOD_BOOST_WEIGHT = 0.15


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_item_document(record):
    """Rappresentazione bag-of-words di un'opera: nazionalita' + classificazione + tag,
    tutti come token stringa unici (prefissati per evitare collisioni fra dimensioni,
    es. una nazionalita' che coincide testualmente con un tag)."""
    tokens = []
    tokens += [f"nat::{n}" for n in extract_nationalities(record)]
    if record.get("classification"):
        tokens.append(f"cls::{record['classification']}")
    tokens += [f"tag::{t}" for t in extract_tags(record)]
    return tokens


def build_user_document(profile):
    """Rappresentazione bag-of-words del profilo utente, dalle preferenze dichiarate."""
    tokens = []
    tokens += [f"nat::{n}" for n in profile["preferred_nationalities"]]
    tokens += [f"cls::{c}" for c in profile["preferred_classifications"]]
    tokens += [f"tag::{t}" for t in profile["preferred_tags"]]
    return tokens


def build_idf(item_documents):
    """IDF classico: log(N / df) con smoothing +1 per evitare divisioni per zero."""
    n_docs = len(item_documents)
    df = Counter()
    for doc in item_documents:
        df.update(set(doc))
    return {token: math.log(n_docs / (1 + freq)) + 1.0 for token, freq in df.items()}


def vectorize(doc, idf):
    """Vettore TF-IDF sparso come dizionario {token: peso}. TF = conteggio grezzo nel
    documento (i documenti qui sono corti, non serve normalizzazione di lunghezza)."""
    tf = Counter(doc)
    return {token: count * idf.get(token, 0.0) for token, count in tf.items()}


def cosine_similarity(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def period_boost(profile, record):
    """Boost blando in [0, 1] se l'opera cade nella fascia temporale del profilo —
    trattato separatamente dal TF-IDF testuale, non e' la stessa formula del ground truth."""
    year = extract_year(record)
    if year is None:
        return 0.5
    dist = abs(year - profile["period_center"])
    return max(0.0, 1.0 - dist / profile["period_bandwidth"])


def recommend_for_profile(profile, item_ids, item_vectors, records_by_id, idf, top_k):
    user_vec = vectorize(build_user_document(profile), idf)
    scored = []
    for oid in item_ids:
        sim = cosine_similarity(user_vec, item_vectors[oid])
        boost = period_boost(profile, records_by_id[oid])
        final_score = (1 - PERIOD_BOOST_WEIGHT) * sim + PERIOD_BOOST_WEIGHT * boost
        scored.append((oid, final_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# --- Metriche di valutazione -----------------------------------------------------

def precision_at_k(recommended_ids, relevant_ids, k):
    top = recommended_ids[:k]
    if not top:
        return 0.0
    hits = len(set(top) & set(relevant_ids))
    return hits / len(top)


def recall_at_k(recommended_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top = recommended_ids[:k]
    hits = len(set(top) & set(relevant_ids))
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, oid in enumerate(recommended_ids[:k]):
        if oid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # posizioni 1-indexed -> log2(rank+1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, default="data/met_objects.jsonl")
    parser.add_argument("--profiles", type=str, default="data/synthetic_users.json")
    parser.add_argument("--relevance", type=str, default="results/user_relevance_sets.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/baseline_recommendations.json")
    parser.add_argument("--metrics_output", type=str, default="results/baseline_metrics.json")
    args = parser.parse_args()

    records = load_jsonl(args.objects)
    profiles = load_json(args.profiles)
    relevance_sets = load_json(args.relevance)

    records_by_id = {r["objectID"]: r for r in records}
    item_ids = list(records_by_id.keys())

    print(f"Opere: {len(records)} | Profili: {len(profiles)}")

    item_documents = {oid: build_item_document(records_by_id[oid]) for oid in item_ids}
    idf = build_idf(list(item_documents.values()))
    item_vectors = {oid: vectorize(doc, idf) for oid, doc in item_documents.items()}

    all_recommendations = {}
    precisions, recalls, ndcgs = [], [], []

    for profile in profiles:
        uid = str(profile["user_id"])
        top = recommend_for_profile(profile, item_ids, item_vectors, records_by_id, idf, args.top_k)
        recommended_ids = [oid for oid, _ in top]
        all_recommendations[uid] = [{"objectID": oid, "score": score} for oid, score in top]

        relevant_ids = relevance_sets.get(uid, [])
        precisions.append(precision_at_k(recommended_ids, relevant_ids, args.top_k))
        recalls.append(recall_at_k(recommended_ids, relevant_ids, args.top_k))
        ndcgs.append(ndcg_at_k(recommended_ids, relevant_ids, args.top_k))

    metrics = {
        "top_k": args.top_k,
        "n_users": len(profiles),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_recommendations, f, ensure_ascii=False, indent=2)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nMetriche baseline (top_k={args.top_k}):")
    print(f"  precision@{args.top_k}: {metrics['precision_at_k']:.3f}")
    print(f"  recall@{args.top_k}:    {metrics['recall_at_k']:.3f}")
    print(f"  ndcg@{args.top_k}:      {metrics['ndcg_at_k']:.3f}")
    print(f"\nRaccomandazioni salvate in {args.output}")
    print(f"Metriche salvate in {args.metrics_output}")


if __name__ == "__main__":
    main()
