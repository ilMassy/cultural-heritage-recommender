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
testuale (non e' un token discreto) ed e' trattata come boost separato e di peso
contenuto (default 0.15).

LIMITE DICHIARATO (circolarita' parziale)
--------------------------------------------------------------------------------------
Il TF-IDF non riusa la formula pesata del ground truth, ma il boost del periodo ha la
STESSA forma funzionale del termine "periodo" del ground truth:
max(0, 1 - |anno - period_center| / period_bandwidth). Il recommender quindi conosce in
anticipo una parte dell'informazione che ha generato le etichette, e i punteggi
risultano gonfiati. Il problema non e' eliminato ma QUANTIFICATO con l'ablation su
--period_weight: 0 = solo contenuto testuale, 0.15 = baseline, 1 = solo periodo
(riga di controllo). Vedi Report_Finale_SII.md, sez. 7.2 e 9.3.

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
from metrics import precision_at_k, recall_at_k, ndcg_at_k


# Peso del boost per le opere entro la fascia temporale di interesse del profilo. Tenuto
# separato dal bag-of-words testuale e di peso contenuto, ma NON indipendente dal ground
# truth: condivide con esso la forma del termine periodo (vedi docstring del modulo).
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
    """IDF smussato: ln(N / (1 + df)) + 1, con logaritmo naturale, calcolato sulle N opere.
    Il +1 finale mantiene il peso positivo anche per i token presenti in quasi tutto il
    catalogo."""
    n_docs = len(item_documents)
    df = Counter()
    for doc in item_documents:
        df.update(set(doc))
    return {token: math.log(n_docs / (1 + freq)) + 1.0 for token, freq in df.items()}


def vectorize(doc, idf):
    """Vettore TF-IDF sparso come dizionario {token: peso}, con peso = TF * IDF.
    TF = conteggio grezzo nel documento (senza logaritmo): i documenti sono corti e i
    token compaiono di norma una sola volta, quindi non serve normalizzare per lunghezza
    (l'unica normalizzazione e' quella L2 del coseno). Anche il profilo utente e'
    vettorizzato con l'IDF del catalogo; token assenti dal catalogo hanno peso 0."""
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
    """Boost in [0, 1]: decresce linearmente con la distanza dell'anno dell'opera dal
    period_center del profilo, fino a 0 oltre period_bandwidth. E' separato dal TF-IDF
    testuale ma ha la stessa forma del termine periodo del ground truth (circolarita'
    parziale, quantificata con --period_weight). Se l'anno e' ignoto restituisce 0.5
    (ramo di fatto inattivo sul dataset usato, dove l'anno e' sempre disponibile)."""
    year = extract_year(record)
    if year is None:
        return 0.5
    dist = abs(year - profile["period_center"])
    return max(0.0, 1.0 - dist / profile["period_bandwidth"])


def recommend_for_profile(profile, item_ids, item_vectors, records_by_id, idf, top_k,
                          period_weight=PERIOD_BOOST_WEIGHT):
    user_vec = vectorize(build_user_document(profile), idf)
    scored = []
    for oid in item_ids:
        sim = cosine_similarity(user_vec, item_vectors[oid])
        boost = period_boost(profile, records_by_id[oid])
        final_score = (1 - period_weight) * sim + period_weight * boost
        scored.append((oid, final_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# --- Metriche di valutazione -----------------------------------------------------
# Spostate in metrics.py (punto 11 della roadmap): un solo posto condiviso con
# random_baseline.py e recommender_clip.py, invece di tre copie che potrebbero
# disallinearsi in futuro. Logica invariata, solo spostata — vedi metrics.py.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, default="data/met_objects.jsonl")
    parser.add_argument("--profiles", type=str, default="data/synthetic_users.json")
    parser.add_argument("--relevance", type=str, default="results/user_relevance_sets.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--period_weight", type=float, default=PERIOD_BOOST_WEIGHT,
                        help="Peso del boost del periodo in [0, 1]: score = (1-w)*coseno + w*boost. "
                             "Default 0.15 (baseline). 0 = solo TF-IDF (isola il contenuto); "
                             "1 = ranking solo per periodo (riga di controllo: e' la stessa funzione "
                             "del ground truth, non un recommender).")
    parser.add_argument("--output", type=str, default="results/baseline_recommendations.json")
    parser.add_argument("--metrics_output", type=str, default="results/baseline_metrics.json")
    args = parser.parse_args()

    if not 0.0 <= args.period_weight <= 1.0:
        raise ValueError("--period_weight deve essere in [0, 1].")

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
        top = recommend_for_profile(profile, item_ids, item_vectors, records_by_id, idf, args.top_k,
                                    period_weight=args.period_weight)
        recommended_ids = [oid for oid, _ in top]
        all_recommendations[uid] = [{"objectID": oid, "score": score} for oid, score in top]

        relevant_ids = relevance_sets.get(uid, [])
        precisions.append(precision_at_k(recommended_ids, relevant_ids, args.top_k))
        recalls.append(recall_at_k(recommended_ids, relevant_ids, args.top_k))
        ndcgs.append(ndcg_at_k(recommended_ids, relevant_ids, args.top_k))

    metrics = {
        "top_k": args.top_k,
        "period_weight": args.period_weight,
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

    print(f"\nMetriche baseline (top_k={args.top_k}, peso periodo={args.period_weight}):")
    print(f"  precision@{args.top_k}: {metrics['precision_at_k']:.3f}")
    print(f"  recall@{args.top_k}:    {metrics['recall_at_k']:.3f}")
    print(f"  ndcg@{args.top_k}:      {metrics['ndcg_at_k']:.3f}")
    print(f"\nRaccomandazioni salvate in {args.output}")
    print(f"Metriche salvate in {args.metrics_output}")


if __name__ == "__main__":
    main()
