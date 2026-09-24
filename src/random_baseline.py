"""
random_baseline.py
Baseline di confronto per il recommender MET (European Paintings):
  1. random            : k opere a caso dal catalogo, ripetuto n_runs volte
  2. popolarita_highlight : k opere a caso tra quelle con isHighlight=True
  3. popolarita_tag    : top-k opere per frequenza media dei tag (deterministica)

Stesse metriche della baseline TF-IDF: precision@k, recall@k, NDCG@k
(gain binario, IDCG con min(|rilevanti|, k)), mediate sugli utenti.
Stesso ground truth: results/user_relevance_sets.json.

Uso:
    python src/random_baseline.py --top_k 10 --n_runs 200 --seed 42
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------- caricamento
def load_catalog(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _tag_names(tags):
    """I tag del MET sono tipicamente [{'term': 'Men', ...}, ...]; gestisco anche stringhe."""
    out = []
    for t in tags or []:
        if isinstance(t, dict):
            term = t.get("term")
            if term:
                out.append(term)
        elif isinstance(t, str):
            out.append(t)
    return out


def load_relevance_sets(path):
    """Restituisce {user_id: set(objectID)}. Accetta dict o lista di dict."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def to_ids(v):
        if isinstance(v, dict):  # es. {"relevant_ids": [...]} o {"relevant": [...]}
            for key in ("relevant_ids", "relevant", "relevance_set", "objectIDs"):
                if key in v:
                    return to_ids(v[key])
            raise ValueError(f"Formato relevance set non riconosciuto: chiavi {list(v)}")
        return {int(x["objectID"]) if isinstance(x, dict) else int(x) for x in v}

    if isinstance(raw, dict):
        return {str(u): to_ids(v) for u, v in raw.items()}
    if isinstance(raw, list):
        result = {}
        for i, entry in enumerate(raw):
            uid = str(entry.get("user_id", entry.get("id", i)))
            result[uid] = to_ids(entry)
        return result
    raise ValueError("Formato user_relevance_sets.json non riconosciuto")


# ------------------------------------------------------------------- metriche
def precision_at_k(recs, rel, k):
    return sum(1 for r in recs[:k] if r in rel) / k


def recall_at_k(recs, rel, k):
    return sum(1 for r in recs[:k] if r in rel) / len(rel) if rel else 0.0


def ndcg_at_k(recs, rel, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(recs_by_user, rel_sets, k):
    """Media sugli utenti (e std tra utenti) delle tre metriche."""
    p, r, n = [], [], []
    for u, rel in rel_sets.items():
        recs = recs_by_user[u]
        p.append(precision_at_k(recs, rel, k))
        r.append(recall_at_k(recs, rel, k))
        n.append(ndcg_at_k(recs, rel, k))
    return {
        "precision_at_k": statistics.fmean(p),
        "recall_at_k": statistics.fmean(r),
        "ndcg_at_k": statistics.fmean(n),
        "std_users": {
            "precision_at_k": statistics.pstdev(p),
            "recall_at_k": statistics.pstdev(r),
            "ndcg_at_k": statistics.pstdev(n),
        },
    }


# ------------------------------------------------------------------ baseline
def run_random(all_ids, rel_sets, k, n_runs, rng, pool=None):
    """Ripete n_runs volte il campionamento casuale; ritorna media e std tra run."""
    pool = pool if pool is not None else all_ids
    per_run = {"precision_at_k": [], "recall_at_k": [], "ndcg_at_k": []}
    for _ in range(n_runs):
        recs = {u: rng.sample(pool, k) for u in rel_sets}
        m = evaluate(recs, rel_sets, k)
        for key in per_run:
            per_run[key].append(m[key])
    return {
        "n_runs": n_runs,
        **{key: statistics.fmean(vals) for key, vals in per_run.items()},
        "std_runs": {key: statistics.pstdev(vals) for key, vals in per_run.items()},
    }


def run_popularity_tags(items, rel_sets, k):
    """Non personalizzata: stessa lista top-k per tutti, ordinata per frequenza media dei tag."""
    df = Counter()
    for it in items:
        df.update(set(_tag_names(it.get("tags"))))
    scores = {}
    for it in items:
        tags = set(_tag_names(it.get("tags")))
        scores[it["objectID"]] = statistics.fmean(df[t] for t in tags) if tags else 0.0
    # tie-break deterministico su objectID
    ranked = sorted(scores, key=lambda oid: (-scores[oid], oid))
    top = ranked[:k]
    recs = {u: top for u in rel_sets}
    return evaluate(recs, rel_sets, k)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/met_objects.jsonl")
    ap.add_argument("--relevance", default="results/user_relevance_sets.json")
    ap.add_argument("--tfidf_metrics", default="results/baseline_metrics.json")
    ap.add_argument("--output", default="results/random_baseline_metrics.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n_runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = load_catalog(args.data)
    rel_sets = load_relevance_sets(args.relevance)

    all_ids = [it["objectID"] for it in items]
    highlight_ids = [it["objectID"] for it in items if it.get("isHighlight")]
    k = args.top_k
    n_rel = statistics.fmean(len(v) for v in rel_sets.values())

    # controllo di coerenza: gli ID dei relevance set devono stare nel catalogo
    catalog = set(all_ids)
    missing = sum(1 for s in rel_sets.values() for x in s if x not in catalog)
    if missing:
        print(f"ATTENZIONE: {missing} ID nei relevance set non sono nel catalogo")

    results = {
        "config": {
            "top_k": k,
            "n_users": len(rel_sets),
            "n_items": len(all_ids),
            "n_highlight": len(highlight_ids),
            "mean_relevance_set_size": n_rel,
            "n_runs": args.n_runs,
            "seed": args.seed,
        },
        "expected_random_precision": n_rel / len(all_ids),
        "max_recall_at_k": k / n_rel,
        "random": run_random(all_ids, rel_sets, k, args.n_runs, rng),
        "popularity_highlight": run_random(
            all_ids, rel_sets, k, args.n_runs, rng, pool=highlight_ids
        ),
        "popularity_tags": run_popularity_tags(items, rel_sets, k),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------- stampa
    tfidf = None
    if Path(args.tfidf_metrics).exists():
        with open(args.tfidf_metrics, encoding="utf-8") as f:
            tfidf = json.load(f)

    max_rec = results["max_recall_at_k"]
    print(f"\nk={k} | utenti={len(rel_sets)} | catalogo={len(all_ids)} | "
          f"|rilevanti| medio={n_rel:.0f} | recall massima teorica={max_rec:.3f}")
    print(f"Precision attesa random (teorica) = {results['expected_random_precision']:.3f}\n")
    print(f"{'metodo':<24}{'P@k':>8}{'R@k':>8}{'R norm.':>9}{'NDCG@k':>9}")

    def row(name, m):
        print(f"{name:<24}{m['precision_at_k']:>8.3f}{m['recall_at_k']:>8.3f}"
              f"{m['recall_at_k'] / max_rec:>9.2f}{m['ndcg_at_k']:>9.3f}")

    row("random", results["random"])
    row("popolarita_highlight", results["popularity_highlight"])
    row("popolarita_tag", results["popularity_tags"])
    if tfidf:
        row("TF-IDF (baseline)", tfidf)
    print(f"\nstd tra run (random): P={results['random']['std_runs']['precision_at_k']:.4f}")
    print(f"Salvato in {args.output}")


if __name__ == "__main__":
    main()
