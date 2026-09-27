"""
ablation_study.py — Ablation più ampio (punto 10 della roadmap, sez. 10 del report).

Copre i due assi che gli script esistenti non coprono ancora:
  1. --k_sweep: precision/recall/NDCG a k multipli (default 5,10,20,50), calcolati da
     UN'UNICA ranking per utente (non si ricalcola coseno/embedding per ogni k) — per
     TF-IDF baseline, "solo periodo", e CLIP alpha di default (se non --skip_clip).
     Rende omogenea la tabella di sez. 4.4, oggi fissa a k=10.
  2. --seed_sweep: rigenera i profili utente con più seed (default: 10, da --seed_start)
     e ricalcola precision@10 del baseline TF-IDF per ciascun seed, per stimare quanto i
     risultati dipendano dal seed=42 fissato finora in generate_user_profiles.py. Il CLIP
     è incluso solo con --seed_sweep_clip (lento: re-incorpora i prompt per ogni seed).

Per l'ablation su alpha (CLIP) NON serve questo script: recommender_clip.py supporta già
--alphas con una lista arbitraria di valori, quindi per una griglia fine basta:

    python src/recommender_clip.py --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --output results/clip_alpha_sweep.json

Uso:
    python src/ablation_study.py --k_sweep --output results/ablation_k.json
    python src/ablation_study.py --seed_sweep --n_seeds 10 --output results/ablation_seed.json
    python src/ablation_study.py --k_sweep --seed_sweep --skip_clip   # solo TF-IDF, veloce
"""

import argparse
import json
from pathlib import Path

import numpy as np

from recommender_baseline import (
    build_item_document, build_user_document, build_idf, vectorize,
    cosine_similarity, period_boost as tfidf_period_boost,
    PERIOD_BOOST_WEIGHT, load_jsonl, load_json,
    precision_at_k, recall_at_k, ndcg_at_k,
)
from generate_user_profiles import (
    build_vocabularies, generate_profile, score_item, extract_year,
)


# ---------------------------------------------------------------------------
# 1. k-sweep omogeneo — un'unica ranking per utente, metriche a più k
# ---------------------------------------------------------------------------

def full_ranking_tfidf(profile, item_ids, item_vectors, records_by_id, idf, period_weight):
    """Ranking completo (tutti gli item ordinati), non solo il top-k — le metriche a
    k diversi si calcolano tutte da questa unica lista con recommended_ids[:k]."""
    user_vec = vectorize(build_user_document(profile), idf)
    scored = []
    for oid in item_ids:
        sim = cosine_similarity(user_vec, item_vectors[oid])
        boost = tfidf_period_boost(profile, records_by_id[oid])
        final_score = (1 - period_weight) * sim + period_weight * boost
        scored.append((oid, final_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [oid for oid, _ in scored]


def k_sweep_tfidf(profiles, relevance_sets, item_ids, item_vectors, records_by_id, idf,
                  period_weight, k_values):
    """Precision/recall/NDCG a più k per un modello TF-IDF-like (baseline o solo periodo,
    a seconda di period_weight)."""
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    for profile in profiles:
        uid = str(profile["user_id"])
        ranking = full_ranking_tfidf(profile, item_ids, item_vectors, records_by_id, idf, period_weight)
        relevant = relevance_sets.get(uid, [])
        for k in k_values:
            results[k]["precision"].append(precision_at_k(ranking, relevant, k))
            results[k]["recall"].append(recall_at_k(ranking, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(ranking, relevant, k))
    return {
        k: {metric: float(np.mean(vals)) for metric, vals in m.items()}
        for k, m in results.items()
    }


def k_sweep_clip(profiles, relevance_sets, S, item_ids, k_values):
    """Come k_sweep_tfidf ma per una matrice di score CLIP (U, N) già calcolata
    (score_users() di recommender_clip.py, un solo alpha alla volta)."""
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    max_k = max(k_values)
    for i, profile in enumerate(profiles):
        uid = str(profile["user_id"])
        order = np.argsort(-S[i], kind="stable")[:max_k]
        ranking = [item_ids[j] for j in order]
        relevant = relevance_sets.get(uid, [])
        for k in k_values:
            results[k]["precision"].append(precision_at_k(ranking, relevant, k))
            results[k]["recall"].append(recall_at_k(ranking, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(ranking, relevant, k))
    return {
        k: {metric: float(np.mean(vals)) for metric, vals in m.items()}
        for k, m in results.items()
    }


# ---------------------------------------------------------------------------
# 2. seed-sweep — robustezza rispetto al seed di generazione dei profili
# ---------------------------------------------------------------------------

def regenerate_profiles_and_relevance(records, seed, n_users, relevance_top_pct):
    """Stessa logica di generate_user_profiles.py main(), isolata in funzione per poter
    essere richiamata più volte con seed diversi senza passare da file su disco."""
    rng = np.random.default_rng(seed)
    nat_counter, cls_counter, tag_counter, years = build_vocabularies(records)
    profiles = [
        generate_profile(uid, nat_counter, cls_counter, tag_counter, years, rng)
        for uid in range(n_users)
    ]
    n_relevant = max(1, int(len(records) * relevance_top_pct))
    relevance_sets = {}
    for profile in profiles:
        scored = [(r["objectID"], score_item(profile, r, rng)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        relevance_sets[str(profile["user_id"])] = [oid for oid, _ in scored[:n_relevant]]
    return profiles, relevance_sets


def seed_sweep_tfidf(records, item_ids, item_vectors, records_by_id, idf,
                     seeds, n_users, relevance_top_pct, top_k, period_weight):
    """Per ogni seed: rigenera profili + relevance, ricalcola precision/recall/NDCG@k
    del baseline TF-IDF. Riusa item_vectors/idf già calcolati una volta sola (non
    dipendono dai profili utente, solo dal catalogo — quindi non vanno rifatti a ogni seed)."""
    rows = []
    for seed in seeds:
        profiles, relevance_sets = regenerate_profiles_and_relevance(
            records, seed, n_users, relevance_top_pct)
        precisions, recalls, ndcgs = [], [], []
        for profile in profiles:
            uid = str(profile["user_id"])
            ranking = full_ranking_tfidf(profile, item_ids, item_vectors, records_by_id,
                                         idf, period_weight)[:top_k]
            relevant = relevance_sets.get(uid, [])
            precisions.append(precision_at_k(ranking, relevant, top_k))
            recalls.append(recall_at_k(ranking, relevant, top_k))
            ndcgs.append(ndcg_at_k(ranking, relevant, top_k))
        rows.append({
            "seed": seed,
            "precision_at_k": float(np.mean(precisions)),
            "recall_at_k": float(np.mean(recalls)),
            "ndcg_at_k": float(np.mean(ndcgs)),
        })
    precisions_across_seeds = [r["precision_at_k"] for r in rows]
    summary = {
        "per_seed": rows,
        "precision_mean_across_seeds": float(np.mean(precisions_across_seeds)),
        "precision_std_across_seeds": float(np.std(precisions_across_seeds)),
        "precision_min": float(np.min(precisions_across_seeds)),
        "precision_max": float(np.max(precisions_across_seeds)),
    }
    return summary


def seed_sweep_clip(records, records_by_id, clip_item_ids, text_embeddings, image_embeddings,
                    years, model, processor, device, seeds, n_users, relevance_top_pct,
                    top_k, alpha, period_weight):
    """Come seed_sweep_tfidf ma per CLIP — lento: re-incorpora i prompt per ogni seed
    (il catalogo di embedding immagine/testo non cambia, solo i profili/prompt utente)."""
    import torch
    from embed_clip import encode_text as clip_encode_text, normalize
    from recommender_clip import build_prompt, period_boost as clip_period_boost, minmax

    rows = []
    for seed in seeds:
        profiles, relevance_sets = regenerate_profiles_and_relevance(
            records, seed, n_users, relevance_top_pct)
        prompts = [build_prompt(p) for p in profiles]
        inputs = processor(text=prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            Q = clip_encode_text(model, inputs).cpu().numpy()
        Q = normalize(Q)

        boosts = np.array([
            [clip_period_boost(y, p["period_center"], p.get("period_bandwidth", 50)) for y in years]
            for p in profiles
        ])
        s_img = minmax(Q @ image_embeddings.T)
        s_txt = minmax(Q @ text_embeddings.T)
        combined = alpha * s_img + (1 - alpha) * s_txt
        S = (1 - period_weight) * combined + period_weight * boosts

        precisions, recalls, ndcgs = [], [], []
        for i, profile in enumerate(profiles):
            uid = str(profile["user_id"])
            order = np.argsort(-S[i], kind="stable")[:top_k]
            ranking = [clip_item_ids[j] for j in order]
            relevant = relevance_sets.get(uid, [])
            precisions.append(precision_at_k(ranking, relevant, top_k))
            recalls.append(recall_at_k(ranking, relevant, top_k))
            ndcgs.append(ndcg_at_k(ranking, relevant, top_k))
        rows.append({
            "seed": seed,
            "precision_at_k": float(np.mean(precisions)),
            "recall_at_k": float(np.mean(recalls)),
            "ndcg_at_k": float(np.mean(ndcgs)),
        })
    precisions_across_seeds = [r["precision_at_k"] for r in rows]
    return {
        "per_seed": rows,
        "precision_mean_across_seeds": float(np.mean(precisions_across_seeds)),
        "precision_std_across_seeds": float(np.std(precisions_across_seeds)),
        "precision_min": float(np.min(precisions_across_seeds)),
        "precision_max": float(np.max(precisions_across_seeds)),
    }


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="data/met_objects.jsonl")
    ap.add_argument("--profiles", default="data/synthetic_users.json")
    ap.add_argument("--relevance", default="results/user_relevance_sets.json")
    ap.add_argument("--cache_dir", default="data/clip_cache")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--top_k", type=int, default=10, help="k di riferimento per il seed-sweep")
    ap.add_argument("--period_weight", type=float, default=PERIOD_BOOST_WEIGHT)
    ap.add_argument("--alpha", type=float, default=0.5, help="alpha CLIP per k-sweep/seed-sweep")
    ap.add_argument("--period_weight_clip", type=float, default=0.15)
    ap.add_argument("--k_sweep", action="store_true")
    ap.add_argument("--seed_sweep", action="store_true")
    ap.add_argument("--seed_sweep_clip", action="store_true",
                    help="Include CLIP nel seed-sweep (lento: re-incorpora i prompt per ogni seed).")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed_start", type=int, default=100,
                    help="I seed usati partono da qui (evita di sovrapporsi al seed=42 già usato).")
    ap.add_argument("--n_users", type=int, default=50)
    ap.add_argument("--relevance_top_pct", type=float, default=0.05)
    ap.add_argument("--skip_clip", action="store_true")
    ap.add_argument("--output", default="results/ablation_study.json")
    args = ap.parse_args()

    records = load_jsonl(args.objects)
    records_by_id = {r["objectID"]: r for r in records}
    item_ids_all = list(records_by_id.keys())
    profiles = load_json(args.profiles)
    relevance_sets = load_json(args.relevance)

    item_documents = {oid: build_item_document(records_by_id[oid]) for oid in item_ids_all}
    idf = build_idf(list(item_documents.values()))
    item_vectors = {oid: vectorize(doc, idf) for oid, doc in item_documents.items()}

    out = {}

    # ---------------------------------------------------------------- k-sweep
    if args.k_sweep:
        print("\n=== k-sweep: TF-IDF baseline ===")
        out["k_sweep_tfidf_baseline"] = k_sweep_tfidf(
            profiles, relevance_sets, item_ids_all, item_vectors, records_by_id, idf,
            args.period_weight, args.k_values)
        for k, m in out["k_sweep_tfidf_baseline"].items():
            print(f"  k={k:3d}: P={m['precision']:.3f}  R={m['recall']:.3f}  NDCG={m['ndcg']:.3f}")

        print("\n=== k-sweep: solo periodo (w=1) ===")
        out["k_sweep_solo_periodo"] = k_sweep_tfidf(
            profiles, relevance_sets, item_ids_all, item_vectors, records_by_id, idf,
            1.0, args.k_values)
        for k, m in out["k_sweep_solo_periodo"].items():
            print(f"  k={k:3d}: P={m['precision']:.3f}  R={m['recall']:.3f}  NDCG={m['ndcg']:.3f}")

        if not args.skip_clip:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from embed_clip import encode_text as clip_encode_text, normalize
            from recommender_clip import build_prompt, period_boost as clip_period_boost, minmax

            cache = Path(args.cache_dir)
            clip_item_ids = json.load(open(cache / "object_ids.json", encoding="utf-8"))
            text_embeddings = np.load(cache / "text_embeddings.npy")
            image_embeddings = np.load(cache / "image_embeddings.npy")
            years = [records_by_id[oid].get("objectBeginDate") for oid in clip_item_ids]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(args.model).to(device).eval()
            processor = CLIPProcessor.from_pretrained(args.model)

            prompts = [build_prompt(p) for p in profiles]
            inputs = processor(text=prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                Q = clip_encode_text(model, inputs).cpu().numpy()
            Q = normalize(Q)
            boosts = np.array([
                [clip_period_boost(y, p["period_center"], p.get("period_bandwidth", 50)) for y in years]
                for p in profiles
            ])
            s_img = minmax(Q @ image_embeddings.T)
            s_txt = minmax(Q @ text_embeddings.T)
            combined = args.alpha * s_img + (1 - args.alpha) * s_txt
            S = (1 - args.period_weight_clip) * combined + args.period_weight_clip * boosts

            print(f"\n=== k-sweep: CLIP (alpha={args.alpha}) ===")
            out["k_sweep_clip"] = k_sweep_clip(profiles, relevance_sets, S, clip_item_ids, args.k_values)
            for k, m in out["k_sweep_clip"].items():
                print(f"  k={k:3d}: P={m['precision']:.3f}  R={m['recall']:.3f}  NDCG={m['ndcg']:.3f}")

    # ------------------------------------------------------------- seed-sweep
    if args.seed_sweep:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
        print(f"\n=== seed-sweep: TF-IDF baseline su {args.n_seeds} seed ({seeds[0]}..{seeds[-1]}) ===")
        out["seed_sweep_tfidf"] = seed_sweep_tfidf(
            records, item_ids_all, item_vectors, records_by_id, idf,
            seeds, args.n_users, args.relevance_top_pct, args.top_k, args.period_weight)
        s = out["seed_sweep_tfidf"]
        print(f"  precision@{args.top_k}: media={s['precision_mean_across_seeds']:.3f}  "
             f"std={s['precision_std_across_seeds']:.3f}  "
             f"[{s['precision_min']:.3f}, {s['precision_max']:.3f}]")
        print(f"  (per confronto: con seed=42 la baseline dà precision@10=0.588, sez. 4.1)")

        if args.seed_sweep_clip and not args.skip_clip:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            cache = Path(args.cache_dir)
            clip_item_ids = json.load(open(cache / "object_ids.json", encoding="utf-8"))
            text_embeddings = np.load(cache / "text_embeddings.npy")
            image_embeddings = np.load(cache / "image_embeddings.npy")
            years = [records_by_id[oid].get("objectBeginDate") for oid in clip_item_ids]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(args.model).to(device).eval()
            processor = CLIPProcessor.from_pretrained(args.model)

            print(f"\n=== seed-sweep: CLIP (alpha={args.alpha}) su {args.n_seeds} seed ===")
            out["seed_sweep_clip"] = seed_sweep_clip(
                records, records_by_id, clip_item_ids, text_embeddings, image_embeddings,
                years, model, processor, device, seeds, args.n_users, args.relevance_top_pct,
                args.top_k, args.alpha, args.period_weight_clip)
            s = out["seed_sweep_clip"]
            print(f"  precision@{args.top_k}: media={s['precision_mean_across_seeds']:.3f}  "
                 f"std={s['precision_std_across_seeds']:.3f}  "
                 f"[{s['precision_min']:.3f}, {s['precision_max']:.3f}]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSalvato: {args.output}")


if __name__ == "__main__":
    main()
