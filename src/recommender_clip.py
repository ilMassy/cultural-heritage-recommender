"""
recommender_clip.py
Recommender content-based in spazio CLIP, valutato con le stesse metriche, gli stessi
utenti e lo stesso ground truth della baseline TF-IDF.

Schema:
  1. Profilo utente -> prompt testuale (nazionalita', classificazioni, tag) -> embedding
     testuale CLIP q (normalizzato). Il periodo NON entra nel prompt (CLIP non lo
     rappresenta bene): e' un boost separato, come nella baseline.
  2. Similarita' coseno di q con
       - gli embedding immagine di tutte le opere   -> s_img
       - gli embedding testuali delle opere         -> s_txt
     Ogni serie e' normalizzata min-max per utente sul catalogo (i coseni CLIP sono
     compressi in un intervallo stretto e non sono confrontabili con il boost).
  3. score = (1 - w_p) * [alpha * s_img + (1 - alpha) * s_txt] + w_p * period_boost
     con w_p = 0.15 di default (stesso peso della baseline TF-IDF).
     alpha = 0: solo testo | alpha = 1: solo immagine.
  4. Ranking, top-k, metriche (precision@k, recall@k, NDCG@k).

Prerequisito: python src/embed_clip.py (crea data/clip_cache/).

Uso:
    python src/recommender_clip.py --top_k 10
    python src/recommender_clip.py --alphas 0 0.25 0.5 0.75 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from embed_clip import encode_text, normalize
from random_baseline import evaluate, load_catalog, load_relevance_sets


# ------------------------------------------------------------------ profili
def build_prompt(profile):
    """Prompt testuale dalle preferenze dichiarate (il periodo e' escluso di proposito)."""
    parts = []
    classes = " and ".join(c.lower() for c in profile.get("preferred_classifications", []))
    nats = " and ".join(profile.get("preferred_nationalities", []))
    tags = ", ".join(profile.get("preferred_tags", []))
    head = classes or "artworks"
    if nats:
        head += f" by {nats} artists"
    parts.append(head)
    if tags:
        parts.append(f"depicting {tags}")
    return ", ".join(parts)


def period_boost(year, center, bandwidth):
    """Stessa forma del period_score del ground truth e del boost della baseline:
    1 - |anno - centro| / ampiezza, troncato a 0; 0.5 se l'anno e' ignoto."""
    if year is None:
        return 0.5
    return max(0.0, 1.0 - abs(year - center) / bandwidth)


# ------------------------------------------------------------------ scoring
def minmax(x):
    """Normalizzazione min-max per riga (per utente, sul catalogo)."""
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def score_users(Q, img, txt, boosts, alpha, period_weight):
    """Q: (U, D) prompt normalizzati | img, txt: (N, D) | boosts: (U, N) -> (U, N)."""
    s_img = minmax(Q @ img.T)
    s_txt = minmax(Q @ txt.T)
    clip_score = alpha * s_img + (1.0 - alpha) * s_txt
    return (1.0 - period_weight) * clip_score + period_weight * boosts


def top_k_ids(scores_row, item_ids, k):
    order = np.argsort(-scores_row, kind="stable")[:k]
    return [item_ids[i] for i in order]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="data/synthetic_users.json")
    ap.add_argument("--data", default="data/met_objects.jsonl")
    ap.add_argument("--relevance", default="results/user_relevance_sets.json")
    ap.add_argument("--cache_dir", default="data/clip_cache")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--period_weight", type=float, default=0.15)
    ap.add_argument("--output", default="results/clip_metrics.json")
    ap.add_argument("--recs_output", default="results/clip_recommendations.json")
    ap.add_argument("--tfidf_metrics", default="results/baseline_metrics.json")
    ap.add_argument("--random_metrics", default="results/random_baseline_metrics.json")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    item_ids = json.load(open(cache / "object_ids.json", encoding="utf-8"))
    image_ids = json.load(open(cache / "image_object_ids.json", encoding="utf-8"))
    if item_ids != image_ids:
        raise SystemExit("object_ids.json e image_object_ids.json non coincidono: "
                         "ricalcola gli embedding o gestisci le opere senza immagine.")
    txt = np.load(cache / "text_embeddings.npy")
    img = np.load(cache / "image_embeddings.npy")
    assert txt.shape == img.shape and len(item_ids) == txt.shape[0]

    catalog = {it["objectID"]: it for it in load_catalog(args.data)}
    missing = [oid for oid in item_ids if oid not in catalog]
    if missing:
        raise SystemExit(f"{len(missing)} ID della cache non sono nel catalogo")
    years = [catalog[oid].get("objectBeginDate") for oid in item_ids]

    profiles = json.load(open(args.profiles, encoding="utf-8"))
    rel_sets = load_relevance_sets(args.relevance)
    users = [str(p["user_id"]) for p in profiles]
    if set(users) != set(rel_sets):
        raise SystemExit("gli user_id dei profili e dei relevance set non coincidono")

    # -------------------------------------------- embedding dei prompt utente
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    prompts = [build_prompt(p) for p in profiles]
    inputs = processor(text=prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        Q = encode_text(model, inputs).cpu().numpy()
    Q = normalize(Q)

    print("Esempi di prompt utente:")
    for u, pr in list(zip(users, prompts))[:3]:
        print(f"  utente {u}: {pr}")

    boosts = np.array([
        [period_boost(y, p["period_center"], p.get("period_bandwidth", 50)) for y in years]
        for p in profiles
    ])

    # --------------------------------------------------------- valutazione
    k = args.top_k
    n_rel = float(np.mean([len(v) for v in rel_sets.values()]))
    max_rec = k / n_rel
    results = {"config": {"top_k": k, "n_users": len(users), "model": args.model,
                          "period_weight": args.period_weight,
                          "alphas": args.alphas, "normalization": "min-max per utente"},
               "max_recall_at_k": max_rec, "by_alpha": {}}
    all_recs = {}

    for alpha in args.alphas:
        S = score_users(Q, img, txt, boosts, alpha, args.period_weight)
        recs = {u: top_k_ids(S[i], item_ids, k) for i, u in enumerate(users)}
        results["by_alpha"][str(alpha)] = evaluate(recs, rel_sets, k)
        all_recs[str(alpha)] = recs

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(args.recs_output, "w", encoding="utf-8") as f:
        json.dump(all_recs, f, ensure_ascii=False)

    # ------------------------------------------------------------- stampa
    print(f"\nk={k} | utenti={len(users)} | recall massima teorica={max_rec:.3f} "
          f"| peso periodo={args.period_weight}")
    print(f"{'metodo':<26}{'P@k':>8}{'std':>8}{'R norm.':>9}{'NDCG@k':>9}")

    def row(name, m, std=None):
        s = f"{std:>8.3f}" if std is not None else f"{'-':>8}"
        print(f"{name:<26}{m['precision_at_k']:>8.3f}{s}"
              f"{m['recall_at_k'] / max_rec:>9.2f}{m['ndcg_at_k']:>9.3f}")

    if Path(args.random_metrics).exists():
        row("random", json.load(open(args.random_metrics))["random"])
    if Path(args.tfidf_metrics).exists():
        row("TF-IDF (baseline)", json.load(open(args.tfidf_metrics)))
    for alpha in args.alphas:
        label = {0.0: "solo testo", 1.0: "solo immagine"}.get(alpha, "mista")
        m = results["by_alpha"][str(alpha)]
        row(f"CLIP a={alpha:g} ({label})", m, m["std_users"]["precision_at_k"])
    print(f"\nSalvato: {args.output}, {args.recs_output}")
    print("Nota: 'std' = deviazione standard di P@k tra utenti (le righe random e "
          "TF-IDF non la riportano nello stesso modo).")


if __name__ == "__main__":
    main()
