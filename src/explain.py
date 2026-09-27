"""
explain.py — Modulo di spiegabilità per il recommender system (beni culturali, MET).

Copre il punto 9 della roadmap (REPORT_AVANZAMENTO.md, sez. 10):
  - TF-IDF: contributo di ogni token (nat::, cls::, tag::) al coseno + boost periodo
  - CLIP:   contributo di ogni singola preferenza del profilo (leave-one-out) e
            confronto tra componente testuale (alpha=0) e visiva (alpha=1)

Va messo in src/, accanto agli altri script (usa gli stessi import relativi).

Uso:
    python src/explain.py --user_ids 0 1 2 --output results/explanations.md
    python src/explain.py --user_ids 0 --skip_clip     # solo TF-IDF, niente GPU/modello
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from recommender_baseline import (
    build_item_document, build_user_document, build_idf, vectorize,
    cosine_similarity, period_boost as tfidf_period_boost,
    PERIOD_BOOST_WEIGHT, load_jsonl, load_json, recommend_for_profile,
)
from embed_clip import item_text, encode_text as clip_encode_text, normalize
from recommender_clip import build_prompt, period_boost as clip_period_boost, minmax


# ---------------------------------------------------------------------------
# 1. TF-IDF: decomposizione ESATTA del coseno in contributi per token
# ---------------------------------------------------------------------------
#
# cos(u, v) = (u . v) / (|u| |v|) = sum_i [ u_i * v_i / (|u| |v|) ]
#
# È una somma esatta sui token in comune (esattamente come cosine_similarity() in
# recommender_baseline.py, qui non riscritta ma richiamata direttamente per il
# punteggio totale, così i numeri combaciano sempre con quelli del recommender vero).

def explain_tfidf(profile, record, idf, period_weight=PERIOD_BOOST_WEIGHT, top_n=8):
    user_vec = vectorize(build_user_document(profile), idf)
    item_vec = vectorize(build_item_document(record), idf)

    norm_u = math.sqrt(sum(v * v for v in user_vec.values()))
    norm_v = math.sqrt(sum(v * v for v in item_vec.values()))
    denom = (norm_u * norm_v) or 1.0

    common = set(user_vec) & set(item_vec)
    contributions = [
        {
            "token": t,
            "user_weight": user_vec[t],
            "item_weight": item_vec[t],
            "contribution": user_vec[t] * item_vec[t] / denom,
        }
        for t in common
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    cosine_total = cosine_similarity(user_vec, item_vec)  # stessa funzione del recommender
    boost = tfidf_period_boost(profile, record)
    final_score = (1 - period_weight) * cosine_total + period_weight * boost

    by_type = {"nat": 0.0, "cls": 0.0, "tag": 0.0}
    for c in contributions:
        prefix = c["token"].split("::", 1)[0]
        if prefix in by_type:
            by_type[prefix] += c["contribution"]

    return {
        "user_id": profile["user_id"],
        "item_id": record["objectID"],
        "item_title": record.get("title"),
        "cosine_total": cosine_total,
        "period_boost": boost,
        "period_weight": period_weight,
        "final_score": final_score,
        "contribution_by_type": by_type,
        "top_contributions": contributions[:top_n],
    }


def format_tfidf_explanation(exp):
    lines = [
        f"**TF-IDF — utente {exp['user_id']}, opera {exp['item_id']}** "
        f"({exp['item_title'] or 'senza titolo'})",
        f"- score finale = (1-w)·coseno + w·period_boost = "
        f"{1-exp['period_weight']:.2f}·{exp['cosine_total']:.3f} + "
        f"{exp['period_weight']:.2f}·{exp['period_boost']:.3f} = **{exp['final_score']:.3f}**",
        "- Contributo per categoria (somma = coseno totale, non lo score finale):",
    ]
    for k, v in exp["contribution_by_type"].items():
        lines.append(f"  - {k}: {v:+.3f}")
    lines.append("- Token che pesano di più:")
    for c in exp["top_contributions"]:
        lines.append(f"  - `{c['token']}` — contributo {c['contribution']:+.4f} "
                     f"(peso utente {c['user_weight']:.3f}, peso opera {c['item_weight']:.3f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. CLIP: contributo per singola preferenza (leave-one-out) + confronto alpha
# ---------------------------------------------------------------------------
#
# Il prompt utente è un'unica frase (build_prompt di recommender_clip.py): non esiste
# una decomposizione esatta del coseno di un embedding di frase in "quota per
# preferenza" (9.14 del report). Attribuzione APPROSSIMATA per leave-one-out: si
# ricostruisce il prompt SENZA una preferenza alla volta, si ricalcola l'embedding e la
# similarità con l'opera (con la STESSA minmax() per-utente usata nel ranking reale),
# e si misura la caduta rispetto al prompt completo. Dichiarato come proxy, non come
# decomposizione esatta — da scrivere così anche nel report finale.

def _clip_prompt_row_scores(prompt_text, model, processor, device, image_embeddings, text_embeddings):
    """Ricalcola s_img e s_txt per TUTTO il catalogo con lo stesso minmax() per-riga
    usato in recommender_clip.py — necessario perché la normalizzazione è per-utente
    sull'intero catalogo, non per singolo item."""
    import torch  # import locale: non serve per la parte TF-IDF-only (--skip_clip)

    inputs = processor(text=[prompt_text], return_tensors="pt", padding=True,
                       truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        q = clip_encode_text(model, inputs).cpu().numpy()
    q = normalize(q)  # (1, D)
    s_img_row = minmax(q @ image_embeddings.T)[0]  # (N,)
    s_txt_row = minmax(q @ text_embeddings.T)[0]    # (N,)
    return s_img_row, s_txt_row


def explain_clip(profile, item_id, model, processor, device,
                 text_embeddings, image_embeddings, item_ids, years,
                 alpha=0.5, period_weight=0.15, top_n=6):
    idx = item_ids.index(item_id)

    full_prompt = build_prompt(profile)
    s_img_row, s_txt_row = _clip_prompt_row_scores(
        full_prompt, model, processor, device, image_embeddings, text_embeddings)
    s_img = float(s_img_row[idx])
    s_txt = float(s_txt_row[idx])

    boost = clip_period_boost(years[idx], profile["period_center"], profile.get("period_bandwidth", 50))
    combined = alpha * s_img + (1 - alpha) * s_txt
    final_score = (1 - period_weight) * combined + period_weight * boost

    contributions = []
    for kind, key in [("nationality", "preferred_nationalities"),
                       ("classification", "preferred_classifications"),
                       ("tag", "preferred_tags")]:
        for value in profile[key]:
            reduced = dict(profile)
            reduced[key] = [v for v in profile[key] if v != value]
            reduced_prompt = build_prompt(reduced)
            _, s_txt_reduced_row = _clip_prompt_row_scores(
                reduced_prompt, model, processor, device, image_embeddings, text_embeddings)
            s_txt_reduced = float(s_txt_reduced_row[idx])
            contributions.append({
                "preference": f"{kind}: {value}",
                "sim_with": s_txt,
                "sim_without": s_txt_reduced,
                "marginal_drop": s_txt - s_txt_reduced,
            })
    contributions.sort(key=lambda c: abs(c["marginal_drop"]), reverse=True)

    return {
        "user_id": profile["user_id"],
        "item_id": item_id,
        "alpha": alpha,
        "s_text": s_txt,
        "s_image": s_img,
        "period_boost": boost,
        "period_weight": period_weight,
        "final_score": final_score,
        "preference_contributions": contributions[:top_n],
    }


def format_clip_explanation(exp):
    lines = [
        f"**CLIP — utente {exp['user_id']}, opera {exp['item_id']}** (alpha={exp['alpha']})",
        f"- s_testo={exp['s_text']:.3f}  s_immagine={exp['s_image']:.3f}  "
        f"period_boost={exp['period_boost']:.3f} (peso {exp['period_weight']})",
        f"- score finale = **{exp['final_score']:.3f}**",
        "- Preferenze che spiegano di più il punteggio testuale (leave-one-out, proxy — vedi nota 9.14):",
    ]
    for c in exp["preference_contributions"]:
        lines.append(
            f"  - {c['preference']} — contributo stimato {c['marginal_drop']:+.4f} "
            f"(con: {c['sim_with']:.3f} → senza: {c['sim_without']:.3f})"
        )
    return "\n".join(lines)


def compare_alpha_extremes(exp):
    """Riusa s_text/s_image già calcolati in explain_clip — non richiede altre chiamate
    al modello, perché alpha cambia solo il peso della combinazione, non il prompt."""
    w = exp["period_weight"]
    score_text_only = (1 - w) * exp["s_text"] + w * exp["period_boost"]
    score_image_only = (1 - w) * exp["s_image"] + w * exp["period_boost"]
    return {
        "user_id": exp["user_id"],
        "item_id": exp["item_id"],
        "score_alpha0_text": score_text_only,
        "score_alpha1_image": score_image_only,
        "difference": score_text_only - score_image_only,
    }


def format_alpha_comparison(cmp):
    return (
        f"**Confronto alpha=0 vs alpha=1 — utente {cmp['user_id']}, opera {cmp['item_id']}**\n"
        f"- alpha=0 (solo testo): {cmp['score_alpha0_text']:.3f}\n"
        f"- alpha=1 (solo immagine): {cmp['score_alpha1_image']:.3f}\n"
        f"- differenza (testo - immagine): {cmp['difference']:+.3f}\n"
        f"- Nota: differenza a livello di singolo item, solo aneddotica per il report — "
        f"la conclusione di sistema resta quella del test appaiato su 50 utenti "
        f"(sez. 5.2/9.13, p=0.106/0.732, non significativo)."
    )


# ---------------------------------------------------------------------------
# 3. Main — genera 2-3 esempi concreti (utente, opera, spiegazione) in markdown
# ---------------------------------------------------------------------------

def idf_summary_by_prefix(idf):
    """IDF medio/mediano per categoria di token — quantifica quanto i tag (rari, molti
    vocabolario) pesino strutturalmente di più di nat/cls (pochi valori, alta frequenza)."""
    buckets = {"nat": [], "cls": [], "tag": []}
    for token, w in idf.items():
        prefix = token.split("::", 1)[0]
        if prefix in buckets:
            buckets[prefix].append(w)
    summary = {}
    for prefix, values in buckets.items():
        if values:
            summary[prefix] = {
                "n_token_distinti": len(values),
                "idf_medio": float(np.mean(values)),
                "idf_mediano": float(np.median(values)),
                "idf_max": float(np.max(values)),
            }
        else:
            summary[prefix] = {"n_token_distinti": 0, "idf_medio": 0.0, "idf_mediano": 0.0, "idf_max": 0.0}
    return summary


def overlap_diagnostics(profiles, item_ids_all, item_vectors, records_by_id, idf,
                        period_weight=PERIOD_BOOST_WEIGHT):
    """Per ogni utente, guarda l'opera top-1 raccomandata da TF-IDF e verifica se
    condivide almeno un token nat::/cls::/tag:: col profilo — su tutti gli utenti, non
    solo 2-3 esempi aneddotici. Risponde alla domanda: quanto spesso il top-1 ignora
    completamente una delle tre dimensioni del profilo?"""
    counts = {"nat": 0, "cls": 0, "tag": 0}
    n_users = 0
    per_user_rows = []
    for profile in profiles:
        uid = str(profile["user_id"])
        top = recommend_for_profile(profile, item_ids_all, item_vectors, records_by_id,
                                    idf, 1, period_weight=period_weight)
        if not top:
            continue
        item_id, score = top[0]
        item_vec = item_vectors[item_id]
        user_vec = vectorize(build_user_document(profile), idf)
        common = set(user_vec) & set(item_vec)
        has = {"nat": False, "cls": False, "tag": False}
        for t in common:
            prefix = t.split("::", 1)[0]
            if prefix in has:
                has[prefix] = True
        for k in counts:
            counts[k] += int(has[k])
        n_users += 1
        per_user_rows.append((uid, item_id, has["nat"], has["cls"], has["tag"]))

    fractions = {k: (v / n_users if n_users else 0.0) for k, v in counts.items()}
    return fractions, n_users, per_user_rows


def mcnemar_test(rows, key_a, key_b):
    """Test di McNemar (chi2 con correzione di continuità, 1 g.l.) per confrontare due
    proporzioni APPAIATE sugli stessi utenti (es. nat_match vs tag_match). Non richiede
    scipy: l'area della coda del chi2 a 1 g.l. si calcola in forma chiusa con erfc."""
    b = sum(1 for r in rows if r[key_a] and not r[key_b])
    c = sum(1 for r in rows if not r[key_a] and r[key_b])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(chi2 / 2))  # P(X > chi2) per chi2_1
    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value}


def print_diagnostics(idf, profiles, item_ids_all, item_vectors, records_by_id,
                      period_weight=PERIOD_BOOST_WEIGHT):
    print("\n=== Diagnostica: IDF medio per categoria di token ===")
    summary = idf_summary_by_prefix(idf)
    for prefix, s in summary.items():
        print(f"  {prefix:4s}: {s['n_token_distinti']:4d} token distinti | "
             f"IDF medio={s['idf_medio']:.3f} | mediano={s['idf_mediano']:.3f} | max={s['idf_max']:.3f}")

    print("\n=== Diagnostica: quanti top-1 condividono almeno un token con l'utente ===")
    fractions, n_users, rows = overlap_diagnostics(
        profiles, item_ids_all, item_vectors, records_by_id, idf, period_weight)
    for k, frac in fractions.items():
        print(f"  match su {k:4s}: {frac*100:.1f}% degli utenti ({int(round(frac*n_users))}/{n_users})")
    print(f"\n  Peso nel ground truth (per confronto, sez. 3.2): nationality=0.25, "
         f"classification=0.15, tags=0.40 — se il match% di 'nat' è molto sotto quello "
         f"di 'tag' nonostante pesi quasi il doppio di 'cls', è un segnale che il TF-IDF "
         f"non riflette i pesi del ground truth ma la rarità (IDF) dei token.")

    print("\n=== Test di McNemar (proporzioni appaiate sugli stessi utenti) ===")
    per_user_dicts = [{"nat_match": n, "cls_match": c, "tag_match": t} for _, _, n, c, t in rows]
    mcnemar_results = {}
    for a, b_ in [("nat_match", "tag_match"), ("nat_match", "cls_match"), ("cls_match", "tag_match")]:
        res = mcnemar_test(per_user_dicts, a, b_)
        sig = "SIGNIFICATIVO" if res["p_value"] < 0.05 else "non significativo"
        print(f"  {a} vs {b_}: discordanti a favore di {b_}={res['c']}, a favore di {a}={res['b']}, "
             f"p={res['p_value']:.4g} ({sig})")
        mcnemar_results[f"{a}_vs_{b_}"] = res

    return summary, fractions, n_users, rows, mcnemar_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="data/met_objects.jsonl")
    ap.add_argument("--profiles", default="data/synthetic_users.json")
    ap.add_argument("--relevance", default="results/user_relevance_sets.json")
    ap.add_argument("--cache_dir", default="data/clip_cache")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--user_ids", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--top_k_per_user", type=int, default=1)
    ap.add_argument("--period_weight_tfidf", type=float, default=PERIOD_BOOST_WEIGHT)
    ap.add_argument("--period_weight_clip", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--skip_clip", action="store_true",
                    help="Genera solo le spiegazioni TF-IDF (non serve GPU/modello CLIP).")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Stampa statistiche IDF per categoria e % di match nat/cls/tag "
                         "nel top-1 su TUTTI gli utenti (non solo --user_ids).")
    ap.add_argument("--diagnostics_output", default="results/tfidf_diagnostics.json")
    ap.add_argument("--output", default="results/explanations.md")
    args = ap.parse_args()

    # ------------------------------------------------------------ dati comuni
    records = load_jsonl(args.objects)
    records_by_id = {r["objectID"]: r for r in records}
    item_ids_all = list(records_by_id.keys())
    profiles = load_json(args.profiles)
    profiles_by_id = {str(p["user_id"]): p for p in profiles}
    relevance_sets = load_json(args.relevance)

    item_documents = {oid: build_item_document(records_by_id[oid]) for oid in item_ids_all}
    idf = build_idf(list(item_documents.values()))
    item_vectors = {oid: vectorize(doc, idf) for oid, doc in item_documents.items()}

    if args.diagnostics:
        summary, fractions, n_users, rows, mcnemar_results = print_diagnostics(
            idf, profiles, item_ids_all, item_vectors, records_by_id,
            period_weight=args.period_weight_tfidf)
        Path(args.diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.diagnostics_output, "w", encoding="utf-8") as f:
            json.dump({
                "idf_by_prefix": summary,
                "top1_match_fraction": fractions,
                "n_users": n_users,
                "mcnemar_tests": mcnemar_results,
                "per_user": [{"user_id": u, "top1_item": it, "nat_match": n, "cls_match": c, "tag_match": t}
                             for u, it, n, c, t in rows],
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSalvato: {args.diagnostics_output}")

    # ------------------------------------------------------------ CLIP (opzionale)
    model = processor = device = None
    text_embeddings = image_embeddings = clip_item_ids = years = None
    if not args.skip_clip:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        cache = Path(args.cache_dir)
        clip_item_ids = json.load(open(cache / "object_ids.json", encoding="utf-8"))
        image_ids = json.load(open(cache / "image_object_ids.json", encoding="utf-8"))
        if clip_item_ids != image_ids:
            raise SystemExit("object_ids.json e image_object_ids.json non coincidono — "
                             "stesso vincolo di recommender_clip.py.")
        text_embeddings = np.load(cache / "text_embeddings.npy")
        image_embeddings = np.load(cache / "image_embeddings.npy")
        years = [records_by_id[oid].get("objectBeginDate") for oid in clip_item_ids]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(args.model).to(device).eval()
        processor = CLIPProcessor.from_pretrained(args.model)

    # ------------------------------------------------------------ per utente
    report_sections = []
    for uid in args.user_ids:
        if uid not in profiles_by_id:
            print(f"ATTENZIONE: utente {uid} non trovato in {args.profiles}, salto.")
            continue
        profile = profiles_by_id[uid]

        top_tfidf = recommend_for_profile(
            profile, item_ids_all, item_vectors, records_by_id, idf,
            args.top_k_per_user, period_weight=args.period_weight_tfidf)

        section = [f"## Utente {uid}\n"]
        section.append(f"Preferenze: nazionalità {profile['preferred_nationalities']}, "
                       f"classificazioni {profile['preferred_classifications']}, "
                       f"tag {profile['preferred_tags']}, periodo centrato su "
                       f"{profile['period_center']} (±{profile.get('period_bandwidth', 50)}).\n")

        for item_id, score in top_tfidf:
            exp = explain_tfidf(profile, records_by_id[item_id], idf,
                                period_weight=args.period_weight_tfidf)
            section.append(format_tfidf_explanation(exp) + "\n")

            if not args.skip_clip and item_id in clip_item_ids:
                clip_exp = explain_clip(
                    profile, item_id, model, processor, device,
                    text_embeddings, image_embeddings, clip_item_ids, years,
                    alpha=args.alpha, period_weight=args.period_weight_clip)
                section.append(format_clip_explanation(clip_exp) + "\n")
                section.append(format_alpha_comparison(compare_alpha_extremes(clip_exp)) + "\n")
            elif not args.skip_clip:
                section.append(f"_(opera {item_id} non presente nella cache CLIP — "
                               f"probabilmente immagine non scaricata, sez. 6 del report)_\n")

        report_sections.append("\n".join(section))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Esempi di spiegazione — recommender system MET\n\n")
        f.write("\n---\n\n".join(report_sections))

    print(f"Salvato: {args.output}")


if __name__ == "__main__":
    main()
