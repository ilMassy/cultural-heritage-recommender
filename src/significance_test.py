"""
Test di significatività appaiati per il progetto cultural-heritage-recommender.

Confronta, per ogni utente, precision@10 tra coppie di configurazioni, usando:
- Wilcoxon signed-rank test (appaiato, non parametrico)
- Intervallo di confidenza bootstrap (95%) sulla differenza media di P@10

Richiede solo i file *_recommendations*.json già generati (nessuna GPU/modello).
Uso: python significance_test.py
Dipendenze: numpy, scipy  (pip install numpy scipy --break-system-packages, se mancano)
"""

import json
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURAZIONE: adatta i path se i tuoi file sono in una cartella diversa
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
K = 10
N_USERS = 50
N_BOOTSTRAP = 10000
SEED = 42

RELEVANCE_FILE = f"{RESULTS_DIR}/user_relevance_sets.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _extract_id(item):
    """Un item di raccomandazione può essere un int (objectID diretto)
    o un dict tipo {'objectID': ..., 'score': ...}."""
    if isinstance(item, dict):
        return item.get("objectID", item.get("object_id", item.get("id")))
    return item


def precision_per_user(recs_by_user, relevance_sets, k=K):
    """Restituisce un array numpy di P@k, uno per utente (ordine per user_id 0..N-1)."""
    scores = []
    for uid in sorted(relevance_sets.keys(), key=int):
        raw_recs = recs_by_user.get(str(uid), recs_by_user.get(uid, []))[:k]
        recs = [_extract_id(item) for item in raw_recs]
        relevant = set(relevance_sets[uid])
        hit = sum(1 for item in recs if item in relevant)
        scores.append(hit / k)
    return np.array(scores)


def paired_test(name_a, p_a, name_b, p_b):
    diff = p_a - p_b
    mean_diff = diff.mean()

    # Wilcoxon signed-rank (salta se tutte le differenze sono zero)
    if np.allclose(diff, 0):
        wilcoxon_stat, wilcoxon_p = float("nan"), 1.0
    else:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(p_a, p_b)

    # Bootstrap sulla differenza media (resampling per utente, con reinserimento)
    rng = np.random.default_rng(SEED)
    n = len(diff)
    boot_means = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot_means[i] = diff[idx].mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n{name_a}  vs  {name_b}")
    print(f"  media {name_a}: {p_a.mean():.3f} (std {p_a.std():.3f})")
    print(f"  media {name_b}: {p_b.mean():.3f} (std {p_b.std():.3f})")
    print(f"  differenza media: {mean_diff:+.3f}")
    print(f"  IC bootstrap 95% sulla differenza: [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"  Wilcoxon: stat={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}"
          f" -> {'SIGNIFICATIVO (p<0.05)' if wilcoxon_p < 0.05 else 'non significativo'}")
    return {
        "mean_diff": mean_diff,
        "ci_95": (ci_low, ci_high),
        "wilcoxon_p": wilcoxon_p,
    }


def main():
    relevance_sets = load_json(RELEVANCE_FILE)

    # Ogni entry: (etichetta leggibile, path del file, chiave dentro il json se serve)
    # Adatta i path/chiavi alla struttura reale dei tuoi file se differisce.
    configs = {
        "TF-IDF (w=0.15, baseline)": (f"{RESULTS_DIR}/baseline_recommendations.json", None),
        "TF-IDF w=0 (solo testo)": (f"{RESULTS_DIR}/baseline_recommendations_pw0.json", None),
        "TF-IDF w=1 (solo periodo)": (f"{RESULTS_DIR}/baseline_recommendations_pw1.json", None),
        "CLIP a=0.5 w=0.15 (mista, default)": (f"{RESULTS_DIR}/clip_recommendations.json", "0.5"),
        "CLIP a=0 w=0.15 (solo testo)": (f"{RESULTS_DIR}/clip_recommendations.json", "0.0"),
        "CLIP a=1 w=0.15 (solo immagine)": (f"{RESULTS_DIR}/clip_recommendations.json", "1.0"),
        "CLIP a=0.5 w=0 (mista, no periodo)": (f"{RESULTS_DIR}/clip_recommendations_pw0.json", "0.5"),
        "CLIP a=0.5 w=1 (solo periodo, via CLIP)": (f"{RESULTS_DIR}/clip_recommendations_pw1.json", "0.5"),
        "CLIP a=0 w=0 (solo testo, no periodo)": (f"{RESULTS_DIR}/clip_recommendations_pw0_extremes.json", "0.0"),
        "CLIP a=1 w=0 (solo immagine, no periodo)": (f"{RESULTS_DIR}/clip_recommendations_pw0_extremes.json", "1.0"),
    }

    precisions = {}
    for label, (path, key) in configs.items():
        try:
            data = load_json(path)
        except FileNotFoundError:
            print(f"[SALTATO] {label}: file non trovato ({path})")
            continue
        recs_by_user = data[key] if key is not None else data
        precisions[label] = precision_per_user(recs_by_user, relevance_sets)

    # ------------------------------------------------------------------
    # Coppie di confronto rilevanti per il report (sez. 9.13, 9.15, 5.2)
    # ------------------------------------------------------------------
    pairs = [
        ("CLIP a=0 w=0.15 (solo testo)", "CLIP a=1 w=0.15 (solo immagine)"),
        ("CLIP a=0.5 w=0 (mista, no periodo)", "CLIP a=0.5 w=0.15 (mista, default)"),
        ("CLIP a=0.5 w=0.15 (mista, default)", "CLIP a=0.5 w=1 (solo periodo, via CLIP)"),
        ("CLIP a=0 w=0 (solo testo, no periodo)", "CLIP a=1 w=0 (solo immagine, no periodo)"),
        ("TF-IDF w=0 (solo testo)", "CLIP a=0.5 w=0 (mista, no periodo)"),
        ("TF-IDF (w=0.15, baseline)", "CLIP a=0.5 w=0.15 (mista, default)"),
    ]

    print("=" * 70)
    print(f"Test di significatività appaiati — k={K}, {N_USERS} utenti, seed={SEED}")
    print("=" * 70)

    results = {}
    for a, b in pairs:
        if a in precisions and b in precisions:
            results[f"{a} vs {b}"] = paired_test(a, precisions[a], b, precisions[b])
        else:
            print(f"\n[SALTATO] {a} vs {b}: dati mancanti per uno dei due")

    with open(f"{RESULTS_DIR}/significance_tests.json", "w") as f:
        json.dump(
            {k: {"mean_diff": v["mean_diff"], "ci_95": list(v["ci_95"]), "wilcoxon_p": v["wilcoxon_p"]}
             for k, v in results.items()},
            f, indent=2,
        )
    print(f"\nSalvato: {RESULTS_DIR}/significance_tests.json")


if __name__ == "__main__":
    main()
