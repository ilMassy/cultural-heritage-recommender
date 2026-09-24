"""
Generazione di profili utente sintetici per il recommender system sul dataset MET.

CONTESTO METODOLOGICO (da riportare nel report finale)
--------------------------------------------------------------------------------------
L'API del MET Museum non fornisce dati di interazione utente (rating, click, visite) —
solo metadati delle opere. Per allenare/valutare un recommender servono quindi profili
utente sintetici. Due estremi sono stati scartati:

  - Profili "netti" (hard filter, es. "ama solo arte francese 1800-1850"): facili da
    valutare ma poco realistici — un utente reale ha preferenze sfumate, non un filtro
    booleano.
  - Profili "sfumati puri" (pesi casuali su tutte le dimensioni senza una preferenza
    dominante): più realistici ma rendono impossibile definire un ground truth chiaro
    per il calcolo di precision@k/recall@k/NDCG.

Scelta adottata: ogni utente ha un NUCLEO di preferenze (nazionalità dell'artista,
fascia temporale, classificazione, tag), con pesi diversi per dimensione (i tag pesano
di più: dal QA del dataset raccolto risultano il segnale testuale più ricco, dato che
`classification` è fortemente sbilanciata — 85% "Paintings" — e `culture`/`period` sono
vuoti al 100% per questo dipartimento MET). Un termine di rumore controllato ammorbidisce
i confini netti del nucleo, così il punteggio di affinità utente-opera è continuo, non
binario. La "rilevanza" per la valutazione (precision@k/recall@k/NDCG) è definita come il
top 5% delle opere per punteggio di affinità — non un'assunzione arbitraria ma un criterio
esplicito, dichiarato e riproducibile (seed fisso).

NOTA METODOLOGICA IMPORTANTE (circolarità nella valutazione)
--------------------------------------------------------------------------------------
Il ground truth qui generato (`score_item`, formula pesata nationality/period/
classification/tags) NON deve essere riusato come algoritmo del recommender da valutare:
se il recommender ottimizzasse la stessa formula usata per etichettare la "rilevanza",
la valutazione sarebbe circolare (precision@k artificialmente vicina a 1, perché il
recommender starebbe semplicemente ripetendo la funzione che ha creato le risposte
giuste). Il ground truth va trattato come "verità di sfondo" indipendente dal metodo di
raccomandazione: il recommender baseline (vedi recommender_baseline.py) usa quindi un
approccio diverso (TF-IDF + cosine similarity classico), in modo che la valutazione sia
significativa. Da citare esplicitamente nel report come scelta di validità sperimentale.

Riferimento di dominio: la generazione di profili sintetici per il cultural heritage in
assenza di log di interazione è una scelta metodologica nota in letteratura per questo
dominio (cfr. lavori del NDS Lab, Università di Palermo) — qui operazionalizzata con un
criterio di rilevanza esplicito e riproducibile, non lasciata implicita.

Uso:
    python generate_user_profiles.py --n_users 50 --relevance_top_pct 0.05
    python generate_user_profiles.py --n_users 50 --stats   # stampa report diagnostico
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

# Pesi di base per dimensione (poi perturbati per singolo utente, vedi generate_profile).
# I tag pesano di più: segnale testuale più ricco per questo dataset (vedi docstring).
BASE_WEIGHTS = {
    "nationality": 0.25,
    "period": 0.20,
    "classification": 0.15,
    "tags": 0.40,
}

PERIOD_BANDWIDTH_YEARS = 50   # ampiezza della finestra temporale di interesse di un utente
NOISE_STD = 0.03              # rumore gaussiano per ammorbidire i confini del nucleo

# Fallback usato SOLO se il dataset in input non ha nessun anno noto (caso limite,
# non atteso su European Paintings dove il 76.3% delle opere ha objectDate/objectBeginDate).
FALLBACK_YEAR_RANGE = (1200, 1920)


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Nessun record trovato in {path} — file vuoto o malformato.")
    return records


def extract_nationalities(record):
    raw = record.get("artistNationality") or ""
    return [n.strip() for n in raw.split("|") if n.strip()]


def extract_tags(record):
    raw = record.get("tags") or []
    out = []
    for t in raw:
        if isinstance(t, dict):
            term = t.get("term")
            if term:
                out.append(term)
        elif isinstance(t, str):
            out.append(t)
    return out


def extract_year(record):
    """Anno di riferimento dell'opera, con fallback su objectBeginDate. None se assente
    (23.7% dei casi nel dipartimento European Paintings, vedi QA del dataset)."""
    year = record.get("objectBeginDate")
    if year in (None, 0):
        return None
    return year


def build_vocabularies(records):
    nat_counter = Counter()
    cls_counter = Counter()
    tag_counter = Counter()
    years = []
    for r in records:
        nat_counter.update(extract_nationalities(r))
        if r.get("classification"):
            cls_counter[r["classification"]] += 1
        tag_counter.update(extract_tags(r))
        y = extract_year(r)
        if y is not None:
            years.append(y)

    if not nat_counter:
        raise ValueError(
            "Nessuna nazionalità trovata in artistNationality su tutto il dataset — "
            "impossibile generare profili con questo campo. Verifica il fetch."
        )
    if not tag_counter:
        raise ValueError(
            "Nessun tag trovato — impossibile generare profili (i tag pesano il 40%)."
        )
    if not years:
        # Caso limite non atteso sul dataset reale (vedi QA), ma gestito esplicitamente
        # invece di far fallire rng.choice su una lista vuota.
        print(
            "ATTENZIONE: nessun anno noto nel dataset — uso un range di fallback "
            f"{FALLBACK_YEAR_RANGE} per il centro della fascia temporale dei profili. "
            "Da segnalare come limite nel report."
        )
        years = list(range(FALLBACK_YEAR_RANGE[0], FALLBACK_YEAR_RANGE[1] + 1))

    return nat_counter, cls_counter, tag_counter, years


def weighted_sample(counter, k, rng):
    """Campiona k chiavi da un Counter, con probabilità proporzionale alla frequenza
    (mimica il bias di popolarità reale dei gusti, evitando nicchie irrilevanti)."""
    items = list(counter.keys())
    weights = np.array([counter[i] for i in items], dtype=float)
    weights = weights / weights.sum()
    k = min(k, len(items))
    chosen = rng.choice(items, size=k, replace=False, p=weights)
    return list(chosen)


def generate_profile(user_id, nat_counter, cls_counter, tag_counter, years, rng):
    n_nat = rng.integers(1, 3)          # 1-2 nazionalità preferite
    n_cls = rng.integers(1, 3)          # 1-2 classificazioni preferite
    n_tags = rng.integers(3, 6)         # 3-5 tag preferiti

    preferred_nationalities = weighted_sample(nat_counter, n_nat, rng)
    preferred_classifications = weighted_sample(cls_counter, n_cls, rng)
    preferred_tags = weighted_sample(tag_counter, n_tags, rng)

    # Centro della fascia temporale d'interesse: campionato pesando per densità reale di
    # opere in quell'anno (evita utenti con periodi "vuoti" nel dataset).
    period_center = int(rng.choice(years))

    # Pesi per dimensione: base + piccola perturbazione, poi rinormalizzati a somma 1.
    weights = {}
    for dim, base_w in BASE_WEIGHTS.items():
        weights[dim] = max(0.02, base_w + rng.normal(0, 0.05))
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return {
        "user_id": user_id,
        "preferred_nationalities": preferred_nationalities,
        "preferred_classifications": preferred_classifications,
        "preferred_tags": preferred_tags,
        "period_center": period_center,
        "period_bandwidth": PERIOD_BANDWIDTH_YEARS,
        "weights": weights,
    }


def score_item(profile, record, rng=None):
    """Punteggio di affinità continuo in [0,1] tra un profilo utente e un'opera.

    NB: usato per generare il ground truth di rilevanza, NON va riusato come algoritmo
    del recommender da valutare (vedi nota di circolarità in cima al file)."""
    w = profile["weights"]

    item_nats = set(extract_nationalities(record))
    nat_score = 1.0 if item_nats & set(profile["preferred_nationalities"]) else 0.0

    year = extract_year(record)
    if year is None:
        period_score = 0.5  # neutro: non penalizzare/premiare opere senza data nota
    else:
        dist = abs(year - profile["period_center"])
        period_score = max(0.0, 1.0 - dist / profile["period_bandwidth"])

    cls_score = 1.0 if record.get("classification") in profile["preferred_classifications"] else 0.0

    item_tags = set(extract_tags(record))
    pref_tags = set(profile["preferred_tags"])
    tag_score = len(item_tags & pref_tags) / len(pref_tags) if pref_tags else 0.0

    score = (
        w["nationality"] * nat_score
        + w["period"] * period_score
        + w["classification"] * cls_score
        + w["tags"] * tag_score
    )

    noise = rng.normal(0, NOISE_STD) if rng is not None else 0.0
    return float(np.clip(score + noise, 0.0, 1.0))


def print_profile_stats(profiles):
    """Report diagnostico sulla diversità dei profili generati — utile per verificare
    che il campionamento pesato non collassi tutti gli utenti sulle stesse preferenze
    (nicchie sotto-rappresentate, mono-cultura del set di profili sintetici)."""
    n = len(profiles)
    nat_counter = Counter()
    cls_counter = Counter()
    tag_counter = Counter()
    for p in profiles:
        nat_counter.update(p["preferred_nationalities"])
        cls_counter.update(p["preferred_classifications"])
        tag_counter.update(p["preferred_tags"])

    avg_w = {
        dim: float(np.mean([p["weights"][dim] for p in profiles]))
        for dim in BASE_WEIGHTS
    }

    print("\n--- Report diagnostico profili ---")
    print(f"Profili generati: {n}")
    print(f"Nazionalità distinte usate: {len(nat_counter)} (top 5: {nat_counter.most_common(5)})")
    print(f"Classificazioni distinte usate: {len(cls_counter)} (top 5: {cls_counter.most_common(5)})")
    print(f"Tag distinti usati: {len(tag_counter)} (top 5: {tag_counter.most_common(5)})")
    print(f"Peso medio per dimensione: {avg_w}")
    print("-----------------------------------\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/met_objects.jsonl")
    parser.add_argument("--n_users", type=int, default=50)
    parser.add_argument("--relevance_top_pct", type=float, default=0.05,
                         help="Frazione delle opere con punteggio più alto considerate "
                              "'rilevanti' per un utente (ground truth di valutazione).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profiles_output", type=str, default="data/synthetic_users.json")
    parser.add_argument("--relevance_output", type=str, default="results/user_relevance_sets.json")
    parser.add_argument("--stats", action="store_true", help="Stampa report diagnostico dei profili generati.")
    args = parser.parse_args()

    if not 0 < args.relevance_top_pct < 1:
        raise ValueError("--relevance_top_pct deve essere in (0, 1).")
    if args.n_users < 1:
        raise ValueError("--n_users deve essere >= 1.")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    records = load_records(args.input)
    print(f"Opere caricate: {len(records)}")

    nat_counter, cls_counter, tag_counter, years = build_vocabularies(records)
    print(f"Nazionalità distinte: {len(nat_counter)} | Classificazioni: {len(cls_counter)} "
          f"| Tag distinti: {len(tag_counter)} | Opere con anno noto: {len(years)}")

    profiles = [
        generate_profile(uid, nat_counter, cls_counter, tag_counter, years, rng)
        for uid in range(args.n_users)
    ]

    if args.stats:
        print_profile_stats(profiles)

    relevance_sets = {}
    n_relevant = max(1, int(len(records) * args.relevance_top_pct))
    for profile in profiles:
        scored = [(r["objectID"], score_item(profile, r, rng)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        relevance_sets[str(profile["user_id"])] = [oid for oid, _ in scored[:n_relevant]]

    Path(args.profiles_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.relevance_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.profiles_output, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    with open(args.relevance_output, "w", encoding="utf-8") as f:
        json.dump(relevance_sets, f, ensure_ascii=False, indent=2)

    print(f"\n{len(profiles)} profili utente salvati in {args.profiles_output}")
    print(f"Relevance set ({n_relevant} opere/utente, top {args.relevance_top_pct*100:.0f}%) "
          f"salvati in {args.relevance_output}")


if __name__ == "__main__":
    main()
