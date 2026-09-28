"""
metrics.py — Metriche di valutazione condivise (punto 11 della roadmap, sez. 10 del report).

Prima erano duplicate identiche in recommender_baseline.py (e probabilmente in
random_baseline.py — verificare) e usate anche da recommender_clip.py tramite
random_baseline.evaluate(). Spostate qui in un solo posto: chi importa da un file
diverso ora importa la stessa funzione, non una copia che potrebbe disallinearsi in
futuro (es. se si corregge un bug in una copia e ci si dimentica dell'altra).

Logica INVARIATA rispetto a recommender_baseline.py — solo spostata, non riscritta:
i numeri prodotti sono identici bit per bit a quelli già nel report.

Per usarla in recommender_baseline.py: rimuovere le definizioni locali di
precision_at_k/recall_at_k/ndcg_at_k e aggiungere in cima al file:
    from metrics import precision_at_k, recall_at_k, ndcg_at_k
Stessa cosa in random_baseline.py e in qualunque altro script che le ridefinisca.
"""

import math


def precision_at_k(recommended_ids, relevant_ids, k):
    """Frazione dei primi k raccomandati che sono effettivamente rilevanti."""
    top = recommended_ids[:k]
    if not top:
        return 0.0
    hits = len(set(top) & set(relevant_ids))
    return hits / len(top)


def recall_at_k(recommended_ids, relevant_ids, k):
    """Frazione degli item rilevanti che compaiono nei primi k raccomandati."""
    if not relevant_ids:
        return 0.0
    top = recommended_ids[:k]
    hits = len(set(top) & set(relevant_ids))
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    """NDCG binario (rilevanza 0/1) a k, normalizzato sull'ordinamento ideale."""
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
