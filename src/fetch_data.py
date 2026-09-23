"""
Raccolta dati dal Metropolitan Museum of Art Open Access API (nessuna API key richiesta,
licenza CC0). Scarica i metadati delle opere di uno o più dipartimenti e li salva in formato
JSONL, pronti per costruire i profili "contenuto" delle opere (testo + immagine) usati dal
recommender system.

Endpoint usati (verificati manualmente, base URL: collectionapi.metmuseum.org/public/collection/v1):
- GET /objects?departmentIds=<id>   -> lista di objectID per dipartimento
- GET /objects/<objectID>           -> metadati completi di una singola opera

Nota metodologica importante (da riportare nel report finale): l'API NON fornisce dati di
interazione utente (rating, click, visite) — solo metadati delle opere. Per costruire e
valutare il recommender servirà quindi generare profili utente sintetici (preferenze per
artista/cultura/periodo/classificazione), una scelta metodologica nota in letteratura per
questo dominio (cfr. lavori del NDS Lab, Università di Palermo, su recommender per il cultural
heritage) ma da dichiarare esplicitamente come limite.

Uso:
    python src/fetch_data.py --department_id 11 --max_items 2000 --output data/met_objects.jsonl
    python src/fetch_data.py --department_id 11 --department_id 6 --max_items 2000

Dipartimenti utili (id — nome, via GET /departments):
    1  — American Decorative Arts
    6  — Asian Art
    11 — European Paintings
    13 — Egyptian Art
    21 — Greek and Roman Art
(lista completa interrogabile a runtime con --list_departments)
"""

import argparse
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

# Campi rilevanti per il recommender: testuali (per la baseline content-based classica e per
# il testo da dare in input a CLIP) + immagine (per l'embedding visivo).
RELEVANT_FIELDS = [
    "objectID", "title", "artistDisplayName", "artistNationality", "department",
    "objectName", "classification", "culture", "period", "medium", "objectDate",
    "objectBeginDate", "objectEndDate", "tags", "primaryImage", "isPublicDomain",
    "isHighlight", "objectURL",
]


def list_departments():
    resp = requests.get(f"{BASE_URL}/departments", timeout=30)
    resp.raise_for_status()
    for dept in resp.json()["departments"]:
        print(f"{dept['departmentId']:>3} — {dept['displayName']}")


def get_object_ids(department_ids, has_images=True):
    """Recupera gli objectID di uno o più dipartimenti (filtrando per opere con immagine)."""
    all_ids = []
    for dept_id in department_ids:
        params = {"departmentIds": dept_id}
        resp = requests.get(f"{BASE_URL}/objects", params=params, timeout=30)
        resp.raise_for_status()
        ids = resp.json().get("objectIDs") or []
        print(f"Dipartimento {dept_id}: {len(ids)} oggetti totali")
        all_ids.extend(ids)
    return all_ids


def fetch_object(object_id, session, retries=3, backoff=1.0):
    """Scarica i metadati di una singola opera, con retry su errori transitori."""
    url = f"{BASE_URL}/objects/{object_id}"
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(backoff * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--department_id", type=int, action="append", required=False,
                         help="ID dipartimento MET (ripetibile, es. --department_id 11 "
                              "--department_id 6). Se omesso, usa --list_departments.")
    parser.add_argument("--list_departments", action="store_true",
                         help="Stampa la lista dei dipartimenti disponibili ed esce.")
    parser.add_argument("--max_items", type=int, default=2000,
                         help="Numero massimo di opere da scaricare (per non saturare l'API "
                              "né i tempi — l'API chiede di restare sotto 80 richieste/sec, "
                              "qui si sta molto più cauti).")
    parser.add_argument("--require_image", action="store_true", default=True,
                         help="Scarta le opere senza immagine primaria (default: attivo, "
                              "necessario per la parte CLIP).")
    parser.add_argument("--require_public_domain", action="store_true", default=True,
                         help="Scarta le opere non in pubblico dominio (default: attivo, "
                              "per uso sicuro delle immagini nel progetto/report).")
    parser.add_argument("--output", type=str, default="data/met_objects.jsonl")
    parser.add_argument("--request_delay", type=float, default=0.15,
                         help="Pausa (secondi) tra una richiesta e l'altra, per restare ben "
                              "sotto i limiti di rate dell'API.")
    args = parser.parse_args()

    if args.list_departments:
        list_departments()
        return

    if not args.department_id:
        parser.error("Specifica almeno un --department_id (oppure usa --list_departments).")

    object_ids = get_object_ids(args.department_id)
    print(f"Totale oggetti candidati: {len(object_ids)} (scarico al massimo {args.max_items})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    saved = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for object_id in object_ids:
            if saved >= args.max_items:
                break

            data = fetch_object(object_id, session)
            time.sleep(args.request_delay)

            if data is None:
                continue
            if args.require_image and not data.get("primaryImage"):
                continue
            if args.require_public_domain and not data.get("isPublicDomain"):
                continue

            record = {k: data.get(k) for k in RELEVANT_FIELDS}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

            if saved % 100 == 0:
                print(f"  {saved} opere salvate...")

    print(f"\nFatto: {saved} opere salvate in {output_path}")


if __name__ == "__main__":
    main()
