# 🖼️ Recommender System per il Cultural Heritage

> Confronto tra un recommender **content-based classico** (similarità su metadati/descrizioni)
> e un recommender **content-based con embedding CLIP** (immagine + testo), applicati al
> dominio del cultural heritage, con dataset open access del **Metropolitan Museum of Art**.

| | |
|---|---|
| 🖼️ **Dataset** | [MET Museum Open Access API](https://metmuseum.github.io/) (CC0) |
| 🖥️ **Hardware** | GPU dedicata, esecuzione locale |
| 🐧 **Sistema operativo** | Unix-based (Linux) |
| 🧩 **Approcci confrontati** | Content-based classico (metadati) · Content-based con CLIP |

---

## 📖 Descrizione

L'obiettivo è progettare, sviluppare e valutare un sistema di raccomandazione per il dominio
del cultural heritage, confrontando due paradigmi:

- 📋 **Content-based classico** — similarità calcolata su metadati/descrizioni testuali delle
  opere (es. TF-IDF, cosine similarity)
- 🧠 **Content-based con CLIP** — embedding congiunti immagine+testo per catturare similarità
  visive e semantiche che i soli metadati non colgono

Il progetto copre:

- 📊 **Valutazione rigorosa** — precision@k, recall@k, NDCG
- 🔍 **Spiegabilità delle raccomandazioni** — modulo dedicato a motivare ogni suggerimento
- 🧪 **Ablation study** — peso testo vs immagine, valore di k, design dei profili utente sintetici
- ⚖️ **Dichiarazione esplicita dei limiti** — assenza di dati di interazione utente reali nel
  dataset, compensata con profili utente sintetici (scelta nota in letteratura per questo
  dominio, non nascosta nel report)

Il progetto nasce nell'ambito del corso di **Sistemi Intelligenti per Internet**, come
evoluzione approvata di una proposta precedente.

---

## 🗂️ Struttura del progetto

```
cultural-heritage-recommender/
├── data/                # Dataset (scaricato, non versionato su Git)
├── src/                 # Codice sorgente
│   └── fetch_data.py    # Raccolta metadati dal MET Museum Open Access API
├── results/             # Metriche di valutazione, esempi di spiegabilità
├── configs/             # File di configurazione esperimenti (YAML)
├── .gitignore           # File e cartelle esclusi dal controllo versione
├── README.md            # Documentazione e stato di avanzamento del progetto
└── requirements.txt     # Dipendenze Python del progetto
```

---

## 📦 Dataset

**Metropolitan Museum of Art (MET) — Open Access API**
(`https://collectionapi.metmuseum.org/public/collection/v1`), nessuna API key richiesta,
licenza **CC0** su metadati e immagini in pubblico dominio (~470.000 opere totali).

L'API fornisce solo metadati delle opere (titolo, artista, cultura, periodo, classificazione,
tag, immagine, ecc.) — **nessun dato di interazione utente** (rating, click, visite). Per
allenare e valutare i recommender verranno quindi generati **profili utente sintetici**,
basati su preferenze per artista/cultura/periodo/classificazione — un limite metodologico
dichiarato esplicitamente, non nascosto.

Scope iniziale: **dipartimento 11 (European Paintings)**.

---

## ⚙️ Setup ambiente

```bash
# 1. Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Installa dipendenze
pip install -r requirements.txt
```

## 📥 Raccolta dati

```bash
# Esplora i dipartimenti disponibili sul MET
python src/fetch_data.py --list_departments

# Raccolta dati (dipartimento 11 = European Paintings)
python src/fetch_data.py --department_id 11 --max_items 2000 --output data/met_objects.jsonl
```

---

## 🗺️ Roadmap

- [x] Proposta di progetto approvata dal docente
- [x] Dataset individuato e verificato (MET Museum Open Access API)
- [x] `fetch_data.py` scritto
- [x] Repository GitHub creato e popolato con il setup iniziale
- [ ] Raccolta dati eseguita (`fetch_data.py` lanciato con successo)
- [ ] Generazione profili utente sintetici
- [ ] Recommender content-based classico (baseline)
- [ ] Recommender content-based con embedding CLIP
- [ ] Valutazione (precision@k, recall@k, NDCG)
- [ ] Modulo di spiegabilità delle raccomandazioni
- [ ] Ablation study (peso testo/immagine, valore di k, design profili sintetici)
- [ ] Report finale

---

## 📊 Metriche di valutazione

| Metrica | Perché |
|---|---|
| **Precision@k** | Frazione di raccomandazioni rilevanti tra le prime k |
| **Recall@k** | Frazione di item rilevanti effettivamente raccomandati tra i primi k |
| **NDCG** | Penalizza gli item rilevanti posizionati in basso nel ranking |

---

## 👥 Autore

Massimiliano Giangreco — Matricola 561883
Corso di Sistemi Intelligenti per Internet — Prof. Giuseppe Sansonetti
Università degli Studi Roma Tre — A.A. 2025/2026
