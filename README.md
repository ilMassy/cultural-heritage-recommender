# 🖼️ Recommender System per il Cultural Heritage

> Confronto tra un recommender **content-based classico** (similarità su metadati/descrizioni)
> e un recommender **content-based con embedding CLIP** (immagine + testo), applicati al
> dominio del cultural heritage, con dataset open access del **Metropolitan Museum of Art**.

| | |
|---|---|
| 🖼️ **Dataset** | [MET Museum Open Access API](https://metmuseum.github.io/) (CC0) |
| 🖥️ **Hardware** | GPU dedicata (CUDA), esecuzione locale |
| 🐧 **Sistema operativo** | Unix-based (Linux) |
| 🧩 **Approcci confrontati** | Content-based classico (TF-IDF + coseno) · Content-based con CLIP |
| 🤖 **Modello CLIP** | `openai/clip-vit-base-patch32` (Hugging Face `transformers`) |

---

## 📖 Descrizione

L'obiettivo è progettare, sviluppare e valutare un sistema di raccomandazione per il dominio
del cultural heritage, confrontando due paradigmi:

- 📋 **Content-based classico** — similarità calcolata su metadati/descrizioni testuali delle
  opere (TF-IDF, cosine similarity)
- 🧠 **Content-based con CLIP** — embedding congiunti immagine+testo, per verificare se
  catturano similarità visive e semantiche che i soli metadati non colgono

Il progetto prevede:

- 📊 **Valutazione rigorosa** — precision@k, recall@k, NDCG, con baseline random e di popolarità
- 🔍 **Spiegabilità delle raccomandazioni** — modulo dedicato a motivare ogni suggerimento
- 🧪 **Ablation study** — peso testo vs immagine, peso del periodo, valore di k, design dei
  profili utente sintetici
- ⚖️ **Dichiarazione esplicita dei limiti** — assenza di dati di interazione utente reali nel
  dataset, compensata con profili utente sintetici (scelta nota in letteratura per questo
  dominio, non nascosta nel report)

Il progetto nasce nell'ambito del corso di **Sistemi Intelligenti per Internet**, come
evoluzione approvata di una proposta precedente. Lo stato di avanzamento dettagliato, con i
limiti metodologici e i prossimi passi, è in [`docs/REPORT_AVANZAMENTO.md`](docs/REPORT_AVANZAMENTO.md).

---

## 🗂️ Struttura del progetto

```
cultural-heritage-recommender/
├── data/
│   ├── met_objects.jsonl           # 2000 opere (European Paintings)
│   ├── synthetic_users.json        # 50 profili utente sintetici
│   └── clip_cache/                 # Embedding CLIP (rigenerabili, non versionati)
├── docs/
│   └── REPORT_AVANZAMENTO.md       # Report di avanzamento (stato, risultati, limiti)
├── results/                        # Metriche, raccomandazioni, esempi di spiegabilità
├── src/
│   ├── fetch_data.py               # Raccolta metadati dal MET Museum Open Access API
│   ├── generate_user_profiles.py   # Profili sintetici + relevance set (ground truth)
│   ├── recommender_baseline.py     # Baseline TF-IDF + coseno + metriche
│   ├── random_baseline.py          # Baseline random e di popolarità (highlight, tag)
│   ├── embed_clip.py               # Embedding CLIP testo+immagine (con cache)
│   ├── recommender_clip.py         # Recommender nello spazio CLIP
│   └── test_pipeline.py            # Test su mini-dataset sintetico
├── .gitignore                      # File e cartelle esclusi dal controllo versione
├── README.md                       # Documentazione e stato di avanzamento del progetto
├── requirements.txt                # Dipendenze Python del progetto
└── requirements-lock.txt           # Versioni esatte usate per i risultati (pip freeze)
```

Gli script in `src/` si importano a vicenda (es. `recommender_clip.py` usa
`embed_clip.py` e `random_baseline.py`), quindi devono restare nella stessa cartella.

---

## 📦 Dataset

**Metropolitan Museum of Art (MET) — Open Access API**
(`https://collectionapi.metmuseum.org/public/collection/v1`), nessuna API key richiesta,
licenza **CC0** su metadati e immagini in pubblico dominio (~470.000 opere totali).

L'API fornisce solo metadati delle opere (titolo, artista, cultura, periodo, classificazione,
tag, immagine, ecc.) — **nessun dato di interazione utente** (rating, click, visite). Per
allenare e valutare i recommender vengono quindi generati **profili utente sintetici**,
un limite metodologico dichiarato esplicitamente, non nascosto.

**Scope:** dipartimento 11 (**European Paintings**), 2000 opere scaricate (con `primaryImage`
e `isPublicDomain`). In questo dipartimento i campi `culture` e `period` sono vuoti al 100%:
i profili sono quindi basati su `artistNationality`, fascia temporale (da `objectBeginDate`),
`classification` e `tags`.

**Profili sintetici e ground truth:**

- 50 utenti, seed 42. Ogni profilo ha un nucleo di preferenze (1–2 nazionalità, una fascia
  temporale di ±50 anni, 1–2 classificazioni, 3–5 tag) con pesi perturbati per utente.
- Le opere rilevanti per ogni utente sono il **top 5% per affinità (100 opere)**, calcolata
  con un punteggio a pesi additivi più rumore controllato.
- Con 100 opere rilevanti e k=10, la **recall@10 ha un massimo teorico di 0.10**: nei
  risultati viene riportata anche normalizzata (R/0.10).

---

## ⚙️ Setup ambiente

```bash
# 1. Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Installa dipendenze
pip install -r requirements.txt          # dipendenze del progetto
# oppure, per riprodurre esattamente l'ambiente dei risultati:
pip install -r requirements-lock.txt
```

**Versioni e hardware.** I risultati sono stati ottenuti con le versioni in
`requirements-lock.txt` (generato con `pip freeze` nell'ambiente usato). La versione di
`transformers` conta: in quella installata `get_text_features` restituiva un oggetto invece
di un tensore, per questo `embed_clip.py` passa da `text_model`/`vision_model` più la
proiezione. Il calcolo degli embedding CLIP richiede una GPU CUDA per tempi ragionevoli
(su CPU funziona ma è molto più lento); TF-IDF, baseline e valutazione girano su CPU.

## 📥 Raccolta dati

```bash
# Esplora i dipartimenti disponibili sul MET
python src/fetch_data.py --list_departments

# Raccolta dati (dipartimento 11 = European Paintings)
python src/fetch_data.py --department_id 11 --max_items 2000 --output data/met_objects.jsonl
```

## ▶️ Esecuzione della pipeline

Comandi nell'ordine in cui sono stati eseguiti (i risultati riportati sotto valgono per
seed 42, 50 utenti, top 5%, k=10):

```bash
# Profili utente sintetici + ground truth
python src/generate_user_profiles.py --n_users 50 --stats

# Baseline: TF-IDF e random/popolarità
python src/recommender_baseline.py --top_k 10
python src/random_baseline.py --top_k 10 --n_runs 200 --seed 42

# CLIP: embedding (richiede GPU per tempi ragionevoli) e recommender
python src/embed_clip.py
python src/recommender_clip.py --top_k 10
```

**Analisi di sensibilità sul peso del periodo (TF-IDF).** Producono le righe "TF-IDF senza
boost periodo" (`--period_weight 0`) e "Solo periodo" (`--period_weight 1`) della tabella dei
risultati; con il default (0.15) si ottiene la baseline principale.

```bash
python src/recommender_baseline.py --top_k 10 --period_weight 0 \
    --output results/baseline_recommendations_pw0.json --metrics_output results/baseline_metrics_pw0.json
python src/recommender_baseline.py --top_k 10 --period_weight 1 \
    --output results/baseline_recommendations_pw1.json --metrics_output results/baseline_metrics_pw1.json
```

---

## 🗺️ Roadmap

- [x] Proposta di progetto approvata dal docente
- [x] Dataset individuato e verificato (MET Museum Open Access API)
- [x] `fetch_data.py` scritto
- [x] Repository GitHub creato
- [x] Raccolta dati eseguita (2000 opere, European Paintings)
- [x] Generazione profili utente sintetici (50 utenti) e ground truth
- [x] Recommender content-based classico (baseline TF-IDF)
- [x] Baseline random e di popolarità
- [x] Analisi di sensibilità sul peso del periodo per TF-IDF (0, 0.15, 1)
- [x] Recommender content-based con embedding CLIP (α = 0, 0.5, 1)
- [x] Valutazione a k=10 (precision@k, recall@k, NDCG) per tutti i modelli
- [ ] Ablation sul peso del periodo per CLIP
- [ ] Test di significatività appaiati per utente
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

## 📈 Risultati preliminari (k=10, 50 utenti, seed 42)

| Metodo | P@10 | R normalizzata (R/0.10) | NDCG@10 |
|---|---|---|---|
| Random (media su 200 run) | 0.049 | 0.05 | 0.050 |
| Popolarità (highlight) | 0.061 | 0.06 | 0.061 |
| Popolarità (tag) | 0.040 | 0.04 | 0.035 |
| **TF-IDF** (con boost periodo 0.15) | **0.588** | 0.59 | 0.613 |
| TF-IDF senza boost periodo (`--period_weight 0`) | 0.432 | 0.43 | 0.451 |
| Solo periodo (`--period_weight 1`, riferimento) | 0.354 | 0.35 | 0.343 |
| CLIP α=0 (solo testo) | 0.314 | 0.31 | 0.344 |
| CLIP α=0.5 (mista) | 0.302 | 0.30 | 0.315 |
| CLIP α=1 (solo immagine) | 0.252 | 0.25 | 0.260 |

Le configurazioni CLIP usano un boost del periodo di 0.15. La deviazione standard tra utenti
di P@10 è circa 0.21 per TF-IDF e 0.24 per CLIP.

**Come leggere questi numeri:**

- Tutti i modelli personalizzati sono molto sopra il random e le baseline di popolarità.
- Il termine "periodo" è identico nel ground truth e nei recommender, quindi 0.588 va sempre
  riportato insieme a 0.432 (senza periodo) e 0.354 (solo periodo).
- Nessuna configurazione CLIP supera il ranking "solo periodo". Non è ancora possibile dire
  cosa aggiunga CLIP oltre al periodo finché non è stata eseguita l'ablation con peso del
  periodo pari a 0.
- Con 50 utenti e deviazione standard ≈ 0.24, le differenze tra configurazioni CLIP non sono
  interpretabili senza test appaiati: al momento non si può concludere che il testo batta
  l'immagine.
- Il ground truth è costruito su metadati, quindi TF-IDF parte avvantaggiato rispetto a CLIP.

---

## ⚠️ Limiti noti

- **Termine "periodo" condiviso:** la funzione del periodo (`1 − |anno − centro| / 50`) è
  identica nel ground truth e nei recommender. Il solo periodo raggiunge P@10 = 0.354, quindi
  parte del risultato dei modelli dipende da un termine in comune con l'etichettatura.
- **Profili sintetici:** nessun dato di interazione reale; un solo seed di generazione.
- **Feature condivise:** ground truth e TF-IDF usano gli stessi attributi (nazionalità,
  classificazione, tag, periodo), quindi un accordo elevato è in parte atteso; il ground truth
  è basato su metadati e favorisce i metodi che li usano direttamente.
- **Nessuna misura di incertezza né test di significatività:** metriche medie su 50 utenti con
  deviazione standard ≈ 0.21–0.25; test appaiati e intervalli di confidenza sono in roadmap.
- **Diversità dei profili limitata:** 14 nazionalità su 34, 4 classificazioni su 6, 105 tag su
  527 usati dai profili; i gusti popolari sono favoriti dal campionamento per frequenza.
- **Scale diverse tra i modelli:** nel TF-IDF il coseno è grezzo, in CLIP le serie sono
  normalizzate min-max per utente, quindi lo stesso peso nominale del periodo ha un'influenza
  effettiva diversa.
- **Costruzione del profilo in CLIP:** il prompt utente è un elenco di tag eterogenei che un
  singolo embedding riduce a una media di concetti; questo può penalizzare CLIP per come è
  costruito il profilo e non per il modello.
- **Testo delle opere senza nazionalità** nell'input CLIP, mentre il prompt utente la contiene.
- **Scope ristretto:** un solo dipartimento (European Paintings) e un solo modello CLIP
  (`clip-vit-base-patch32`).

---

## 👥 Autore

Massimiliano Giangreco
