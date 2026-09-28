<div align="center">

# 🏛️ Recommender System per il Cultural Heritage

**TF-IDF contro CLIP su dati MET Open Access**
*Confronto sperimentale con profili utente sintetici, test di significatività e spiegabilità*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-5.17-FFD21E)
![Dataset](https://img.shields.io/badge/Dataset-MET_Open_Access_(CC0)-b31b1b)
![Tests](https://img.shields.io/badge/unit_test-58_passed-2ea44f)

*Progetto per il corso di Sistemi Intelligenti per Internet · Università degli Studi Roma Tre*

</div>

---

## 📖 Di cosa si tratta

Come mostrare a ciascun visitatore di un museo le opere più affini ai suoi interessi? Il progetto confronta due recommender **content-based** su 2000 dipinti del Metropolitan Museum of Art (dipartimento *European Paintings*):

- 📋 **TF-IDF + similarità coseno** sui metadati testuali (nazionalità, classificazione, tag)
- 🧠 **CLIP** (`openai/clip-vit-base-patch32`) con embedding di testo, immagine o una loro combinazione

Il MET non fornisce interazioni utente, quindi la valutazione usa **50 profili sintetici** con ground truth costruito dai metadati (limite dichiarato esplicitamente). Metriche: Precision@k, Recall@k, NDCG@k.

## 🎯 Risultati principali

| Modello | P@10 |
|---|---:|
| Random | 0.049 |
| Popolarità (highlight) | 0.061 |
| Solo periodo *(controllo)* | 0.354 |
| CLIP + periodo (α = 0.5) | 0.302 |
| TF-IDF senza periodo | 0.432 |
| **TF-IDF + periodo** | **0.588** |

- ✅ **TF-IDF > CLIP**, con e senza il termine periodo (p < 0.0001, 10 seed su 10)
- ⚠️ **Il periodo è un confondente**: è condiviso col ground truth e da solo dà 0.354
- ❔ **Testo e immagine in CLIP non si distinguono** con questo campione

I risultati completi (ablation, test di significatività, spiegabilità) sono in `results/`.

## 🗂️ Struttura

```
cultural-heritage-recommender/
├── data/         # met_objects.jsonl, synthetic_users.json (clip_cache/ non versionata)
├── results/      # metriche, raccomandazioni, test di significatività, spiegazioni
├── src/          # codice: recommender, metriche, ablation, test statistici, spiegabilità
├── requirements.txt
└── requirements-lock.txt   # versioni esatte dei risultati
```

## 🚀 Avvio rapido

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/fetch_data.py --department_id 11 --max_items 2000 --output data/met_objects.jsonl
python src/generate_user_profiles.py --n_users 50 --stats
python src/recommender_baseline.py --top_k 10
python src/random_baseline.py --top_k 10 --n_runs 200 --seed 42
python src/embed_clip.py                       # richiede GPU CUDA
python src/recommender_clip.py --top_k 10
python src/significance_test.py
```

Test unitari: `pytest src/test_metrics.py src/test_recommender_clip_scoring.py src/test_random_baseline.py -v`.

## ⚠️ Limiti in breve

Ground truth e recommender condividono il termine *periodo*; profili sintetici e un solo seed principale; un solo dominio e un solo modello CLIP.

## 🛣️ Stato

- [x] Dati, profili sintetici, baseline, TF-IDF e CLIP
- [x] Ablation (periodo, α, k, 10 seed) e test di significatività
- [x] Modulo di spiegabilità
- [x] Analisi finale completata

---

<div align="center">

**Massimiliano Giangreco** · Prof. Giuseppe Sansonetti · Roma Tre

</div>
