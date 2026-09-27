# Esempi di spiegazione — recommender system MET

## Utente 0

Preferenze: nazionalità ['German'], classificazioni ['Paintings', 'Pastels & Oil Sketches on Paper'], tag ['Saint Peter', 'Flowers', 'Men', 'Venice'], periodo centrato su 1810 (±50).

**TF-IDF — utente 0, opera 437582** (Robert Shurlock (1772–1847))
- score finale = (1-w)·coseno + w·period_boost = 0.85·0.431 + 0.15·0.820 = **0.489**
- Contributo per categoria (somma = coseno totale, non lo score finale):
  - nat: +0.000
  - cls: +0.391
  - tag: +0.040
- Token che pesano di più:
  - `cls::Pastels & Oil Sketches on Paper` — contributo +0.3906 (peso utente 5.893, peso opera 5.893)
  - `tag::Men` — contributo +0.0403 (peso utente 1.893, peso opera 1.893)

---

## Utente 1

Preferenze: nazionalità ['Netherlandish', 'French'], classificazioni ['Paintings'], tag ['Joseph', 'Reading', 'Portraits', 'Night'], periodo centrato su 1775 (±50).

**TF-IDF — utente 1, opera 437736** (Christ among the Doctors)
- score finale = (1-w)·coseno + w·period_boost = 0.85·0.495 + 0.15·0.000 = **0.421**
- Contributo per categoria (somma = coseno totale, non lo score finale):
  - nat: +0.000
  - cls: +0.010
  - tag: +0.485
- Token che pesano di più:
  - `tag::Joseph` — contributo +0.2446 (peso utente 5.656, peso opera 5.656)
  - `tag::Reading` — contributo +0.2401 (peso utente 5.605, peso opera 5.605)
  - `cls::Paintings` — contributo +0.0102 (peso utente 1.158, peso opera 1.158)

---

## Utente 2

Preferenze: nazionalità ['Dutch'], classificazioni ['Paintings'], tag ['Satyrs', 'Portraits', 'Dogs', 'Winter'], periodo centrato su 1872 (±50).

**TF-IDF — utente 2, opera 437269** (January: Cernay, near Rambouillet)
- score finale = (1-w)·coseno + w·period_boost = 0.85·0.349 + 0.15·0.720 = **0.405**
- Contributo per categoria (somma = coseno totale, non lo score finale):
  - nat: +0.000
  - cls: +0.012
  - tag: +0.337
- Token che pesano di più:
  - `tag::Winter` — contributo +0.3367 (peso utente 6.036, peso opera 6.036)
  - `cls::Paintings` — contributo +0.0124 (peso utente 1.158, peso opera 1.158)
