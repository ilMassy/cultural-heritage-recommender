"""
embed_clip.py
Calcola gli embedding CLIP (immagine + testo) per le opere del dataset MET e li salva su disco.

- Modello di default: openai/clip-vit-base-patch32 (Hugging Face transformers)
- Riprendibile: ogni embedding e' salvato in data/clip_cache/{img,txt}/<objectID>.npy;
  se rilanci lo script, salta quelli gia' calcolati.
- Alla fine assembla:
    data/clip_cache/image_embeddings.npy   (N x D, L2-normalizzati)
    data/clip_cache/text_embeddings.npy    (N x D, L2-normalizzati)
    data/clip_cache/object_ids.json        (ordine delle righe)
  Le opere con immagine non scaricabile restano fuori da image_embeddings e sono
  elencate in failed_images.txt; il testo e' calcolato per tutte.

Uso:
    python src/embed_clip.py
    python src/embed_clip.py --size_variant web-large --batch_size 16
    python src/embed_clip.py --limit 20        # prova rapida
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_catalog(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def tag_names(tags):
    out = []
    for t in tags or []:
        if isinstance(t, dict):
            if t.get("term"):
                out.append(t["term"])
        elif isinstance(t, str):
            out.append(t)
    return out


def item_text(it):
    """Testo descrittivo dell'opera. CLIP tronca a 77 token: metto prima l'info piu' utile."""
    parts = [it.get("title") or ""]
    if it.get("artistDisplayName"):
        parts.append(f"by {it['artistDisplayName']}")
    if it.get("classification"):
        parts.append(it["classification"])
    tags = tag_names(it.get("tags"))
    if tags:
        parts.append(", ".join(tags))
    return ", ".join(p for p in parts if p)


def download_image(url, size_variant, retries=3, timeout=30):
    """Scarica un'immagine; se size_variant='web-large' prova prima l'URL ridotto."""
    candidates = []
    if size_variant and "/original/" in url:
        candidates.append(url.replace("/original/", f"/{size_variant}/"))
    candidates.append(url)

    for cand in candidates:
        for attempt in range(retries):
            try:
                r = requests.get(cand, timeout=timeout)
                if r.status_code == 200:
                    return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1))
    return None


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def encode_text(model, inputs):
    """Embedding testuale proiettato. Passa dai sotto-moduli invece di get_text_features,
    che in alcune versioni di transformers restituisce un oggetto anziche' un tensore."""
    out = model.text_model(**inputs)
    return model.text_projection(out.pooler_output)


def encode_image(model, inputs):
    """Embedding visivo proiettato (vedi encode_text)."""
    out = model.vision_model(pixel_values=inputs["pixel_values"])
    return model.visual_projection(out.pooler_output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/met_objects.jsonl")
    ap.add_argument("--out_dir", default="data/clip_cache")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--size_variant", default=None,
                    help="es. web-large: prova a scaricare una versione ridotta dell'immagine")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "txt").mkdir(parents=True, exist_ok=True)

    items = load_catalog(args.data)
    if args.limit:
        items = items[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modello: {args.model} | device: {device} | opere: {len(items)}")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    # ---------------------------------------------------------------- testo
    todo_txt = [it for it in items if not (out / "txt" / f"{it['objectID']}.npy").exists()]
    for i in range(0, len(todo_txt), 64):
        batch = todo_txt[i : i + 64]
        inputs = processor(text=[item_text(it) for it in batch],
                           return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = encode_text(model, inputs)
        emb = emb.cpu().numpy()
        for it, e in zip(batch, emb):
            np.save(out / "txt" / f"{it['objectID']}.npy", e)
        print(f"testo: {min(i + 64, len(todo_txt))}/{len(todo_txt)}")

    # ------------------------------------------------------------- immagini
    todo_img = [it for it in items
                if it.get("primaryImage")
                and not (out / "img" / f"{it['objectID']}.npy").exists()]
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i in range(0, len(todo_img), args.batch_size):
            batch = todo_img[i : i + args.batch_size]
            images = list(pool.map(
                lambda it: download_image(it["primaryImage"], args.size_variant), batch))
            ok = [(it, im) for it, im in zip(batch, images) if im is not None]
            failed += [it["objectID"] for it, im in zip(batch, images) if im is None]
            if ok:
                inputs = processor(images=[im for _, im in ok], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    emb = encode_image(model, inputs)
                emb = emb.cpu().numpy()
                for (it, _), e in zip(ok, emb):
                    np.save(out / "img" / f"{it['objectID']}.npy", e)
            print(f"immagini: {min(i + args.batch_size, len(todo_img))}/{len(todo_img)} "
                  f"(fallite finora: {len(failed)})")

    if failed:
        with open(out / "failed_images.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(map(str, failed)) + "\n")

    # ------------------------------------------------------------ assemblaggio
    ids = [it["objectID"] for it in items]
    txt = np.stack([np.load(out / "txt" / f"{oid}.npy") for oid in ids])
    np.save(out / "text_embeddings.npy", normalize(txt))
    with open(out / "object_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)

    img_ids = [oid for oid in ids if (out / "img" / f"{oid}.npy").exists()]
    if img_ids:
        img = np.stack([np.load(out / "img" / f"{oid}.npy") for oid in img_ids])
        np.save(out / "image_embeddings.npy", normalize(img))
        with open(out / "image_object_ids.json", "w", encoding="utf-8") as f:
            json.dump(img_ids, f)

    print(f"\nFatto. Testo: {len(ids)} embedding | Immagini: {len(img_ids)} "
          f"| Mancanti: {len(ids) - len(img_ids)}")
    print(f"Salvato in {out}/")


if __name__ == "__main__":
    main()
