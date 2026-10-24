"""Embed every image in manifest.csv with CLIP and cache the result.

Usage:  python 01_embed.py
Writes: cache/embeddings.npy, cache/meta.csv,
        cache/text_embeddings.npy + cache/phrases.json (for cluster name suggestions)
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch32"
BATCH = 64

# Candidate names for clusters. Add your own after looking at the contact sheets.
PHRASES = [
    "two flat boxes stacked on top of each other",
    "several thin flat items lying in a pile",
    "a soft plastic bag next to a box",
    "a soft plastic bag tangled with other items",
    "a small item next to a large box",
    "an item pressed against the wall of a bin",
    "items packed tightly in a crowded bin",
    "a robot suction cup lifting two items at once",
    "a torn or opened cardboard box",
    "a book lying open with pages spread",
    "a crushed or deformed package",
    "a plastic bag that has ripped open",
    "a single item cleanly lifted",
]


def device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def img_feats(model, x):
    """Projected, normalised image embeddings; works across transformers versions."""
    pooled = model.vision_model(pixel_values=x["pixel_values"]).pooler_output
    e = model.visual_projection(pooled)
    return e / e.norm(dim=-1, keepdim=True)


def txt_feats(model, t):
    pooled = model.text_model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask")).pooler_output
    e = model.text_projection(pooled)
    return e / e.norm(dim=-1, keepdim=True)


def load(path):
    try:
        return Image.open(path).convert("RGB"), True
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128)), False


def main():
    os.makedirs("cache", exist_ok=True)
    df = pd.read_csv("manifest.csv")
    dev = device()
    print(f"{len(df)} images on {dev}")
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    embs, ok = [], []
    for i in range(0, len(df), BATCH):
        loaded = [load(p) for p in df.image.iloc[i : i + BATCH]]
        ims = [im for im, _ in loaded]
        ok += [good for _, good in loaded]
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt").to(dev)
            e = img_feats(model, x)
        embs.append(e.cpu().numpy())
        print(f"  {min(i + BATCH, len(df))}/{len(df)}", end="\r")

    E = np.concatenate(embs).astype("float32")
    df["ok"] = ok
    keep = df.ok.values
    print(f"\nSkipping {int((~keep).sum())} unreadable images")
    np.save("cache/embeddings.npy", E[keep])
    df[keep].drop(columns="ok").reset_index(drop=True).to_csv("cache/meta.csv", index=False)

    with torch.no_grad():
        t = proc(text=PHRASES, return_tensors="pt", padding=True).to(dev)
        T = txt_feats(model, t)
    np.save("cache/text_embeddings.npy", T.cpu().numpy().astype("float32"))
    json.dump(PHRASES, open("cache/phrases.json", "w"), indent=1)
    print("Saved cache/")


if __name__ == "__main__":
    main()
