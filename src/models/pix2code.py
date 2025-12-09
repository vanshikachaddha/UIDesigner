#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pix2code_transformer.py

Top-level script wiring together:
- Tokenizer
- Dataset
- VisionTransformerEncoder + TransformerDecoder
- Greedy decoding (Transformer-style)
- CLI:
    * generate: image -> tokens -> HTML/DSL
    * evaluate: dataset JSON -> BLEU + token accuracy
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .encoder import VisionTransformerEncoder
from .decoder import TransformerDecoder  # your transformer decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Tokenizer
# -------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.sos_id = self.token_to_id.get("<SOS>", 1)
        self.eos_id = self.token_to_id.get("<EOS>", 2)

    def encode(self, text: str):
        tokens = text.strip().split()
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: List[int]):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)


# -------------------------------------------------------------------------
# Dataset (image + GT tokens)
# -------------------------------------------------------------------------

class Pix2CodeDataset(Dataset):
    """
    Expects JSON: [{"image": "...", "tokens": [...]}]
    """

    def __init__(self, json_path: str, vocab_path: str, img_size: int = 256):
        super().__init__()
        with open(json_path, "r") as f:
            self.samples = json.load(f)

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.pad_id = self.vocab.get("<PAD>", 0)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        img = self.transform(img)

        tokens = torch.tensor(sample["tokens"], dtype=torch.long)

        return img, tokens


# -------------------------------------------------------------------------
# Pix2CodeModel: ViT encoder + Transformer decoder
# -------------------------------------------------------------------------

class Pix2CodeModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.encoder = VisionTransformerEncoder(
            image_height=256,
            image_width=256,
            patch_size=16,
            embedding_dim=512,
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            dim=512,
            depth=6,
            heads=8,
            max_len=1024,
        )

    def forward(self, images, tgt_tokens):
        """
        Training forward pass.
        images: (B, 3, 256, 256)
        tgt_tokens: (B, T)
        """
        visual_embeds = self.encoder(images)              # (B, S, 512)
        logits = self.decoder(visual_embeds, tgt_tokens)  # (B, T, vocab)
        return logits


# -------------------------------------------------------------------------
# Greedy decoding (Transformer-style)
# -------------------------------------------------------------------------

def greedy_decode(model, img_tensor, tokenizer, max_len=120):
    """
    Decode by growing a prefix:
    prefix = [SOS], then keep appending next_token until EOS.
    """
    model.eval()
    img_tensor = img_tensor.to(DEVICE)

    sos = tokenizer.sos_id
    eos = tokenizer.eos_id

    decoded = [sos]

    with torch.no_grad():
        visual_embeds = model.encoder(img_tensor)   # (1, S, 512)

        for _ in range(max_len):
            prefix = torch.tensor([decoded], dtype=torch.long, device=DEVICE)
            logits = model.decoder(visual_embeds, prefix)
            next_id = logits[:, -1, :].argmax(dim=-1).item()

            if next_id == eos:
                break
            decoded.append(next_id)

    return decoded[1:]   # drop SOS


# -------------------------------------------------------------------------
# Single-image loader
# -------------------------------------------------------------------------

def load_single_image(path: str, img_size=256):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def evaluate(model, dataloader, tokenizer, max_len=120, max_batches=None):
    model.eval()
    total_acc = 0
    total_tokens = 0
    total_bleu = 0
    sample_count = 0

    smooth = SmoothingFunction()

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            B, T = targets.shape
            preds = []

            for b in range(B):
                img = images[b:b+1]
                pred_ids = greedy_decode(model, img, tokenizer, max_len)
                preds.append(pred_ids)

            pad = tokenizer.pad_id
            eos = tokenizer.eos_id

            for b in range(B):
                gt = [t for t in targets[b].tolist() if t not in (pad, eos)]
                pr = preds[b]

                L = min(len(gt), len(pr))
                if L > 0:
                    matches = sum(gt[k] == pr[k] for k in range(L))
                    total_acc += matches
                    total_tokens += L

                if gt and pr:
                    total_bleu += sentence_bleu(
                        [gt], pr,
                        smoothing_function=smooth.method1
                    )
                    sample_count += 1

    avg_acc = total_acc / total_tokens if total_tokens else 0.0
    avg_bleu = total_bleu / sample_count if sample_count else 0.0
    return avg_acc, avg_bleu


# -------------------------------------------------------------------------
# CLI: generate / evaluate
# -------------------------------------------------------------------------

def run_generate(args):
    tokenizer = Tokenizer(args.vocab)

    model = Pix2CodeModel(vocab_size=len(tokenizer.token_to_id))
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(DEVICE)

    img_tensor = load_single_image(args.image, img_size=256)
    token_ids = greedy_decode(model, img_tensor, tokenizer, max_len=args.max_len)
    html = tokenizer.decode(token_ids)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)

    print(f"[generate] Saved DSL/HTML to {out}")


def run_evaluate(args):
    tokenizer = Tokenizer(args.vocab)

    model = Pix2CodeModel(vocab_size=len(tokenizer.token_to_id))
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(DEVICE)

    dataset = Pix2CodeDataset(args.data_json, args.vocab, img_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    acc, bleu = evaluate(model, loader, tokenizer, max_len=args.max_len, max_batches=args.max_batches)

    print(f"[evaluate] Token accuracy: {acc:.4f}")
    print(f"[evaluate] BLEU: {bleu:.4f}")


# -------------------------------------------------------------------------
# main()
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("pix2code-transformer")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -------- generate --------
    gen = subparsers.add_parser("generate")
    gen.add_argument("--image", required=True)
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--vocab", default="data/vocab.json")
    gen.add_argument("--out", default="output.html")
    gen.add_argument("--max_len", type=int, default=120)

    # -------- evaluate --------
    ev = subparsers.add_parser("evaluate")
    ev.add_argument("--data_json", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--vocab", default="data/vocab.json")
    ev.add_argument("--batch_size", type=int, default=4)
    ev.add_argument("--max_len", type=int, default=120)
    ev.add_argument("--max_batches", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
