#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pix2code.py

Top-level script wiring together:
- Tokenizer
- Dataset
- CNNEncoder + LSTMDecoder into Pix2CodeModel
- Greedy decoding
- CLI:
    * generate: image -> tokens -> HTML/DSL
    * evaluate: dataset JSON -> token accuracy + BLEU

Assumptions:
- vocab.json: {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, ...}
- Dataset JSON: list of {"image": "path/to/image.png", "tokens": [int, ...]}
"""

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

from encoder import CNNEncoder
from decoder import LSTMDecoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Tokenizer
# -------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # required special tokens
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.sos_id = self.token_to_id.get("<SOS>", 1)
        self.eos_id = self.token_to_id.get("<EOS>", 2)

    def encode(self, text: str) -> List[int]:
        """
        Very simple example: assume DSL tokens are space-separated.
        Change this to match your actual tokenizer logic.
        """
        tokens = text.strip().split()
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Turn a list of token IDs into a string.
        """
        tokens = [self.id_to_token[i] for i in ids if i in self.id_to_token]
        return " ".join(tokens)


# -------------------------------------------------------------------------
# Dataset (for evaluation)
# -------------------------------------------------------------------------

class Pix2CodeDataset(Dataset):
    """
    Expects a JSON file with entries:
        {"image": "path/to/image.png", "tokens": [int, ...]}
    """

    def __init__(self, json_path: str, vocab_path: str, img_size: int = 224):
        super().__init__()
        self.json_path = json_path
        self.samples = self._load_json(json_path)

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.pad_id = self.vocab.get("<PAD>", 0)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # IMPORTANT: Use same normalization as training
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def _load_json(path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image"]
        tokens = sample["tokens"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        target = torch.tensor(tokens, dtype=torch.long)

        return img, target


# -------------------------------------------------------------------------
# Pix2CodeModel = CNNEncoder + LSTMDecoder
# -------------------------------------------------------------------------

class Pix2CodeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cnn_enc_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=cnn_enc_dim)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            enc_dim=cnn_enc_dim,
            num_layers=num_layers,
        )

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        For training with teacher forcing.
        images: (B, 3, H, W)
        tgt_tokens: (B, T)

        Returns:
            logits: (B, T, vocab_size)
        """
        enc_feat = self.encoder(images)           # (B, enc_dim)
        logits = self.decoder(enc_feat, tgt_tokens)
        return logits

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode_step(
        self,
        enc_feat: torch.Tensor,
        prev_token: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.decoder.decode_step(enc_feat, prev_token, hidden)


# -------------------------------------------------------------------------
# Greedy decoding
# -------------------------------------------------------------------------

def greedy_decode(
    model: Pix2CodeModel,
    img_tensor: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int = 120,
) -> List[int]:
    """
    Greedy decoding for a single image.
    img_tensor: (1, 3, H, W)
    Returns:
        token_ids (without <SOS> / <EOS>)
    """
    model.eval()
    img_tensor = img_tensor.to(DEVICE)

    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id

    decoded_ids = [sos_id]
    hidden = None

    with torch.no_grad():
        enc_feat = model.encode(img_tensor)  # (1, enc_dim)

        for _ in range(max_len):
            prev_token = torch.tensor(
                [[decoded_ids[-1]]],
                dtype=torch.long,
                device=DEVICE,
            )  # (1, 1)
            logits, hidden = model.decode_step(enc_feat, prev_token, hidden)
            # logits: (1, 1, vocab_size)
            next_id = logits.argmax(dim=-1).item()

            if next_id == eos_id:
                break
            decoded_ids.append(next_id)

    # drop SOS
    return decoded_ids[1:]


# -------------------------------------------------------------------------
# Image loader for inference
# -------------------------------------------------------------------------

def load_single_image(path: str, img_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------

def evaluate(
    model: Pix2CodeModel,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    max_len: int = 120,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Returns:
        avg_token_acc, avg_bleu
    """
    model.eval()
    total_token_acc = 0.0
    total_tokens = 0
    total_bleu = 0.0
    n_samples = 0

    chencherry = SmoothingFunction()

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(DEVICE)      # (B, C, H, W)
            targets = targets.to(DEVICE)    # (B, T)
            B, T = targets.shape

            preds: List[List[int]] = []
            for b in range(B):
                img = images[b:b+1]
                pred_ids = greedy_decode(model, img, tokenizer, max_len)
                preds.append(pred_ids)

            pad_id = tokenizer.pad_id
            eos_id = tokenizer.eos_id

            for b in range(B):
                gt = targets[b].tolist()
                # strip padding + EOS
                gt_clean = [t for t in gt if t not in (pad_id, eos_id)]
                pred_clean = preds[b]

                # token accuracy
                L = min(len(gt_clean), len(pred_clean))
                if L > 0:
                    match = sum(
                        1 for k in range(L) if gt_clean[k] == pred_clean[k]
                    )
                    total_token_acc += match
                    total_tokens += L

                # BLEU
                if len(gt_clean) > 0 and len(pred_clean) > 0:
                    reference = [gt_clean]
                    hypothesis = pred_clean
                    bleu = sentence_bleu(
                        reference,
                        hypothesis,
                        smoothing_function=chencherry.method1,
                    )
                    total_bleu += bleu
                    n_samples += 1

    avg_token_acc = total_token_acc / total_tokens if total_tokens else 0.0
    avg_bleu = total_bleu / n_samples if n_samples else 0.0
    return avg_token_acc, avg_bleu


# -------------------------------------------------------------------------
# CLI entry points: generate / evaluate
# -------------------------------------------------------------------------

def run_generate(args):
    tokenizer = Tokenizer(args.vocab)

    model = Pix2CodeModel(
        vocab_size=len(tokenizer.token_to_id),
        cnn_enc_dim=args.cnn_dim,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = ckpt.get("model", ckpt)  # support {"model": ...} or direct dict
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    img_tensor = load_single_image(args.image, img_size=args.img_size)
    token_ids = greedy_decode(model, img_tensor, tokenizer, max_len=args.max_len)
    html = tokenizer.decode(token_ids)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    print(f"[generate] Saved predicted HTML/DSL to {out_path}")


def run_evaluate(args):
    tokenizer = Tokenizer(args.vocab)

    model = Pix2CodeModel(
        vocab_size=len(tokenizer.token_to_id),
        cnn_enc_dim=args.cnn_dim,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    dataset = Pix2CodeDataset(args.data_json, args.vocab, img_size=args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    token_acc, bleu = evaluate(
        model,
        dataloader,
        tokenizer,
        max_len=args.max_len,
        max_batches=args.max_batches,
    )

    print(f"[evaluate] Data: {args.data_json}")
    print(f"[evaluate] Token accuracy: {token_acc:.4f}")
    print(f"[evaluate] BLEU:           {bleu:.4f}")


# -------------------------------------------------------------------------
# main()
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="pix2code-style model script")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----------------------------- generate -----------------------------
    gen_p = subparsers.add_parser("generate", help="Generate HTML/DSL from image")
    gen_p.add_argument("--image", type=str, required=True,
                       help="Path to input UI screenshot")
    gen_p.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pt)")
    gen_p.add_argument("--vocab", type=str, default="data/vocab.json",
                       help="Path to vocab.json")
    gen_p.add_argument("--out", type=str, default="output.html",
                       help="Output HTML/DSL file path")
    gen_p.add_argument("--max_len", type=int, default=120)
    gen_p.add_argument("--img_size", type=int, default=224)
    gen_p.add_argument("--cnn_dim", type=int, default=512)
    gen_p.add_argument("--emb_dim", type=int, default=256)
    gen_p.add_argument("--hidden_dim", type=int, default=512)
    gen_p.add_argument("--num_layers", type=int, default=1)

    # ----------------------------- evaluate -----------------------------
    eval_p = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    eval_p.add_argument("--data_json", type=str, required=True,
                        help="Path to dataset JSON (list of {image, tokens})")
    eval_p.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    eval_p.add_argument("--vocab", type=str, default="data/vocab.json",
                        help="Path to vocab.json")
    eval_p.add_argument("--batch_size", type=int, default=4)
    eval_p.add_argument("--num_workers", type=int, default=0)
    eval_p.add_argument("--max_len", type=int, default=120)
    eval_p.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches for quick testing")
    eval_p.add_argument("--img_size", type=int, default=224)
    eval_p.add_argument("--cnn_dim", type=int, default=512)
    eval_p.add_argument("--emb_dim", type=int, default=256)
    eval_p.add_argument("--hidden_dim", type=int, default=512)
    eval_p.add_argument("--num_layers", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
