#!/usr/bin/env python3
"""
Training entrypoint for the Pix2Code baseline (CNN encoder + LSTM decoder).
"""
import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from src.dataset.dataloader import get_dataloaders
from src.models.decoder import TransformerDecoder
from src.models.encoder import CNNEncoder


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Pix2Code(nn.Module):
    """
    CNN encoder + Transformer decoder with cross-attention.
    Encoder outputs (B, enc_dim); we treat it as a single visual token
    for the decoder's cross-attention.
    """

    def __init__(self, vocab_size, pad_idx, enc_dim=512, dec_dim=512, num_layers=6, heads=8, max_len=256):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=enc_dim)
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            dim=dec_dim,
            depth=num_layers,
            heads=heads,
            max_len=max_len,
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        enc_feat = self.encoder(images)  # (B, enc_dim)
        visual_embeds = self.enc_to_dec(enc_feat).unsqueeze(1)  # (B, 1, dec_dim)
        return self.decoder(visual_embeds, tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    pad_idx = vocab.get("<PAD>", 0)
    return vocab, pad_idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: Path, epoch, model, optimizer, scaler, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_clip, log_every, autocast_fn):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        images = batch["gui"].to(device)
        tokens = batch["token"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast_fn():
            logits = model(images, tokens)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

        if step % log_every == 0:
            avg = total_loss / step
            print(f"[train] step {step}/{len(train_loader)} | loss {avg:.4f}")

    return total_loss / max(1, len(train_loader))


def evaluate(model, val_loader, criterion, device, use_amp, autocast_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["gui"].to(device)
            tokens = batch["token"].to(device)
            targets = batch["target"].to(device)

            with autocast_fn():
                logits = model(images, tokens)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()

    return total_loss / max(1, len(val_loader))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pix2Code baseline.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--enc_dim", type=int, default=512)
    parser.add_argument("--dec_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer decoder layers")
    parser.add_argument("--heads", type=int, default=8, help="Transformer heads")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        autocast_fn = lambda: torch.cuda.amp.autocast(enabled=args.use_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    else:
        autocast_fn = contextlib.nullcontext
        scaler = None
        if args.use_amp:
            print("AMP is only available on CUDA; disabling use_amp.")
            args.use_amp = False

    vocab_path = data_dir / "vocab.json"
    vocab, pad_idx = load_vocab(vocab_path)
    vocab_size = len(vocab)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_loader, val_loader = get_dataloaders(
        root_dir=str(data_dir),
        batch_size=args.batch_size,
        transform=transform,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
    )

    model = Pix2Code(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        enc_dim=args.enc_dim,
        num_layers=args.num_layers,
        dec_dim=args.dec_dim,
        heads=args.heads,
        max_len=args.max_seq_len,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = math.inf

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.use_amp,
            args.grad_clip,
            args.log_every,
            autocast_fn,
        )
        val_loss = evaluate(model, val_loader, criterion, device, args.use_amp, autocast_fn)

        print(f"[epoch {epoch + 1}] train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

        save_checkpoint(save_dir / "latest.pt", epoch, model, optimizer, scaler, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_dir / "best.pt", epoch, model, optimizer, scaler, best_val_loss)
            print("Saved new best checkpoint.")


if __name__ == "__main__":
    main()
