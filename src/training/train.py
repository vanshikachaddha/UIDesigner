#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Complete training script for Pix2Code model.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import deque

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from src.dataset.dataloader import get_dataloaders
from src.models.pix2code import Pix2Code


def load_vocab(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


def create_loss_fn(pad_token_id=0):
    """Create cross-entropy loss function that ignores padding tokens."""
    def loss_fn(pred_logits, target_ids):
        """
        pred_logits: (B, T, vocab_size)
        target_ids: (B, T)
        """
        # Flatten for loss calculation
        pred_flat = pred_logits.view(-1, pred_logits.size(-1))  # (B*T, vocab_size)
        target_flat = target_ids.view(-1)  # (B*T)
        
        # Create mask to ignore padding tokens
        mask = (target_flat != pad_token_id).float()
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            pred_flat, target_flat, reduction='none'
        )
        
        # Apply mask and average
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    return loss_fn


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    log_every=100,
    use_amp=False,
    grad_clip=1.0,
):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    loss_window = deque(maxlen=100)
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for batch_idx, batch in enumerate(train_loader, start=1):
        images, input_tokens, target_tokens = batch
        images = images.to(device)
        input_tokens = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images, input_tokens)
            loss = criterion(logits, target_tokens)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        loss_value = loss.item()
        epoch_loss += loss_value
        num_batches += 1
        loss_window.append(loss_value)
        
        # Logging
        if batch_idx % log_every == 0:
            avg_loss = np.mean(loss_window)
            print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss_value:.4f} | Avg Loss: {avg_loss:.4f}"
            )
    
    avg_epoch_loss = epoch_loss / max(1, num_batches)
    return avg_epoch_loss


def validate(
    model,
    val_loader,
    criterion,
    device,
):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, input_tokens, target_tokens = batch
            images = images.to(device)
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            
            logits = model(images, input_tokens)
            loss = criterion(logits, target_tokens)
            
            val_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = val_loss / max(1, num_batches)
    return avg_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Code model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N batches")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    pad_token_id = vocab.get("<PAD>", 0)
    print(f"Vocabulary size: {vocab_size}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        max_seq_len=args.max_seq_len,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = Pix2Code(
        vocab_size=vocab_size,
        enc_dim=512,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = create_loss_fn(pad_token_id=pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
        )
        
        # Validate
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Print epoch summary
        print(
            f"\nEpoch {epoch} Summary:\n"
            f"  Train Loss: {train_loss:.4f}\n"
            f"  Val Loss: {val_loss:.4f}\n"
        )
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, "latest.pt"))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})\n")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
