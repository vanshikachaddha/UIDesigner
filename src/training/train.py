from collections import deque
import gc
import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.pix2code import Pix2CodeModel, Tokenizer
from dataset.dataloader import get_dataloaders  # your existing helper

# --------- SETUP ---------

tokenizer = Tokenizer("data/vocab.json")
vocab_size = len(tokenizer.token_to_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# If your get_dataloaders requires transform, build it like ViT:
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

# Create dataloaders
train_loader, val_loader = get_dataloaders(
    root_dir="data/",
    batch_size=4,
    transform=transform,
    max_seq_len=512,
)

# Model, loss, optimizer, scaler
model = Pix2CodeModel(vocab_size=vocab_size).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

EPOCHS = 50

train_losses = []
val_losses = []

# --------- TRAINING LOOP ---------

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images = batch[0].to(device)        # (B, 3, 256, 256)
        tokens = batch[1].to(device)        # (B, T_full)

        # teacher forcing split
        dec_in  = tokens[:, :-1]            # input to decoder
        dec_out = tokens[:, 1:]             # prediction targets

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(images, dec_in)    # (B, T-1, vocab)
            B, Tm1, V = pred.shape

            loss = criterion(
                pred.reshape(B * Tm1, V),
                dec_out.reshape(B * Tm1)
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            tokens = batch[1].to(device)

            dec_in  = tokens[:, :-1]
            dec_out = tokens[:, 1:]

            pred = model(images, dec_in)
            B, Tm1, V = pred.shape

            loss = criterion(
                pred.reshape(B * Tm1, V),
                dec_out.reshape(B * Tm1)
            )
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.4f}")

    # optional: cleanup
    gc.collect()
    torch.cuda.empty_cache()
