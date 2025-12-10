import torch
import numpy as np
import gc
from collections import deque
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.pix2code import Pix2CodeModel, Tokenizer
from dataset.dataloader import get_dataloaders
from torchvision import transforms

# ---------------------------------------------------------
# 🔧 SETUP
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

tokenizer = Tokenizer("data/vocab.json")
vocab_size = len(tokenizer.token_to_id)

# Vision transforms (still ViT-style)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_loader, val_loader = get_dataloaders(
    root_dir="data/data/data/",
    batch_size=4,
    transform=transform,
    max_seq_len=512,
)

model = Pix2CodeModel(vocab_size=vocab_size).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

EPOCHS = 50
train_losses, val_losses = [], []

# ---------------------------------------------------------
# TRAINING STEP (ONE BATCH)
# ---------------------------------------------------------
def train_step(batch):
    images, tokens = batch[0].to(device), batch[1].to(device)
    dec_in, dec_out = tokens[:, :-1], tokens[:, 1:]

    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(images, dec_in)        # (B, T-1, vocab)
        B, Tm1, V = logits.shape

        loss = criterion(
            logits.reshape(B * Tm1, V),
            dec_out.reshape(B * Tm1),
        )

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


# ---------------------------------------------------------
# 🧊 VALIDATION STEP (ONE BATCH)
# ---------------------------------------------------------
@torch.no_grad()
def val_step(batch):
    images, tokens = batch[0].to(device), batch[1].to(device)
    dec_in, dec_out = tokens[:, :-1], tokens[:, 1:]

    logits = model(images, dec_in)
    B, Tm1, V = logits.shape

    loss = criterion(
        logits.reshape(B * Tm1, V),
        dec_out.reshape(B * Tm1),
    )
    return loss.item()


# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_train = 0

    for batch in train_loader:
        running_train += train_step(batch)

    avg_train_loss = running_train / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    running_val = 0

    for batch in val_loader:
        running_val += val_step(batch)

    avg_val_loss = running_val / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs("saved_models", exist_ok=True)
    save_path = "saved_models/baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
