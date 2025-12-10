from collections import deque
import gc
import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.pix2code import Pix2CodeModelNoCross, Tokenizer
from dataset.dataloader import get_dataloaders

# --------- SETUP ---------

tokenizer = Tokenizer("data/vocab.json")
vocab_size = len(tokenizer.token_to_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_loader, val_loader, _ = get_dataloaders(
    root_dir="data/",
    batch_size=4,
    transform=transform,
    max_seq_len=512,
)

model = Pix2CodeModelNoCross(vocab_size=vocab_size).to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

EPOCHS = 50
train_losses = []
val_losses = []

# Create logs directory
os.makedirs("logs", exist_ok=True)
LOG_PATH = "logs/training_log.txt"

# Clear previous log file
with open(LOG_PATH, "w") as f:
    f.write("Epoch | Train Loss | Val Loss\n")
    f.write("-------------------------------------\n")

# --------- TRAINING LOOP ---------

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images = batch[0].to(device)
        tokens = batch[1].to(device)

        dec_in  = tokens[:, :-1]
        dec_out = tokens[:, 1:]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(images, dec_in)
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

    # ---- PRINT RESULTS ----
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- WRITE TO LOG FILE ----
    with open(LOG_PATH, "a") as f:
        f.write(f"{epoch+1:3d} | {avg_train_loss:.6f} | {avg_val_loss:.6f}\n")

    # optional cleanup
os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/vit_transformer.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")

print(f"\nTraining complete. Logs saved to: {LOG_PATH}")
