from collections import deque
import gc
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

import src.config as config
from src.models.pix2code import Pix2CodeModel, Tokenizer
from src.dataset.dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image transform, same as in inference/eval
transform = T.Compose([
    T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])

# build dataloaders
train_loader, val_loader = get_dataloaders(
    root_dir=config.DATA_ROOT,
    batch_size=config.BATCH_SIZE,
    transform=transform,
    max_seq_len=config.MAX_SEQ_LEN,
)

# tokenizer + model + optimizer + loss
tokenizer = Tokenizer(vocab_path=f"{config.DATA_ROOT}/vocab.json")
pad_id = tokenizer.pad_id

model = Pix2CodeModel(
    vocab_size=len(tokenizer.token_to_id),
    d_model=config.D_MODEL,
    nhead=config.NHEAD,
    num_decoder_layers=config.DEC_LAYERS,
    dim_feedforward=config.FF_DIM,
    dropout=config.DROPOUT,
    resnet_version=config.RESNET_VERSION,
    pretrained=not config.NO_PRETRAINED,
    freeze_backbone=config.FREEZE_BACKBONE,
    pad_idx=pad_id,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

mean_train_loss = deque(maxlen=100)
mean_val_loss = deque(maxlen=100)

# ---- NEW: loss history containers (per epoch) ----
train_loss_history = []   # list of avg train loss per epoch
val_loss_history = []     # list of avg val loss per epoch

for epoch in range(config.EPOCHS):
    # ---------- TRAIN ----------
    model.train()
    epoch_train_loss = 0.0
    mean_train_loss.clear()

    for batch_idx, (gui, tokens) in enumerate(train_loader, start=1):
        gui    = gui.float().to(device)      # (B, 3, H, W)
        tokens = tokens.to(device)           # (B, L)

        token_in = tokens[:, :-1]            # (B, L-1)
        target   = tokens[:, 1:]             # (B, L-1)

        optimizer.zero_grad()
        logits = model(gui, token_in)        # (B, L-1, V)
        B, T, V = logits.shape

        loss = criterion(
            logits.view(B * T, V),
            target.reshape(B * T),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        optimizer.step()

        epoch_train_loss += loss.item()
        mean_train_loss.append(loss.item())

        del gui, tokens, token_in, target, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    avg_train = epoch_train_loss / len(train_loader)
    train_loss_history.append(avg_train)         # <--- log train loss

    # ---------- VALIDATION ----------
    model.eval()
    epoch_val_loss = 0.0
    mean_val_loss.clear()

    with torch.no_grad():
        for batch_idx, (gui, tokens) in enumerate(val_loader, start=1):
            gui    = gui.float().to(device)
            tokens = tokens.to(device)

            token_in = tokens[:, :-1]
            target   = tokens[:, 1:]

            logits = model(gui, token_in)
            B, T, V = logits.shape

            loss = criterion(
                logits.view(B * T, V),
                target.reshape(B * T),
            )

            epoch_val_loss += loss.item()
            mean_val_loss.append(loss.item())

            del gui, tokens, token_in, target, logits, loss
            gc.collect()
            torch.cuda.empty_cache()

    avg_val = epoch_val_loss / len(val_loader)
    val_loss_history.append(avg_val)            # <--- log val loss

    print(
        f"Epoch {epoch+1}/{config.EPOCHS} | "
        f"train: {avg_train:.4f} | val: {avg_val:.4f}"
    )

# ---- NEW: save histories to disk at the very end ----
os.makedirs("logs", exist_ok=True)
np.savez(
    "logs/loss_history.npz",
    train=np.array(train_loss_history),
    val=np.array(val_loss_history),
)
print("Saved loss history to logs/loss_history.npz")
