from collections import deque
import gc
import torch
import numpy as np

mean_train_loss = deque(maxlen=100)
mean_val_loss = deque(maxlen=100)
total_train_loss = []
total_val_loss = []
print_every = 250
bleu_score = 0  # spelling fix + clarity

global_step = 0

for epoch in range(config.EPOCHS):
    # ---------- TRAIN ----------
    pix2code.train()
    epoch_train_loss = 0.0
    mean_train_loss.clear()   # keep deque focused on this epoch

    for batch_idx, batch in enumerate(train_loader, start=1):
        gui = batch["gui"].float().to(device)      # [B, ...]
        token = batch["token"].to(device)          # [B, ...]
        target = batch["target"].to(device)        # [B, ...]
        
        # reset optimizer
        optimizer.zero_grad()

        # forward
        pred = pix2code(gui, token)
        loss = criterion(pred, target)
