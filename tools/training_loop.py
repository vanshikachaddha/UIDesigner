import gc
import numpy as np
import torch
from collections import deque

use_amp = torch.cuda.is_available() and getattr(config, "USE_AMP", True)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

train_window = deque(maxlen=100)
val_window = deque(maxlen=100)

all_train_losses = []
all_val_losses = []

log_every = 250
best_bleu = 0.0
global_step = 0

for epoch in range(config.EPOCHS):
    # -------------------- TRAIN --------------------
    pix2code.train()
    epoch_train_loss = 0.0
    num_train_batches = 0

    for batch_idx, batch in enumerate(train_loader, start=1):
        gui = batch["gui"].float().to(device, non_blocking=True)      # [B, ...]
        token = batch["token"].to(device, non_blocking=True)          # [B, T]
        target = batch["target"].to(device, non_blocking=True)        # [B, T]

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision for speed on GPU
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = pix2code(gui, token)
            loss = criterion(preds, target)

        # Backprop with amp
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(pix2code.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pix2code.parameters(), config.grad_clip)
            optimizer.step()

        loss_value = loss.detach().item()
        epoch_train_loss += loss_value
        num_train_batches += 1

        train_window.append(loss_value)
        all_train_losses.append(loss_value)

        # Step-level logging (smoothed)
        if batch_idx % log_every == 0:
            global_step += 1
            win_loss = float(np.mean(train_window))
            print(f"[Epoch {epoch} | Step {global_step}] Train Loss: {win_loss:.6f}")
            wandb.log(
                {
                    "step": global_step,
                    "train_loss_window": win_loss,
                },
                step=global_step,
            )

    avg_train_loss = epoch_train_loss / max(1, num_train_batches)

    # -------------------- VALIDATION --------------------
    pix2code.eval()
    epoch_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in valid_loader:
            gui = batch["gui"].float().to(device, non_blocking=True)
            token = batch["token"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            # no need for amp here unless your val is huge
            preds = pix2code(gui, token)
            val_loss = criterion(preds, target)

            val_value = val_loss.detach().item()
            epoch_val_loss += val_value
            num_val_batches += 1

            val_window.append(val_value)
            all_val_losses.append(val_value)

    avg_val_loss = epoch_val_loss / max(1, num_val_batches)
    train_win = float(np.mean(train_window))
    val_win = float(np.mean(val_window))

    print(
        f"[Epoch {epoch}] "
        f"Train Loss (epoch): {avg_train_loss:.6f} | "
        f"Val Loss (epoch): {avg_val_loss:.6f}"
    )

    wandb.log(
        {
            "epoch": epoch,
            "train_loss_epoch": avg_train_loss,
            "val_loss_epoch": avg_val_loss,
            "train_loss_window": train_win,
            "val_loss_window": val_win,
        },
        step=global_step,
    )

    # -------------------- BLEU EVAL --------------------
    if (epoch + 1) % 8 == 0:
        bleu, actual_seqs, pred_seqs = evaluate_model(
            pix2code,
            blue_html_seq,
            blue_gui_seq,
            MAX_LEN,
        )
        print(f"[Epoch {epoch}] BLEU: {bleu:.4f}")

        if bleu > best_bleu:
            best_bleu = bleu
            # optional: save best model
            # torch.save(pix2code.state_dict(), "pix2code_best.pt")

        wandb.log({"bleu": bleu, "best_bleu": best_bleu}, step=global_step)

    # Light mem cleanup if you really need it
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
