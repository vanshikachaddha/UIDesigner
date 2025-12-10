#!/usr/bin/env python3
"""
Evaluate the no-cross-attention model on the held-out test split.
Assumes checkpoints saved via src/training/train.py (Pix2CodeModelNoCross).
"""
import os
import sys

import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pix2code import Pix2CodeModelNoCross, Tokenizer  # noqa: E402
from dataset.dataloader import get_dataloaders  # noqa: E402


def evaluate_on_test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, tokens in test_loader:
            images = images.to(device)
            tokens = tokens.to(device)

            dec_in = tokens[:, :-1]
            dec_out = tokens[:, 1:]

            pred = model(images, dec_in)
            B, Tm1, V = pred.shape

            loss = criterion(
                pred.reshape(B * Tm1, V),
                dec_out.reshape(B * Tm1)
            )

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"TEST LOSS: {avg_test_loss:.4f}")
    return avg_test_loss


if __name__ == "__main__":
    tokenizer = Tokenizer("data/vocab.json")
    vocab_size = len(tokenizer.token_to_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # expect test loader from get_dataloaders
    _, _, test_loader = get_dataloaders(
        root_dir="data/",
        batch_size=4,
        transform=transform,
        max_seq_len=512,
    )

    model = Pix2CodeModelNoCross(vocab_size=vocab_size).to(device)
    state = torch.load("saved_models/vit_transformer.pth", map_location=device)
    model.load_state_dict(state.get("model", state))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    test_loss = evaluate_on_test(model, test_loader, criterion, device)

    with open("test_results.txt", "a") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
