import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.pix2code import Pix2CodeModel, Tokenizer
from dataset.dataloader import get_dataloaders
from torchvision import transforms


def evaluate_on_test(model, test_loader, criterion, tokenizer, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
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

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"TEST LOSS: {avg_test_loss:.4f}")
    return avg_test_loss

    
if __name__ == "__main__":
    tokenizer = Tokenizer("data/vocab.json")
    vocab_size = len(tokenizer.token_to_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAME transform as training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    # Load dataloaders (must return 3 loaders!)
    _, _, test_loader = get_dataloaders(
        root_dir="data/data/data",
        batch_size=4,
        transform=transform,
        max_seq_len=512
    )

    # Load model + checkpoint
    model = Pix2CodeModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load("saved_models/baseline.pth"))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Run test evaluation
    test_loss = evaluate_on_test(model, test_loader, criterion, tokenizer, device)

    # ---- WRITE TEST RESULT TO FILE ----
    with open("test_results.txt", "a") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")