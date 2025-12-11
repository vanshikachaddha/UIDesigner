import sys, os

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.models.pix2code import Pix2CodeModel, Tokenizer
from dataset.dataloader import get_dataloaders
from torchvision import transforms
import torch


def evaluate_on_test(model, test_loader, criterion, tokenizer, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            tokens = batch[1].to(device)

            # Decoder input/output shift
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
    # Tokenizer + vocab size
    tokenizer = Tokenizer("data/vocab.json")
    vocab_size = len(tokenizer.token_to_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # SAME transforms as training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    # Load dataloaders
    _, _, test_loader = get_dataloaders(
        root_dir="data/",
        batch_size=4,
        transform=transform,
        max_seq_len=512
    )

    # ------------------------------
    # Load model + checkpoint
    # ------------------------------
    model = Pix2CodeModel(vocab_size=vocab_size).to(device)

    checkpoint_path = "checkpoints/final_model.pt"
    print("Loading checkpoint from:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load only weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # ------------------------------
    # RUN EVALUATION
    # ------------------------------
    test_loss = evaluate_on_test(model, test_loader, criterion, tokenizer, device)

    # Save result to file
    with open("test_results.txt", "a") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")

    print("Saved test results to test_results.txt")
