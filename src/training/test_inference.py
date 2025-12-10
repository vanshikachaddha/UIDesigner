# --------------------------------------------------------
# test_metrics.py — Run inference + compute test metrics
# --------------------------------------------------------

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from torchvision import transforms

from models.pix2code import Pix2CodeModel, Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu

# -------------------------------
# 1. Define all paths FIRST
# -------------------------------
PROJECT_ROOT = "/content/UIDesigner"
DATA_ROOT    = f"{PROJECT_ROOT}/data"

VOCAB_PATH   = f"{DATA_ROOT}/vocab.json"
IMAGE_DIR    = f"{DATA_ROOT}/images"
TOKEN_DIR    = f"{DATA_ROOT}/tokens"

print("TEST.PY is running...")

# -------------------------------
# 2. Load tokenizer
# -------------------------------
tokenizer = Tokenizer(VOCAB_PATH)
vocab_size = len(tokenizer.token_to_id)

# -------------------------------
# 3. Load model architecture
# -------------------------------
model = Pix2CodeModel(vocab_size=vocab_size)

# -------------------------------
# 4. Load trained weights
# -------------------------------
CHECKPOINT = f"{PROJECT_ROOT}/saved_models/baseline.pth"
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"✔ Loaded weights from {CHECKPOINT}")

# -------------------------------
# 5. Move model to device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# -------------------------------
# 6. Load + preprocess test image
# -------------------------------
TEST_IMAGE = f"{IMAGE_DIR}/001B5BD8-0401-4411-A3C4-A745050326C0.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

img = Image.open(TEST_IMAGE).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# -------------------------------
# 7. Encoder forward
# -------------------------------
enc_feat = model.encoder(img)

# -------------------------------
# 8. Greedy decode tokens
# -------------------------------
start_id = tokenizer.token_to_id["<START>"]
end_id   = tokenizer.token_to_id["<END>"]

prev = torch.tensor([[start_id]], device=device)
hidden = None
output_ids = []

for _ in range(512):
    logits, hidden = model.decoder.decode_step(enc_feat, prev, hidden)
    next_id = logits.argmax(-1).item()
    if next_id == end_id:
        break
    output_ids.append(next_id)
    prev = torch.tensor([[next_id]], device=device)

# Convert to DSL string
decoded = [tokenizer.id_to_token[t] for t in output_ids]
generated_dsl = " ".join(decoded)
print("\nMODEL OUTPUT DSL:")
print(generated_dsl)

# -------------------------------
# 9. Load ground truth DSL
# -------------------------------
img_id = os.path.basename(TEST_IMAGE).replace(".png", "")
gt_path = f"{TOKEN_DIR}/{img_id}.txt"

with open(gt_path, "r") as f:
    gt_tokens = f.read().strip().split()

pred_tokens = [tokenizer.id_to_token[i] for i in output_ids]

# -------------------------------
# 10. Compute Metrics
# -------------------------------
correct = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
token_acc = correct / max(len(gt_tokens), 1)
bleu = sentence_bleu([gt_tokens], pred_tokens)
edit_dist = nltk.edit_distance(gt_tokens, pred_tokens)

print("\nTEST METRICS:")
print(f"Token Accuracy: {token_acc:.4f}")
print(f"BLEU Score: {bleu:.4f}")
print(f"Edit Distance: {edit_dist}")
