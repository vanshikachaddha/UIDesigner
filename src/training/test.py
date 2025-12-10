# --------------------------------------------------------
# test.py — Run inference using your saved Pix2Code model
# --------------------------------------------------------

import torch
from PIL import Image
from torchvision import transforms

from src.models.pix2code import Pix2CodeModel, Tokenizer

VOCAB_PATH = "/content/UIDesigner/data/data/data/vocab.json"
DATA_ROOT  = "/content/UIDesigner/data/data/data/"
tokenizer = Tokenizer(VOCAB_PATH)
vocab_size = len(tokenizer.token_to_id)


print("TEST.PY is running...")

# -------------------------------
# 1. Load tokenizer + vocab
# -------------------------------
VOCAB_PATH = "/content/UIDesigner/data/data/data/vocab.json"
tokenizer = Tokenizer(VOCAB_PATH)
vocab_size = len(tokenizer.token_to_id)

# -------------------------------
# 2. Recreate model architecture
# -------------------------------
model = Pix2CodeModel(vocab_size=vocab_size)

# -------------------------------
# 3. Load trained weights
# -------------------------------
CHECKPOINT = "saved_models/baseline.pth"
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"✔ Loaded weights from {CHECKPOINT}")

# -------------------------------
# 4. Move model to device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# -------------------------------
# 5. Load + preprocess test image
# -------------------------------
TEST_IMAGE = "/content/UIDesigner/data/data/data/images/001B5BD8-0401-4411-A3C4-A745050326C0.png"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

img = Image.open(TEST_IMAGE).convert("RGB")
img = transform(img).unsqueeze(0).to(device)   # (1,3,256,256)

# -------------------------------
# 6. Encode image
# -------------------------------
enc_feat = model.encoder(img)

# -------------------------------
# 7. Greedy decode tokens
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

# -------------------------------
# 8. Convert IDs → DSL text
# -------------------------------
decoded = [tokenizer.id_to_token[t] for t in output_ids]
generated_dsl = " ".join(decoded)

print("\nMODEL OUTPUT DSL:")
print(generated_dsl)
print("\nInference complete!")

import os
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Extract ID of test image
img_id = os.path.basename(TEST_IMAGE).replace(".png", "")

# Load ground truth DSL
gt_path = os.path.join(DATA_ROOT, "tokens", img_id + ".txt")
with open(gt_path, "r") as f:
    gt_tokens = f.read().strip().split()

# Predicted tokens
pred_tokens = [tokenizer.id_to_token[i] for i in output_ids]

# ----- METRICS -----
# 1. Token Accuracy
correct = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
token_acc = correct / max(len(gt_tokens), 1)

# 2. BLEU Score
bleu = sentence_bleu([gt_tokens], pred_tokens)

# 3. Edit Distance
edit_dist = nltk.edit_distance(gt_tokens, pred_tokens)

print("\nTEST METRICS:")
print(f"Token Accuracy: {token_acc:.4f}")
print(f"BLEU Score: {bleu:.4f}")
print(f"Edit Distance: {edit_dist}")

