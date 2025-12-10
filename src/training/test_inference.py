
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from torchvision import transforms

from models.pix2code import Pix2CodeModelNoCross, Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu

PROJECT_ROOT = "/content/UIDesigner"
DATA_ROOT    = f"{PROJECT_ROOT}/data"

VOCAB_PATH   = f"{DATA_ROOT}/vocab.json"
IMAGE_DIR    = f"{DATA_ROOT}/images"
TOKEN_DIR    = f"{DATA_ROOT}/tokens"

print("TEST (Teacher Forced) is running...")

tokenizer = Tokenizer(VOCAB_PATH)
vocab_size = len(tokenizer.token_to_id)

model = Pix2CodeModelNoCross(vocab_size=vocab_size)


CHECKPOINT = f"{PROJECT_ROOT}/saved_models/vit_transformer.pth"
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state.get("model", state))
model.eval()
print(f"✔ Loaded weights from {CHECKPOINT}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

TEST_IMAGE = f"{IMAGE_DIR}/001B5BD8-0401-4411-A3C4-A745050326C0.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

img = Image.open(TEST_IMAGE).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

img_id = os.path.basename(TEST_IMAGE).replace(".png", "")
gt_path = f"{TOKEN_DIR}/{img_id}.txt"

with open(gt_path, "r") as f:
    gt_tokens = f.read().strip().split()


gt_ids = [tokenizer.token_to_id[t] for t in gt_tokens]

start_id = tokenizer.token_to_id["<START>"]
dec_input_ids = [start_id] + gt_ids[:-1]

dec_in = torch.tensor([dec_input_ids], device=device)

logits = model(img, dec_in)     
pred_ids = logits.argmax(-1).squeeze().tolist()

pred_tokens = [tokenizer.id_to_token[i] for i in pred_ids]


correct = sum(p == g for p, g in zip(pred_tokens, gt_tokens))
token_acc = correct / max(len(gt_tokens), 1)

bleu = sentence_bleu([gt_tokens], pred_tokens)

edit_dist = nltk.edit_distance(gt_tokens, pred_tokens)

print("\n===== TEACHER-FORCED METRICS =====")
print(f"Token Accuracy: {token_acc:.4f}")
print(f"BLEU Score:     {bleu:.4f}")
print(f"Edit Distance:  {edit_dist}")
print("==================================\n")
