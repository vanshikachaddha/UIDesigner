import json
import os
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class Pix2CodeDataset(Dataset):
    """
    Returns a dict with:
        gui:    image tensor
        token:  input tokens (starts with <START>)
        target: target tokens (ends with <END>)
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None, max_seq_len: int = 512):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_seq_len = max_seq_len

        vocab_path = os.path.join(self.root_dir, "vocab.json")
        with open(vocab_path, "r") as f:
            self.token_to_id: Dict[str, int] = json.load(f)

        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.start_id = self.token_to_id.get("<START>", self.token_to_id.get("<SOS>", 1))
        self.end_id = self.token_to_id.get("<END>", self.token_to_id.get("<EOS>", 2))
        self.unk_id = self.token_to_id.get("<UNK>", 3)

        split_file = os.path.join(self.root_dir, f"{split}.json")
        try:
            with open(split_file, "r") as f:
                self.samples = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"split file not found: {split_file}") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_path, token_path = self._resolve_paths(sample)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = self._load_tokens(token_path)
        input_tokens, target_tokens = self._build_sequences(tokens)

        return {
            "gui": image,
            "token": input_tokens,
            "target": target_tokens,
        }

    def _resolve_paths(self, sample: Dict) -> Tuple[str, str]:
        """
        Samples may include absolute paths (from the JSON) or just an ID.
        We try the JSON paths first; if missing, fall back to data/images and data/tokens.
        """
        sample_id = sample.get("id")
        image_path = sample.get("image")
        token_path = sample.get("gui") or sample.get("tokens")

        if image_path and os.path.exists(image_path):
            resolved_image = image_path
        else:
            resolved_image = os.path.join(self.root_dir, "images", f"{sample_id}.png")

        if token_path and os.path.exists(token_path):
            resolved_tokens = token_path
        else:
            # accept either .txt or .gui extension
            txt_path = os.path.join(self.root_dir, "tokens", f"{sample_id}.txt")
            gui_path = os.path.join(self.root_dir, "tokens", f"{sample_id}.gui")
            if os.path.exists(txt_path):
                resolved_tokens = txt_path
            else:
                resolved_tokens = gui_path

        if not os.path.exists(resolved_image):
            raise FileNotFoundError(f"Image file not found for sample {sample_id}: {resolved_image}")
        if not os.path.exists(resolved_tokens):
            raise FileNotFoundError(f"Token file not found for sample {sample_id}: {resolved_tokens}")

        return resolved_image, resolved_tokens

    def _load_tokens(self, path: str):
        with open(path, "r") as f:
            text = f.read()
        return text.strip().split()

    def _build_sequences(self, tokens):
        # map to ids with UNK fallback
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        # teacher forcing setup
        input_ids = [self.start_id] + ids
        target_ids = ids + [self.end_id]

        # trim/pad to max_seq_len
        input_ids = input_ids[: self.max_seq_len]
        target_ids = target_ids[: self.max_seq_len]

        input_ids += [self.pad_id] * (self.max_seq_len - len(input_ids))
        target_ids += [self.pad_id] * (self.max_seq_len - len(target_ids))

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return input_tensor, target_tensor
