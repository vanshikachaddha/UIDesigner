import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class Pix2CodeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, max_seq_len=512):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        vocab_path = os.path.join(self.root_dir, "vocab.json")
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        
        self.token_to_id = self.vocab
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}

        split_file = os.path.join(self.root_dir, split + ".json")
        try:
            with open(split_file, "r") as f:
                self.samples = json.load(f)
                """
                sample["image"]  → full image path
                sample["gui"]    → full DSL path
                sample["id"]     → ID string
                """

        except FileNotFoundError:
            print(f"Error: '{split}.json' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{split}.json'.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        id = sample["id"]
        
        # Try to use paths from JSON first, fall back to constructed paths
        if "image" in sample and os.path.exists(sample["image"]):
            image_path = sample["image"]
        else:
            # Try relative path construction
            image_path = os.path.join(self.root_dir, "images", id + ".png")
            if not os.path.exists(image_path):
                # Try alternative locations
                alt_paths = [
                    os.path.join(self.root_dir, "processed", "images", id + ".png"),
                    os.path.join(self.root_dir, id + ".png"),
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
        
        # Handle token/gui file - try multiple extensions and locations
        token_path = None
        if "gui" in sample:
            # Try the path from JSON
            gui_path = sample["gui"]
            # Replace .gui with .txt if needed, or try both
            txt_path = gui_path.replace(".gui", ".txt")
            if os.path.exists(gui_path):
                token_path = gui_path
            elif os.path.exists(txt_path):
                token_path = txt_path
        
        # Fall back to constructed paths
        if token_path is None or not os.path.exists(token_path):
            # Try tokens directory with .txt
            token_path = os.path.join(self.root_dir, "tokens", id + ".txt")
            if not os.path.exists(token_path):
                # Try dsl directory with .gui
                token_path = os.path.join(self.root_dir, "dsl", id + ".gui")
                if not os.path.exists(token_path):
                    # Try processed/dsl
                    token_path = os.path.join(self.root_dir, "processed", "dsl", id + ".gui")
                    if not os.path.exists(token_path):
                        # Last resort: try in root
                        token_path = os.path.join(self.root_dir, id + ".txt")

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        image_tensor = image
        
        # Load tokens
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found: {token_path}")
        with open(token_path, "r") as f:
            text = f.read()
        
        tokens = text.strip().split()
        id_list = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        
        # Add <START> at the beginning and <END> at the end
        start_id = self.token_to_id.get("<START>", 1)
        end_id = self.token_to_id.get("<END>", 2)
        pad_id = self.token_to_id.get("<PAD>", 0)
        
        # Create input sequence: [<START>, token1, token2, ..., tokenN-1]
        # Create target sequence: [token1, token2, ..., tokenN, <END>]
        input_ids = [start_id] + id_list[:-1] if len(id_list) > 0 else [start_id]
        target_ids = id_list + [end_id] if len(id_list) > 0 else [end_id]
        
        # Pad or truncate to max_seq_len
        if len(input_ids) < self.max_seq_len:
            input_ids = input_ids + [pad_id] * (self.max_seq_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_len]
        
        if len(target_ids) < self.max_seq_len:
            target_ids = target_ids + [pad_id] * (self.max_seq_len - len(target_ids))
        else:
            target_ids = target_ids[:self.max_seq_len]
        
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return (image_tensor, input_tensor, target_tensor)
