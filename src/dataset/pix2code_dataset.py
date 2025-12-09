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
        image_path = os.path.join(self.root_dir, "images", id + ".png")
        token_path = os.path.join(self.root_dir, "tokens", id + ".txt")

        image = Image.open(image_path)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        image_tensor = image
        
        with open(token_path, "r") as f:
            text = f.read()
        
        tokens = text.strip().split()
        id_list = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]

        while len(id_list) < self.max_seq_len:
            id_list.append(self.token_to_id["<PAD>"])
        
        if len(id_list) > self.max_seq_len:
            id_list = id_list[:self.max_seq_len]
        
        token_tensor = torch.tensor(id_list, dtype=torch.long)

        return (image_tensor, token_tensor)