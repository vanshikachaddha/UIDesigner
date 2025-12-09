# src/training/test.py

from torchvision import transforms
import src.config as config
from src.dataset.dataloader import get_dataloaders

print("TEST.PY IS RUNNING")

# Use the same transform as in train.py
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

# Use the same data root / batch size / seq len as config
train_loader, val_loader = get_dataloaders(
    root_dir=config.DATA_ROOT,
    batch_size=config.BATCH_SIZE,
    transform=transform,
    max_seq_len=config.MAX_SEQ_LEN,
)

# Grab one batch and inspect shapes
images, tokens = next(iter(train_loader))
print("images shape:", images.shape)   # (B, 3, H, W)
print("tokens shape:", tokens.shape)   # (B, MAX_SEQ_LEN)
