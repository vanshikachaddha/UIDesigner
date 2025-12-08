from torchvision import transforms

from src.dataset.dataloader import get_dataloaders

print("TEST.PY IS RUNNING")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_loader, _ = get_dataloaders(
    root_dir="data",
    batch_size=4,
    transform=transform,
    max_seq_len=512,
)

batch = next(iter(train_loader))
print(batch["gui"].shape)
print(batch["token"].shape, batch["target"].shape)
