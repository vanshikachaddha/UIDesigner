from torchvision import transforms
from src.dataset.dataloader import get_dataloaders

print("TEST.PY IS RUNNING")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_loader, val_loader = get_dataloaders(
    root_dir="data",
    batch_size=4,
    transform=transform,
    max_seq_len=512
)

images, tokens = next(iter(train_loader))
print(images.shape)
print(tokens.shape)