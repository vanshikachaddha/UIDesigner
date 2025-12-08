from torch.utils.data import DataLoader

from .pix2code_dataset import Pix2CodeDataset


def get_dataloaders(root_dir, batch_size, transform, max_seq_len, num_workers=0):
    train_dataset = Pix2CodeDataset(root_dir, split="train", transform=transform, max_seq_len=max_seq_len)
    val_dataset = Pix2CodeDataset(root_dir, split="val", transform=transform, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
