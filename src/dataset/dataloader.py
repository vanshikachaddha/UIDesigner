from .pix2code_dataset import Pix2CodeDataset
from torch.utils.data import DataLoader

def get_dataloaders(root_dir, batch_size, transform, max_seq_len):
    train_dataset = Pix2CodeDataset(root_dir, split="train",
                                    transform=transform, max_seq_len=max_seq_len)
    val_dataset = Pix2CodeDataset(root_dir, split="val",
                                  transform=transform, max_seq_len=max_seq_len)
    test_dataset = Pix2CodeDataset(root_dir, split="test",
                                   transform=transform, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

