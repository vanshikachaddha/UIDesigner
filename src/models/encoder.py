import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Simple CNN encoder usable as a drop-in replacement for ViT-style encoder.
    Accepts unused args like embedding_dim, patch_size, image_height, image_width
    so that Pix2CodeModel can stay unchanged.
    """

    def __init__(
        self,
        out_dim=512,
        embedding_dim=None,   # <-- added
        patch_size=None,      # <-- added
        image_height=None,    # <-- added
        image_width=None,     # <-- added
        **kwargs              # <-- catches ANY extra args
    ):
        super().__init__()

        # Ignore all non-CNN parameters — CNN downsamples spatially anyway.

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
