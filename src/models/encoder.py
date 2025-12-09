#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder.py

Simple CNN encoder for pix2code-style model.
"""

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, mlp_dim=2048):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer (ViT)
    """
    def __init__(self, image_height = 256, image_width = 256, patch_size = 16, embedding_dim=512):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (self.image_height//self.patch_size) * (self.image_width // self.patch_size)

        patch_dim = 3 * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

        self.layers = nn.ModuleList([
                    TransformerEncoderBlock(
                        embedding_dim=embedding_dim,
                        num_heads=8,
                        mlp_dim=embedding_dim * 4 
                    )
                    for _ in range(6) 
                ])
        
        self.to_decoder_dim = nn.Identity()



    def forward(self, x):
        B, C, H, W = x.shape

        patches = x.unfold(2, self.patch_size, self.patch_size) \
                    .unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, self.num_patches, -1)

        patch_embeddings = self.patch_embed(patches)

        cls = self.cls_token.expand(B, 1, self.embedding_dim)

        tokens = torch.cat([cls, patch_embeddings], dim=1)

        tokens = tokens + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens)
        
        visual_embeds = self.to_decoder_dim(tokens)

        return visual_embeds

       



