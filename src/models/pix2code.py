#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pix2code.py

Main model combining CNN encoder and LSTM decoder.
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import CNNEncoder
from .decoder import LSTMDecoder


class Pix2Code(nn.Module):
    """
    Complete Pix2Code model:
    - CNN encoder processes images
    - LSTM decoder generates token sequences
    """
    
    def __init__(
        self,
        vocab_size: int,
        enc_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            enc_dim=enc_dim,
            num_layers=num_layers,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            images: (B, 3, H, W) - input images
            tgt_tokens: (B, T) - target token sequences (for teacher forcing)
        
        Returns:
            logits: (B, T, vocab_size) - predicted logits
        """
        # Encode images
        enc_features = self.encoder(images)  # (B, enc_dim)
        
        # Decode tokens
        logits = self.decoder(enc_features, tgt_tokens)  # (B, T, vocab_size)
        
        return logits
    
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 512,
        start_token: int = 1,  # <START>
        end_token: int = 2,    # <END>
        pad_token: int = 0,    # <PAD>
    ) -> torch.Tensor:
        """
        Generate token sequence from images (greedy decoding).
        
        Args:
            images: (B, 3, H, W) - input images
            max_len: maximum sequence length
            start_token: start token ID
            end_token: end token ID
            pad_token: padding token ID
        
        Returns:
            generated: (B, T) - generated token sequences
        """
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        enc_features = self.encoder(images)  # (B, enc_dim)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        hidden = None
        
        for _ in range(max_len - 1):
            # Get last token
            prev_token = generated[:, -1:]  # (B, 1)
            
            # Decode one step
            logits, hidden = self.decoder.decode_step(enc_features, prev_token, hidden)
            
            # Greedy: take argmax
            next_token = logits.argmax(dim=-1)  # (B, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have ended
            if (next_token == end_token).all():
                break
        
        return generated

