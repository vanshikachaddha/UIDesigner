#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoder.py

Transformer decoder for pix2code-style model.
"""

from typing import Optional
import math

import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    """
    Standard 1D sinusoidal positional encoding for token sequences.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # shape: (1, max_len, d_model) for easy broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerDecoder(nn.Module):
    """
    Transformer-based autoregressive decoder.

    - Cross-attends to image features from ResNetEncoder (memory).
    - Uses token embeddings + 1D positional encoding.
    - Causal mask for autoregressive decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_tgt_len: int = 512,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size: size of the DSL vocabulary.
            d_model: transformer hidden size (must match encoder feature_dim).
            nhead: number of attention heads.
            num_layers: number of transformer decoder layers.
            dim_feedforward: FFN inner dimension.
            dropout: dropout probability.
            max_tgt_len: max target sequence length for positional encoding.
            pad_idx: padding index in the token vocabulary.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding for target sequence
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=max_tgt_len)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask for autoregressive decoding.

        Returns:
            mask: (T, T) with True in positions that should be masked.
        """
        # upper-triangular (excluding diagonal) is True (masked)
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            memory: encoder output, shape (B, S, d_model)
                    (S = num_patches from ResNetEncoder)
                    NOTE: memory can already include 2D positional encoding.
            tgt_tokens: target token ids, shape (B, T)
                        (assumed already shifted by training loop if needed)
            memory_key_padding_mask: optional, shape (B, S), True where memory is PAD.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        # Token embedding + positional encoding
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)  # (B, T, C)
        tgt_emb = self.pos_encoding(tgt_emb)                            # (B, T, C)

        # Causal mask so position i attends only to <= i
        tgt_mask = self._generate_square_subsequent_mask(T, device=device)  # (T, T)

        # Padding mask for target: True where PAD
        tgt_key_padding_mask = (
            tgt_tokens == self.pad_idx if self.pad_idx is not None else None
        )  # (B, T)

        # Run transformer decoder
        dec_out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T, d_model)

        logits = self.fc_out(dec_out)  # (B, T, vocab_size)
        return logits

    def decode_step(
        self,
        memory: torch.Tensor,
        prev_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single autoregressive step (no state cache).

        Args:
            memory: encoder output, (B, S, d_model)
            prev_tokens: full sequence generated so far, (B, T_so_far)
                         (including BOS, etc.)
            memory_key_padding_mask: optional, (B, S)

        Returns:
            logits_last: (B, 1, vocab_size) – logits for next token position
        """
        # Reuse full forward, take last time step
        logits = self.forward(
            memory=memory,
            tgt_tokens=prev_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T_so_far, vocab_size)

        logits_last = logits[:, -1:, :]  # (B, 1, vocab_size)
        return logits_last
