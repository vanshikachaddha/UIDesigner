#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoder.py

Transformer-style decoder with explicit CrossAttention
for pix2code-style model.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import CrossAttention


class TransformerDecoderBlock(nn.Module):
    """
    One decoder block:
      1) masked self-attention over tokens
      2) cross-attention over visual (encoder) features
      3) feed-forward network
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
        )
        self.cross_attn = CrossAttention(dim=dim, heads=heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        visual_embeds: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: (B, T, C)
        visual_embeds: (B, S, C)
        self_attn_mask: (T, T) causal mask
        self_key_padding_mask: (B, T) True where PAD in tokens
        memory_key_padding_mask: (B, S) True where PAD in encoder features
        """
        # 1) Masked self-attention
        x = self.norm1(tokens)
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
        )
        tokens = tokens + self.dropout(x)

        # 2) Cross-attention to encoder features
        x = self.norm2(tokens)
        # CrossAttention is assumed to handle (B, T, C) + (B, S, C)
        # and optionally memory_key_padding_mask internally if you add it.
        context = self.cross_attn(x, visual_embeds)
        tokens = tokens + self.dropout(context)

        # 3) Feed-forward network
        x = self.norm3(tokens)
        x = self.mlp(x)
        tokens = tokens + self.dropout(x)

        return tokens


class TransformerDecoder(nn.Module):
    """
    Autoregressive decoder that:
      - embeds tokens
      - adds learned positional embeddings
      - applies a stack of TransformerDecoderBlock
      - projects to vocabulary logits

    Exposes the same interface as the previous implementation:
      forward(memory, tgt_tokens, memory_key_padding_mask=None) -> (B, T, V)
      decode_step(memory, prev_tokens, ...) -> (B, 1, V)
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
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_tgt_len = max_tgt_len

        # Token + positional embeddings (learned)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_tgt_len, d_model)

        # Decoder blocks
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    dim=d_model,
                    heads=nhead,
                    mlp_ratio=dim_feedforward / d_model,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to vocab
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _generate_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Returns an additive causal mask of shape (T, T) with -inf above the diagonal
        and 0 on/below the diagonal, suitable for MultiheadAttention attn_mask.
        """
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def _embed_tokens(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: (B, T)
        returns: (B, T, d_model)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        pos = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)
        x = self.token_emb(tgt_tokens) * math.sqrt(self.d_model)
        x = x + self.pos_emb(pos)  # (B, T, d_model)
        return x

    def forward(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        memory: (B, S, d_model) encoder output (ResNetEncoder + PosEnc2D)
        tgt_tokens: (B, T) token ids (already shifted by training loop)
        memory_key_padding_mask: (B, S), True where encoder tokens are PAD

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        x = self._embed_tokens(tgt_tokens)  # (B, T, d_model)

        # Causal mask for self-attention
        self_attn_mask = self._generate_causal_mask(T, device=device)  # (T, T)

        # Padding mask for tokens
        tgt_key_padding_mask = (
            tgt_tokens == self.pad_idx if self.pad_idx is not None else None
        )  # (B, T)

        for layer in self.layers:
            x = layer(
                tokens=x,
                visual_embeds=memory,
                self_attn_mask=self_attn_mask,
                self_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        logits = self.fc_out(x)  # (B, T, vocab_size)
        return logits

    def decode_step(
        self,
        memory: torch.Tensor,
        prev_tokens: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single autoregressive step (no caching).

        memory: (B, S, d_model)
        prev_tokens: (B, T_so_far)
        memory_key_padding_mask: (B, S)

        Returns:
            logits_last: (B, 1, vocab_size)
        """
        logits = self.forward(
            memory=memory,
            tgt_tokens=prev_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T_so_far, V)

        logits_last = logits[:, -1:, :]  # (B, 1, V)
        return logits_last
