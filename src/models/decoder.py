#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoder.py

LSTM decoder for pix2code-style model.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    LSTM decoder with:
    - token embedding
    - conditioning on encoder features via initial hidden state
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        enc_dim: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # project encoder feature into initial hidden state
        self.enc_to_h = nn.Linear(enc_dim, hidden_dim)
        self.enc_to_c = nn.Linear(enc_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(
        self, enc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc_feat: (B, enc_dim)
        Returns (h0, c0) each of shape (num_layers, B, hidden_dim)
        """
        h0 = self.enc_to_h(enc_feat)  # (B, hidden_dim)
        c0 = self.enc_to_c(enc_feat)  # (B, hidden_dim)

        # reshape for LSTM
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, H)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(
        self,
        enc_feat: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        enc_feat: (B, enc_dim)
        tgt_tokens: (B, T)

        Returns:
            logits: (B, T, vocab_size)
        """
        h0, c0 = self.init_hidden(enc_feat)  # each: (num_layers, B, H)
        emb = self.embedding(tgt_tokens)     # (B, T, emb_dim)

        outputs, _ = self.lstm(emb, (h0, c0))   # (B, T, hidden_dim)
        logits = self.fc_out(outputs)           # (B, T, vocab_size)
        return logits

    def decode_step(
        self,
        enc_feat: torch.Tensor,
        prev_token: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step for greedy search.

        enc_feat: (B, enc_dim)
        prev_token: (B, 1)
        hidden: (h, c) or None

        Returns:
            logits: (B, 1, vocab_size)
            hidden: updated hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(enc_feat)

        emb = self.embedding(prev_token)     # (B, 1, emb_dim)
        output, hidden = self.lstm(emb, hidden)  # (B, 1, hidden_dim)
        logits = self.fc_out(output)         # (B, 1, vocab_size)
        return logits, hidden
