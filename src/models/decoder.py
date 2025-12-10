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

    Now supports Pix2CodeModel arguments:
    dim, depth, heads, max_len
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,          # NEW → replaces emb_dim + hidden_dim + enc_dim
        depth: int = 1,          # NEW → num_layers
        heads: int = None,       # NEW but unused (Transformer compatibility)
        max_len: int = None,     # NEW but unused
        emb_dim: int = None,     # original param (optional override)
        hidden_dim: int = None,  # original param (optional override)
        enc_dim: int = None,     # original param (optional override)
        num_layers: int = None,  # original param (optional override)
        **kwargs                 # catches all other unused arguments
    ):
        super().__init__()

        # ----------------------------
        # Handle backward compatibility
        # ----------------------------
        emb_dim = emb_dim or dim
        hidden_dim = hidden_dim or dim
        enc_dim = enc_dim or dim
        num_layers = num_layers or depth

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.num_layers = num_layers

        # ----------------------------
        # Layers
        # ----------------------------
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

    # ------------------------------------------------------------------------
    # Hidden state initialization
    # ------------------------------------------------------------------------
    def init_hidden(
        self, enc_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc_feat: (B, enc_dim)
        Returns (h0, c0): (num_layers, B, hidden_dim)
        """
        B = enc_feat.size(0)

        h0 = self.enc_to_h(enc_feat).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.enc_to_c(enc_feat).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    # ------------------------------------------------------------------------
    # Teacher-forcing forward pass
    # ------------------------------------------------------------------------
    def forward(
        self,
        enc_feat: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        enc_feat: (B, enc_dim)
        tgt_tokens: (B, T)

        Returns logits: (B, T, vocab_size)
        """
        h0, c0 = self.init_hidden(enc_feat)
        emb = self.embedding(tgt_tokens)         # (B, T, emb_dim)
        outputs, _ = self.lstm(emb, (h0, c0))    # (B, T, hidden_dim)
        return self.fc_out(outputs)              # (B, T, vocab_size)

    # ------------------------------------------------------------------------
    # One-step decoding for greedy search
    # ------------------------------------------------------------------------
    def decode_step(
        self,
        enc_feat: torch.Tensor,
        prev_token: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        prev_token: (B, 1)
        Returns:
            logits: (B, 1, vocab_size)
            hidden: updated hidden
        """
        if hidden is None:
            hidden = self.init_hidden(enc_feat)

        emb = self.embedding(prev_token)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden
