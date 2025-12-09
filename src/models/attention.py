import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Pure Cross Attention Layer (LSTM Decoder ↔ Encoder Features)

    Query     → decoder hidden states  (B, Tq, D)
    Key/Value → image patch embeddings (B, Tk, D)
    """

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        """
        query     = decoder output      (B, Tq, D)
        key_value = visual embeddings   (B, Tk, D)
        """

        B, Tq, D = query.size()
        Tk = key_value.size(1)

        Q = self.to_q(query)                # (B,Tq,D)
        K = self.to_k(key_value)            # (B,Tk,D)
        V = self.to_v(key_value)            # (B,Tk,D)

        # split into heads
        Q = Q.view(B, Tq, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Tk, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Tk, self.heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B,H,Tq,Tk)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(attn, dim=-1)
        weights = self.dropout(weights)

        context = (weights @ V).transpose(1,2).contiguous().view(B, Tq, D)

        return self.out(context)  # (B,Tq,D)