import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1, use_cross_attn=True):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.use_cross_attn = use_cross_attn
        self.cross_attn = CrossAttention(dim=dim, heads=heads, dropout=dropout) if use_cross_attn else None

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if use_cross_attn else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tokens, visual_embeds=None, mask=None):
        # 1) Masked self-attention (autoregressive)
        x = self.norm1(tokens)
        x,_ = self.self_attn(x, x, x, attn_mask=mask)
        tokens = tokens + x

        # 2) Cross-attention to encoder features
        if self.use_cross_attn and visual_embeds is not None:
            x = self.norm2(tokens)
            context = self.cross_attn(x, visual_embeds)
            tokens = tokens + context

        # 3) Feed-forward expansion
        x = self.norm3(tokens)
        tokens = tokens + self.mlp(x)

        return tokens


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, max_len=1024, use_global_summary=False):
        super().__init__()

        self.use_global_summary = use_global_summary

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Embedding(max_len, dim)

        if use_global_summary:
            self.summary_proj = nn.Linear(dim, dim)
        else:
            self.summary_proj = None

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(dim, heads=heads, use_cross_attn=not use_global_summary) for _ in range(depth)
        ])
        self.output_fc = nn.Linear(dim, vocab_size)

    def forward(self, visual_embeds, token_ids):
        B,T = token_ids.shape
        pos = torch.arange(0,T,device=token_ids.device).unsqueeze(0)

        x = self.token_emb(token_ids) + self.pos_emb(pos)

        # Exp1 variant: inject only a global encoder summary (CLS) and skip cross-attn
        if self.use_global_summary:
            summary = visual_embeds[:, 0, :]  # CLS token
            summary = self.summary_proj(summary).unsqueeze(1)  # (B,1,D)
            x = x + summary

        # autoregressive attention mask (no peeking ahead)
        mask = torch.triu(torch.ones(T,T,device=token_ids.device)*float("-inf"),1)

        for layer in self.layers:
            x = layer(x, None if self.use_global_summary else visual_embeds, mask)

        return self.output_fc(x)  # (B,T,vocab_size)


class TransformerDecoderNoCross(nn.Module):
    """
    Decoder without cross-attention; conditions only on a global encoder summary (CLS).
    Useful for the simpler baseline experiment.
    """
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, max_len=1024):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Embedding(max_len, dim)
        self.summary_proj = nn.Linear(dim, dim)

        # no cross-attention in these blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(dim, heads=heads, use_cross_attn=False) for _ in range(depth)
        ])
        self.output_fc = nn.Linear(dim, vocab_size)

    def forward(self, visual_embeds, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(0, T, device=token_ids.device).unsqueeze(0)

        x = self.token_emb(token_ids) + self.pos_emb(pos)

        # add projected CLS summary to every token embedding
        summary = visual_embeds[:, 0, :]  # CLS token
        summary = self.summary_proj(summary).unsqueeze(1)  # (B,1,D)
        x = x + summary

        mask = torch.triu(torch.ones(T, T, device=token_ids.device) * float("-inf"), 1)

        for layer in self.layers:
            x = layer(x, None, mask)

        return self.output_fc(x)
