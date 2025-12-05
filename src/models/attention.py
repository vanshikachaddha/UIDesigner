class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.ReLU(),
            nn.Linear(dim*mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tokens, visual_embeds):
        # 1) masked self-attention over text tokens
        x = self.norm1(tokens)
        x,_ = self.self_attn(x,x,x, need_weights=False)  # causal mask later
        tokens = tokens + x

        # 2) cross-attention — decoder queries image
        x = self.norm2(tokens)
        x,_ = self.cross_attn(x, visual_embeds, visual_embeds)  # Q=tokens, K/V=image
        tokens = tokens + x

        # 3) feed-forward MLP
        x = self.norm3(tokens)
        tokens = tokens + self.mlp(x)

        return tokens
