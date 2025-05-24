import torch.nn as nn
import torch

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout),
            num_layers=depth
        )
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1)]
        x = self.transformer(x)
        return self.to_logits(x)
