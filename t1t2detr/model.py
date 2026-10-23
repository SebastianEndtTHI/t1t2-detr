"""Detection transformer for T1-T2 compartments (architecture of the diffusion DETR, MLP encoder)."""

import torch
from torch import nn


def _mlp(dims, final_act=True):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or final_act:
            layers += [nn.LayerNorm(b), nn.ReLU()]
    return nn.Sequential(*layers)


class T1T2DETR(nn.Module):
    """
    Per query: [T1, T2, weight, existence logit]; T1/T2 in the normalized [0, 1] space of the ParamScaler.
    The existence scores of all queries are predicted jointly from their concatenated features.
    """

    def __init__(self, input_dim, hidden_dim, fs_dim, n_queries, n_layers, n_heads):
        super().__init__()
        self.n_queries = n_queries
        self.encoder = _mlp([input_dim, hidden_dim, hidden_dim, hidden_dim, fs_dim])
        self.queries = nn.Embedding(n_queries, fs_dim)
        layer = nn.TransformerDecoderLayer(fs_dim, n_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers, norm=nn.LayerNorm(fs_dim))
        self.heads = nn.ModuleList([_mlp([fs_dim, fs_dim, fs_dim, 1], final_act=False) for _ in range(3)])
        red = fs_dim // 4
        self.reduce = _mlp([fs_dim, red])
        self.exist_head = _mlp([red * n_queries, fs_dim, fs_dim, n_queries], final_act=False)

    def forward(self, x):
        memory = self.encoder(x).unsqueeze(1)                                   # (B, 1, fs)
        hs = self.decoder(self.queries.weight.unsqueeze(0).expand(len(x), -1, -1), memory)
        params = torch.cat([torch.sigmoid(h(hs)) for h in self.heads], dim=-1)   # (B, Q, 3)
        exist = self.exist_head(self.reduce(hs).flatten(1)).unsqueeze(-1)       # (B, Q, 1)
        return torch.cat([params, exist], dim=-1)
