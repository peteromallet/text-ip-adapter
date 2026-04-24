from __future__ import annotations

import torch
import torch.nn as nn


class PerceiverLayer(nn.Module):
    # One cross-attn block followed by self-attn on the queries, both with residual + FFN.
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 2) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_self_q = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim))

    def forward(self, queries: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        # context_mask: (B, S) bool, True for valid tokens; MHA expects key_padding_mask True=pad.
        kpm = None
        if context_mask is not None:
            kpm = ~context_mask.bool()
        q = self.norm_q(queries)
        k = v = self.norm_kv(context)
        out, _ = self.cross_attn(q, k, v, key_padding_mask=kpm, need_weights=False)
        queries = queries + out
        q2 = self.norm_self_q(queries)
        out2, _ = self.self_attn(q2, q2, q2, need_weights=False)
        queries = queries + out2
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class ReferenceEncoder(nn.Module):
    # Frozen backbone + learned perceiver queries. Caller supplies the frozen backbone.
    def __init__(self, hidden_size: int, num_queries: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.layers = nn.ModuleList([PerceiverLayer(hidden_size, num_heads) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, ref_hidden: torch.Tensor, ref_mask: torch.Tensor | None) -> torch.Tensor:
        # ref_hidden: (B, S, H) frozen backbone output; ref_mask: (B, S) valid=1.
        bsz = ref_hidden.shape[0]
        q = self.queries.unsqueeze(0).expand(bsz, -1, -1).to(ref_hidden.dtype)
        for layer in self.layers:
            q = layer(q, ref_hidden, ref_mask)
        return self.out_norm(q)
