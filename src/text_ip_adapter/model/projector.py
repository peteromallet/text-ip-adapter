from __future__ import annotations

import torch
import torch.nn as nn


class PrefixProjector(nn.Module):
    # Shared MLP trunk, per-layer zero-init K/V heads producing GQA-aware prefix tensors.
    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_prefix_tokens: int,
        inject_layer_indices: list[int],
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.inject_layer_indices = list(inject_layer_indices)
        per_token_dim = num_kv_heads * head_dim
        trunk_hidden = hidden_size * hidden_mult
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, hidden_size),
        )
        self.k_heads = nn.ModuleDict()
        self.v_heads = nn.ModuleDict()
        for li in self.inject_layer_indices:
            k = nn.Linear(hidden_size, per_token_dim)
            v = nn.Linear(hidden_size, per_token_dim)
            nn.init.zeros_(k.weight); nn.init.zeros_(k.bias)
            nn.init.zeros_(v.weight); nn.init.zeros_(v.bias)
            self.k_heads[str(li)] = k
            self.v_heads[str(li)] = v

    def forward(self, z: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        # z: (B, P, H) -> {layer_idx: (K, V)} with K,V shape (B, P, num_kv_heads, head_dim).
        B, P, _ = z.shape
        h = self.trunk(z)
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in self.inject_layer_indices:
            k = self.k_heads[str(li)](h).view(B, P, self.num_kv_heads, self.head_dim)
            v = self.v_heads[str(li)](h).view(B, P, self.num_kv_heads, self.head_dim)
            out[li] = (k, v)
        return out
