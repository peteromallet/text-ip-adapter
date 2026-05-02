from __future__ import annotations

import torch
import torch.nn as nn


class PrefixProjector(nn.Module):
    # Shared MLP trunk + per-layer zero-init K/V heads producing GQA-aware prefix tensors.
    # When use_trunk=False (experiment 006), the trunk becomes Identity and each per-layer
    # head reads z directly. Tests whether the shared subspace bottleneck (diagnosed in
    # experiments 003 and 005) was the structural cause of weak style conditioning.
    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_prefix_tokens: int,
        inject_layer_indices: list[int],
        layer_kv_specs: dict[int, tuple[int, int]] | None = None,
        hidden_mult: int = 2,
        use_trunk: bool = True,
    ) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.inject_layer_indices = list(inject_layer_indices)
        self.layer_kv_specs = layer_kv_specs or {li: (num_kv_heads, head_dim) for li in self.inject_layer_indices}
        self.use_trunk = use_trunk
        if use_trunk:
            trunk_hidden = hidden_size * hidden_mult
            self.trunk = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, trunk_hidden),
                nn.GELU(),
                nn.Linear(trunk_hidden, hidden_size),
            )
        else:
            # Per-layer heads read z directly. Per-layer LayerNorm only (no shared MLP).
            # Each head specializes independently — the experiment 006 hypothesis.
            self.trunk = nn.Identity()
        self.k_heads = nn.ModuleDict()
        self.v_heads = nn.ModuleDict()
        # Per-layer LayerNorm when no shared trunk, to keep activations stable per head.
        self.k_norms = nn.ModuleDict() if not use_trunk else None
        self.v_norms = nn.ModuleDict() if not use_trunk else None
        for li in self.inject_layer_indices:
            layer_num_kv_heads, layer_head_dim = self.layer_kv_specs[li]
            per_token_dim = layer_num_kv_heads * layer_head_dim
            k = nn.Linear(hidden_size, per_token_dim)
            v = nn.Linear(hidden_size, per_token_dim)
            nn.init.zeros_(k.weight); nn.init.zeros_(k.bias)
            nn.init.zeros_(v.weight); nn.init.zeros_(v.bias)
            self.k_heads[str(li)] = k
            self.v_heads[str(li)] = v
            if not use_trunk:
                self.k_norms[str(li)] = nn.LayerNorm(hidden_size)
                self.v_norms[str(li)] = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        # z: (B, P, H) -> {layer_idx: (K, V)} with K,V shape (B, P, num_kv_heads, head_dim).
        B, P, _ = z.shape
        h_shared = self.trunk(z)  # Identity if use_trunk=False, MLP output otherwise
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for li in self.inject_layer_indices:
            if self.use_trunk:
                hk, hv = h_shared, h_shared
            else:
                # Per-layer LayerNorm before each head (replaces the shared LayerNorm in the trunk).
                hk = self.k_norms[str(li)](z)
                hv = self.v_norms[str(li)](z)
            layer_num_kv_heads, layer_head_dim = self.layer_kv_specs[li]
            k = self.k_heads[str(li)](hk).view(B, P, layer_num_kv_heads, layer_head_dim)
            v = self.v_heads[str(li)](hv).view(B, P, layer_num_kv_heads, layer_head_dim)
            out[li] = (k, v)
        return out
