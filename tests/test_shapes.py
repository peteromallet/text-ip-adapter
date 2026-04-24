import os
from pathlib import Path

import torch

# Ensure HF_TOKEN is populated from local cache so Gemma3TextConfig.from_pretrained works.
_token_path = Path.home() / ".cache" / "huggingface" / "token"
if not os.environ.get("HF_TOKEN") and _token_path.exists():
    os.environ["HF_TOKEN"] = _token_path.read_text().strip()

from transformers import Gemma3TextConfig

from text_ip_adapter.model.encoder import ReferenceEncoder
from text_ip_adapter.model.projector import PrefixProjector


CFG = Gemma3TextConfig.from_pretrained("google/gemma-3-4b-pt")


def test_projector_shapes_match_gqa():
    P = 16
    inject_layers = list(range(17, 33))  # 16 layers, inclusive 17..32
    projector = PrefixProjector(
        hidden_size=CFG.hidden_size,
        num_kv_heads=CFG.num_key_value_heads,
        head_dim=CFG.head_dim,
        num_prefix_tokens=P,
        inject_layer_indices=inject_layers,
    )
    z = torch.randn(2, P, CFG.hidden_size)
    out = projector(z)
    assert set(out.keys()) == set(inject_layers)
    for li, (k, v) in out.items():
        assert k.shape == (2, P, CFG.num_key_value_heads, CFG.head_dim)
        assert v.shape == (2, P, CFG.num_key_value_heads, CFG.head_dim)


def test_projector_zero_init_output_weights():
    P = 16
    inject_layers = [17, 20, 32]
    projector = PrefixProjector(
        hidden_size=CFG.hidden_size,
        num_kv_heads=CFG.num_key_value_heads,
        head_dim=CFG.head_dim,
        num_prefix_tokens=P,
        inject_layer_indices=inject_layers,
    )
    for li in inject_layers:
        assert torch.all(projector.k_heads[str(li)].weight == 0)
        assert torch.all(projector.v_heads[str(li)].weight == 0)
        assert torch.all(projector.k_heads[str(li)].bias == 0)
        assert torch.all(projector.v_heads[str(li)].bias == 0)
    # Initial projector output must be exactly zero.
    z = torch.randn(1, P, CFG.hidden_size)
    out = projector(z)
    for li in inject_layers:
        k, v = out[li]
        assert torch.all(k == 0)
        assert torch.all(v == 0)


def test_encoder_output_shape():
    enc = ReferenceEncoder(hidden_size=CFG.hidden_size, num_queries=16, num_layers=2, num_heads=8)
    ref_hidden = torch.randn(3, 37, CFG.hidden_size)
    ref_mask = torch.ones(3, 37, dtype=torch.long)
    z = enc(ref_hidden, ref_mask)
    assert z.shape == (3, 16, CFG.hidden_size)


def test_inject_layers_count_matches_plan():
    # N=34 -> [N/2, N-2] = [17, 32] inclusive -> 16 layers.
    assert CFG.num_hidden_layers == 34
    inject = list(range(CFG.num_hidden_layers // 2, CFG.num_hidden_layers - 1))
    assert inject == list(range(17, 33))
    assert len(inject) == 16
