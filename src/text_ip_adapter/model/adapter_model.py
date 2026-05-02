from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import ExperimentConfig
from .encoder import ReferenceEncoder
from .injection import install_prefix_hooks, set_prefix_kv
from .projector import PrefixProjector


def _ensure_hf_token() -> None:
    # Populate HF_TOKEN from local cache file if not already in env.
    if os.environ.get("HF_TOKEN"):
        return
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        os.environ["HF_TOKEN"] = token_path.read_text().strip()


def load_base_model(model_id: str, torch_dtype: str = "bfloat16") -> tuple[nn.Module, Any, Any]:
    # Returns (base_causal_lm, tokenizer, text_config).
    _ensure_hf_token()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    raw_config = AutoConfig.from_pretrained(model_id)
    if hasattr(raw_config, "get_text_config"):
        text_config = raw_config.get_text_config()
    else:
        text_config = getattr(raw_config, "text_config", raw_config)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="eager")
    return model, tokenizer, text_config


def _decoder_layers(base_model: nn.Module) -> nn.ModuleList:
    if hasattr(base_model, "model") and hasattr(base_model.model, "language_model") and hasattr(base_model.model.language_model, "layers"):
        return base_model.model.language_model.layers
    if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        return base_model.language_model.layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model.layers
    if hasattr(base_model, "layers"):
        return base_model.layers
    raise AttributeError("Could not locate decoder layers on base model")


def _layer_kv_specs(base_model: nn.Module, text_config: Any, layer_indices: list[int]) -> dict[int, tuple[int, int]]:
    layers = _decoder_layers(base_model)
    specs: dict[int, tuple[int, int]] = {}
    layer_types = getattr(text_config, "layer_types", None)
    for li in layer_indices:
        attn = layers[li].self_attn
        head_dim = int(getattr(attn, "head_dim", getattr(text_config, "head_dim")))
        if hasattr(attn, "k_proj"):
            num_kv_heads = int(attn.k_proj.out_features // head_dim)
        else:
            layer_type = layer_types[li] if layer_types is not None else None
            is_full = layer_type == "full_attention"
            if is_full and getattr(text_config, "global_head_dim", None):
                head_dim = int(text_config.global_head_dim)
            num_kv_heads = int(getattr(text_config, "num_global_key_value_heads", None) or text_config.num_key_value_heads)
        specs[li] = (num_kv_heads, head_dim)
    return specs


class AdapterModel(nn.Module):
    # Glue wrapper: frozen base + trainable encoder + trainable projector + injection hooks.
    def __init__(self, base: nn.Module, encoder: ReferenceEncoder, projector: PrefixProjector, inject_layer_indices: list[int]):
        super().__init__()
        self.base = base
        self.encoder = encoder
        self.projector = projector
        self.inject_layer_indices = inject_layer_indices
        for p in self.base.parameters():
            p.requires_grad_(False)
        self._state = install_prefix_hooks(self.base, inject_layer_indices)

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> tuple["AdapterModel", Any]:
        base, tokenizer, text_config = load_base_model(cfg.model.base_model_id, cfg.model.torch_dtype)
        inject_layers = cfg.adapter.injection.layer_indices or list(
            range(cfg.adapter.injection.inject_layers_start, cfg.adapter.injection.inject_layers_end + 1)
        )
        layer_kv_specs = _layer_kv_specs(base, text_config, inject_layers)
        encoder = ReferenceEncoder(
            hidden_size=text_config.hidden_size,
            num_queries=cfg.adapter.encoder.num_queries,
            num_layers=cfg.adapter.encoder.perceiver_layers,
            num_heads=cfg.adapter.encoder.perceiver_heads,
        )
        projector = PrefixProjector(
            hidden_size=text_config.hidden_size,
            num_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            num_prefix_tokens=cfg.adapter.num_prefix_tokens,
            inject_layer_indices=inject_layers,
            layer_kv_specs=layer_kv_specs,
            hidden_mult=cfg.adapter.projector.hidden_mult,
            use_trunk=cfg.adapter.projector.use_trunk,
        )
        model = cls(base, encoder, projector, inject_layers)
        return model, tokenizer

    def _encode_reference(self, reference_ids: torch.Tensor, reference_mask: torch.Tensor) -> torch.Tensor:
        # Get frozen backbone hidden states for the reference.
        base_text = self._base_text_model()
        with torch.no_grad():
            out = base_text(input_ids=reference_ids, attention_mask=reference_mask, output_hidden_states=False, return_dict=True)
        # Upcast frozen-base output (bf16) to encoder's param dtype (fp32) at the trainable boundary.
        encoder_dtype = next(self.encoder.parameters()).dtype
        hidden = out.last_hidden_state.to(encoder_dtype)
        return self.encoder(hidden, reference_mask)

    def _base_text_model(self) -> nn.Module:
        # Gemma3ForConditionalGeneration has .model.language_model (multimodal wrapper)
        if hasattr(self.base, "model") and hasattr(self.base.model, "language_model"):
            return self.base.model.language_model
        if hasattr(self.base, "language_model"):
            return self.base.language_model
        return self.base.model if hasattr(self.base, "model") else self.base

    def forward(
        self,
        reference_ids: torch.Tensor,
        reference_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_prefix_kv: bool = False,
    ):
        z = self._encode_reference(reference_ids, reference_mask)
        prefix_kv = self.projector(z)
        set_prefix_kv(self._state, prefix_kv)
        try:
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        finally:
            set_prefix_kv(self._state, None)
        if return_prefix_kv:
            return out, prefix_kv
        return out

    @torch.no_grad()
    def generate(
        self,
        reference_ids: torch.Tensor,
        reference_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **gen_kwargs,
    ):
        z = self._encode_reference(reference_ids, reference_mask)
        prefix_kv = self.projector(z)
        set_prefix_kv(self._state, prefix_kv)
        try:
            return self.base.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        finally:
            set_prefix_kv(self._state, None)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        # Save trainable params only (encoder + projector).
        sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                sd[name] = p.detach().cpu()
        return sd

    def load_trainable_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        for name, tensor in sd.items():
            if name in own:
                own[name].data.copy_(tensor.to(own[name].device, dtype=own[name].dtype))
