from __future__ import annotations

from types import MethodType
from typing import Callable, Optional

import torch
import torch.nn as nn

# Prefix K is prepended AFTER the attention module's internal RoPE is applied to content K,
# so prefix tensors are never rotated. The additive attention mask is extended with zeros
# (unmasked) on the left for prefix columns so both sliding and full layers can see the prefix.


class InjectionState:
    # Holds the current forward's prefix K/V dict keyed by base-layer index.
    def __init__(self) -> None:
        self.prefix_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None


def _patched_attn_forward_factory(state: InjectionState):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, eager_attention_forward

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": self.sliding_window}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Prefix injection (prefix K is NOT RoPE-rotated).
        prefix_len = 0
        if state.prefix_kv is not None:
            prefix_kv = state.prefix_kv.get(self.layer_idx)
            if prefix_kv is not None:
                pk, pv = prefix_kv  # (B, P, num_kv_heads, head_dim)
                pk_t = pk.permute(0, 2, 1, 3).to(key_states.dtype).to(key_states.device)
                pv_t = pv.permute(0, 2, 1, 3).to(value_states.dtype).to(value_states.device)
                if pk_t.shape[0] != key_states.shape[0]:
                    pk_t = pk_t.expand(key_states.shape[0], -1, -1, -1)
                    pv_t = pv_t.expand(value_states.shape[0], -1, -1, -1)
                key_states = torch.cat([pk_t, key_states], dim=2)
                value_states = torch.cat([pv_t, value_states], dim=2)
                prefix_len = pk_t.shape[2]

        if prefix_len > 0 and attention_mask is not None:
            pad = torch.zeros(
                (*attention_mask.shape[:-1], prefix_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)

        # Use eager attention since spliced K length will not match SDPA/flash expectations.
        attention_interface: Callable = eager_attention_forward
        if attention_mask is not None:
            attention_mask = attention_mask.to(query_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **{k: v for k, v in kwargs.items() if k in ("output_attentions",)},
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _patched_gemma4_text_attn_forward_factory(state: InjectionState):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values=None,
        **kwargs,
    ):
        from transformers.models.gemma4.modeling_gemma4 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = key_states, value_states

        prefix_len = 0
        if state.prefix_kv is not None:
            prefix_kv = state.prefix_kv.get(self.layer_idx)
            if prefix_kv is not None:
                pk, pv = prefix_kv
                pk_t = pk.permute(0, 2, 1, 3).to(key_states.dtype).to(key_states.device)
                pv_t = pv.permute(0, 2, 1, 3).to(value_states.dtype).to(value_states.device)
                if pk_t.shape[0] != key_states.shape[0]:
                    pk_t = pk_t.expand(key_states.shape[0], -1, -1, -1)
                    pv_t = pv_t.expand(value_states.shape[0], -1, -1, -1)
                key_states = torch.cat([pk_t, key_states], dim=2)
                value_states = torch.cat([pv_t, value_states], dim=2)
                prefix_len = pk_t.shape[2]

        if prefix_len > 0 and attention_mask is not None:
            pad = torch.zeros(
                (*attention_mask.shape[:-1], prefix_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(query_states)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def install_prefix_hooks(base_model: nn.Module, inject_layer_indices: list[int]) -> InjectionState:
    state = InjectionState()
    layers = _find_decoder_layers(base_model)
    for li in inject_layer_indices:
        attn = layers[li].self_attn
        if attn.__class__.__name__ == "Gemma4TextAttention":
            patched = _patched_gemma4_text_attn_forward_factory(state)
        else:
            patched = _patched_attn_forward_factory(state)
        attn.forward = MethodType(patched, attn)
    return state


def set_prefix_kv(state: InjectionState, prefix_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None) -> None:
    state.prefix_kv = prefix_kv


def _find_decoder_layers(base_model: nn.Module) -> nn.ModuleList:
    # Gemma3ForConditionalGeneration (multimodal) -> .model.language_model.layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "language_model") and hasattr(base_model.model.language_model, "layers"):
        return base_model.model.language_model.layers
    if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        return base_model.language_model.layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model.layers
    if hasattr(base_model, "layers"):
        return base_model.layers
    raise AttributeError("Could not locate decoder layers on base model")
