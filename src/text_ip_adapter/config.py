from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    base_model_id: str = "google/gemma-3-4b-pt"
    torch_dtype: str = "bfloat16"


class EncoderConfig(BaseModel):
    num_queries: int = 16
    perceiver_layers: int = 2
    perceiver_heads: int = 8


class ProjectorConfig(BaseModel):
    hidden_mult: int = 2
    use_trunk: bool = True  # exp 006: set False to drop shared MLP trunk; per-layer heads read z directly


class InjectionConfig(BaseModel):
    inject_layers_start: int = 17
    inject_layers_end: int = 32
    layer_indices: list[int] | None = None


class AdapterConfig(BaseModel):
    num_prefix_tokens: int = 16
    num_queries: int = 16
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    projector: ProjectorConfig = Field(default_factory=ProjectorConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)

    @model_validator(mode="after")
    def validate_prefix_contract(self) -> "AdapterConfig":
        if self.num_prefix_tokens != self.num_queries:
            raise ValueError("num_prefix_tokens must equal num_queries")
        if self.num_queries != self.encoder.num_queries:
            raise ValueError("adapter.num_queries must equal adapter.encoder.num_queries")
        return self


class DataConfig(BaseModel):
    train_path: str = "data/pairs/train.jsonl"
    val_path: str = "data/pairs/val.jsonl"
    test_path: str = "data/pairs/test.jsonl"
    reference_max: int = 512
    instruction_max: int = 128
    prompt_format: str = "instruction"  # instruction | paired_completion
    prompt_reference_max: int = 384
    target_max: int = 256


class TrainingConfig(BaseModel):
    output_dir: str = "checkpoints/stage1_gemma"
    batch_size: int = 4
    max_steps: int = 2000
    lr_projector: float = 1e-4
    lr_encoder: float = 5e-5
    warmup: int = 200
    min_lr_ratio: float = 0.0
    log_every: int = 10
    save_every: int = 500
    gradient_clip: float = 1.0
    seed: int = 42
    mixed_precision: str = "bf16"
    sample_every: int = 100
    sample_max_new_tokens: int = 120
    probe_path: str = "data/pairs/probes.jsonl"
    # Contrastive loss on projector K/V outputs (exp 004+).
    contrastive_weight: float = 0.0  # 0 = disabled; >0 enables contrastive term
    contrastive_clamp: bool = True  # clamp off-diag cosines at 0 before averaging (penalize positive)
    style_triplet_weight: float = 0.0  # optional own/positive-author/negative-author K/V triplet loss
    style_triplet_margin: float = 0.2
    style_contrastive_weight: float = 0.0  # batch-size-independent own/positive vs own/negative K/V loss
    style_contrastive_temperature: float = 0.2
    init_from: str = ""  # path to a trainable state_dict (.pt) to warm-start from; empty = train from scratch


class RunPodStageConfig(BaseModel):
    storage_name: str | None = "Peter"
    remote_root: str = "/workspace/text-ip-adapter"
    keep_alive_on_success: bool = True


class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    runpod: RunPodStageConfig = Field(default_factory=RunPodStageConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ExperimentConfig.model_validate(data)
