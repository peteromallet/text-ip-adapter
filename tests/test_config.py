import pytest
from pydantic import ValidationError

from text_ip_adapter.config import AdapterConfig, EncoderConfig, ExperimentConfig


def test_default_prefix_contract_ok():
    cfg = ExperimentConfig()
    assert cfg.adapter.num_prefix_tokens == cfg.adapter.num_queries == 16


def test_prefix_mismatch_rejected():
    with pytest.raises(ValidationError):
        AdapterConfig(num_prefix_tokens=8, num_queries=16)


def test_encoder_num_queries_mismatch_rejected():
    with pytest.raises(ValidationError):
        AdapterConfig(num_prefix_tokens=16, num_queries=16, encoder=EncoderConfig(num_queries=8))
