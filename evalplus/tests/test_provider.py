import pytest

from evalplus.provider import make_model


def test_peft_name_rejected_for_unsupported_backend():
    with pytest.raises(ValueError, match="PEFT adapters are only supported"):
        make_model(
            model="base-model",
            peft_name="adapter-model",
            backend="vllm",
            dataset="humaneval",
        )
