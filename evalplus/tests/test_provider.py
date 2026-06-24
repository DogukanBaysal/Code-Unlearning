import pytest
import sys
import types

from evalplus.provider import make_model


def test_peft_name_rejected_for_unsupported_backend():
    with pytest.raises(ValueError, match="PEFT adapters are only supported"):
        make_model(
            model="base-model",
            peft_name="adapter-model",
            backend="vllm",
            dataset="humaneval",
        )


def test_peft_subfolder_rejected_without_peft_name():
    with pytest.raises(ValueError, match="requires peft_name"):
        make_model(
            model="base-model",
            peft_subfolder="adapter-subfolder",
            backend="hf",
            dataset="humaneval",
        )


def test_peft_subfolder_rejected_for_unsupported_backend():
    with pytest.raises(ValueError, match="PEFT adapters are only supported"):
        make_model(
            model="base-model",
            peft_name="adapter-model",
            peft_subfolder="adapter-subfolder",
            backend="vllm",
            dataset="humaneval",
        )


def test_peft_subfolder_forwarded_to_hf_backend(monkeypatch):
    captured_kwargs = {}

    class FakeHuggingFaceDecoder:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_hf_module = types.ModuleType("evalplus.provider.hf")
    fake_hf_module.HuggingFaceDecoder = FakeHuggingFaceDecoder
    monkeypatch.setitem(sys.modules, "evalplus.provider.hf", fake_hf_module)

    make_model(
        model="base-model",
        peft_name="adapter-model",
        peft_subfolder="adapter-subfolder",
        backend="hf",
        dataset="humaneval",
    )

    assert captured_kwargs["peft_subfolder"] == "adapter-subfolder"
