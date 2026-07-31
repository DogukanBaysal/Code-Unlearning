import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def unlearn_modules():
    module_names = (
        "trainer",
        "trainer.utils",
        "trainer.unlearn",
        "trainer.unlearn.base",
        "trainer.unlearn.grad_diff",
        "trainer.unlearn.npo",
        "trainer.unlearn.prod",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    trainer_package = types.ModuleType("trainer")
    trainer_package.__path__ = [str(SRC_ROOT / "trainer")]
    unlearn_package = types.ModuleType("trainer.unlearn")
    unlearn_package.__path__ = [str(SRC_ROOT / "trainer" / "unlearn")]
    base_module = types.ModuleType("trainer.unlearn.base")
    base_module.UnlearnTrainer = object

    sys.modules["trainer"] = trainer_package
    sys.modules["trainer.unlearn"] = unlearn_package
    sys.modules["trainer.unlearn.base"] = base_module

    try:
        _load_source_module("trainer.utils", SRC_ROOT / "trainer" / "utils.py")
        grad_diff = _load_source_module(
            "trainer.unlearn.grad_diff",
            SRC_ROOT / "trainer" / "unlearn" / "grad_diff.py",
        )
        npo = _load_source_module(
            "trainer.unlearn.npo",
            SRC_ROOT / "trainer" / "unlearn" / "npo.py",
        )
        prod = _load_source_module(
            "trainer.unlearn.prod",
            SRC_ROOT / "trainer" / "unlearn" / "prod.py",
        )
        yield SimpleNamespace(grad_diff=grad_diff, npo=npo, prod=prod)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


class CountingModel:
    def __init__(self, offset=0.0, trainable=True):
        self.calls = 0
        self.offset = offset
        self.trainable = trainable

    def __call__(self, input_ids, attention_mask=None, labels=None):
        self.calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.arange(
            batch_size * sequence_length * 5,
            dtype=torch.float32,
        ).reshape(batch_size, sequence_length, 5)
        logits = (logits + self.offset).requires_grad_(self.trainable)
        return SimpleNamespace(logits=logits, loss=logits.mean())


def model_batch():
    return {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]]),
    }


def test_npo_skips_zero_weight_retain_forward(unlearn_modules, monkeypatch):
    trainer = unlearn_modules.npo.NPO.__new__(unlearn_modules.npo.NPO)
    trainer.alpha = 0.0
    trainer.gamma = 1.0
    trainer.beta = 0.1
    trainer.retain_loss_type = "NLL"
    trainer.ref_model = CountingModel(trainable=False)
    student = CountingModel()

    def fake_dpo_loss(model, ref_model, win_inputs, lose_inputs, beta):
        outputs = model(**lose_inputs)
        return outputs.loss, outputs

    monkeypatch.setattr(unlearn_modules.npo, "compute_dpo_loss", fake_dpo_loss)

    _, outputs = trainer.compute_loss(
        student,
        {"forget": model_batch(), "retain": model_batch()},
        return_outputs=True,
    )

    assert student.calls == 1
    assert outputs is not None


def test_prod_skips_zero_weight_retain_forward(unlearn_modules):
    trainer = unlearn_modules.prod.PROD.__new__(unlearn_modules.prod.PROD)
    trainer.alpha = 0.0
    trainer.gamma = 1.0
    trainer.top_p = 0.8
    trainer.temperature = None
    trainer.N = 1
    trainer.max_N = None
    trainer.prod_alpha = 0.0
    trainer.retain_loss_type = "NLL"
    trainer.ref_model = CountingModel(offset=0.5, trainable=False)
    student = CountingModel()

    loss, outputs = trainer.compute_loss(
        student,
        {"forget": model_batch(), "retain": model_batch()},
        return_outputs=True,
    )

    assert student.calls == 1
    assert trainer.ref_model.calls == 1
    assert torch.isfinite(loss)
    assert outputs is not None


@pytest.mark.parametrize("trainer_name", ["NPO", "PROD"])
def test_kl_retain_batch_uses_one_student_forward(unlearn_modules, trainer_name):
    module = unlearn_modules.npo if trainer_name == "NPO" else unlearn_modules.prod
    trainer_class = getattr(module, trainer_name)
    trainer = trainer_class.__new__(trainer_class)
    trainer.alpha = 1.0
    trainer.gamma = 1.0
    trainer.retain_loss_type = "KL"
    trainer.ref_model = CountingModel(offset=0.5, trainable=False)
    student = CountingModel()
    trainer.model = student

    loss, outputs = trainer.compute_loss(
        student,
        {"retain": model_batch()},
        return_outputs=True,
    )

    assert student.calls == 1
    assert trainer.ref_model.calls == 1
    assert torch.isfinite(loss)
    assert outputs is not None
