import hydra
import logging
import os
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything

logger = logging.getLogger(__name__)


def _is_world_process_zero(trainer):
    is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
    if callable(is_world_process_zero):
        return is_world_process_zero()
    return True


@contextmanager
def track_codecarbon_emissions(codecarbon_cfg, trainer, trainer_args, task_name):
    if not codecarbon_cfg or not codecarbon_cfg.get("enabled", False):
        yield
        return

    if not _is_world_process_zero(trainer):
        yield
        return

    try:
        from codecarbon import EmissionsTracker
    except ImportError as e:
        raise ImportError(
            "CodeCarbon tracking is enabled, but `codecarbon` is not installed. "
            "Install the project requirements or run with codecarbon.enabled=false."
        ) from e

    output_dir = codecarbon_cfg.get(
        "output_dir", os.path.join(trainer_args.output_dir, "emissions")
    )
    os.makedirs(output_dir, exist_ok=True)

    tracker_kwargs = {
        "project_name": codecarbon_cfg.get("project_name", task_name),
        "output_dir": output_dir,
        "output_file": codecarbon_cfg.get("output_file", "emissions.csv"),
        "measure_power_secs": codecarbon_cfg.get("measure_power_secs", 15),
        "log_level": codecarbon_cfg.get("log_level", "error"),
        "save_to_file": codecarbon_cfg.get("save_to_file", True),
    }
    tracker_kwargs = {
        key: value for key, value in tracker_kwargs.items() if value is not None
    }

    tracker = EmissionsTracker(**tracker_kwargs)
    tracker.start()
    try:
        yield
    finally:
        emissions = tracker.stop()
        logger.info(
            "CodeCarbon emissions for %s: %s kg CO2eq. Logs saved to %s",
            task_name,
            emissions,
            output_dir,
        )


def _to_container(value):
    if value is None or isinstance(value, str):
        return value
    return OmegaConf.to_container(value, resolve=True)


def _default_adapter_allow_patterns(hub_cfg):
    adapter_patterns = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
        "README.md",
        "trainer_state.json",
        "training_args.bin",
        "run_config.yaml",
    ]

    if hub_cfg.get("include_tokenizer", True):
        adapter_patterns.extend(
            [
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.*",
                "merges.txt",
            ]
        )

    allow_patterns = list(adapter_patterns)
    if hub_cfg.get("include_checkpoints", True):
        allow_patterns.extend(
            [os.path.join("checkpoint-*", pattern) for pattern in adapter_patterns]
        )

    if hub_cfg.get("include_emissions", True):
        allow_patterns.append(os.path.join("emissions", "*"))

    if hub_cfg.get("include_settings", True):
        allow_patterns.append(os.path.join(".hydra", "*"))

    return allow_patterns


def _default_full_model_allow_patterns(hub_cfg):
    model_patterns = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
        "README.md",
        "trainer_state.json",
        "training_args.bin",
        "run_config.yaml",
    ]

    allow_patterns = list(model_patterns)
    if hub_cfg.get("include_checkpoints", True):
        allow_patterns.extend(
            [os.path.join("checkpoint-*", pattern) for pattern in model_patterns]
        )

    if hub_cfg.get("include_tokenizer", True):
        tokenizer_patterns = [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.*",
            "merges.txt",
        ]
        allow_patterns.extend(tokenizer_patterns)
        if hub_cfg.get("include_checkpoints", True):
            allow_patterns.extend(
                [
                    os.path.join("checkpoint-*", pattern)
                    for pattern in tokenizer_patterns
                ]
            )

    if hub_cfg.get("include_emissions", True):
        allow_patterns.append(os.path.join("emissions", "*"))

    if hub_cfg.get("include_settings", True):
        allow_patterns.append(os.path.join(".hydra", "*"))

    return allow_patterns


def save_run_settings(cfg, trainer, trainer_args):
    if not _is_world_process_zero(trainer):
        return

    os.makedirs(trainer_args.output_dir, exist_ok=True)
    config_path = os.path.join(trainer_args.output_dir, "run_config.yaml")
    OmegaConf.save(config=cfg, f=config_path, resolve=True)


def push_artifacts_to_hub(hub_cfg, trainer, trainer_args, task_name):
    if not hub_cfg or not hub_cfg.get("enabled", False):
        return

    if not _is_world_process_zero(trainer):
        return

    repo_id = hub_cfg.get("repo_id", None)
    if not repo_id:
        raise ValueError(
            "hub_adapter.enabled=true requires hub_adapter.repo_id to be set."
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "Hub adapter upload is enabled, but `huggingface_hub` is not installed."
        ) from e

    folder_path = hub_cfg.get("folder_path", trainer_args.output_dir)
    if not os.path.isdir(folder_path):
        raise ValueError(f"Hub adapter upload folder does not exist: {folder_path}")

    allow_patterns = _to_container(hub_cfg.get("allow_patterns", None))
    if allow_patterns is None:
        artifact_type = hub_cfg.get("artifact_type", "adapter")
        if artifact_type == "adapter":
            allow_patterns = _default_adapter_allow_patterns(hub_cfg)
        elif artifact_type == "full_model":
            allow_patterns = _default_full_model_allow_patterns(hub_cfg)
        else:
            raise ValueError(
                "hub_adapter.artifact_type must be adapter or full_model, got "
                f"{artifact_type!r}."
            )

    token = hub_cfg.get("token", None)
    repo_type = hub_cfg.get("repo_type", "model")
    private = hub_cfg.get("private", None)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type=repo_type,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        path_in_repo=hub_cfg.get("path_in_repo", None),
        commit_message=hub_cfg.get(
            "commit_message", f"Upload unlearning artifacts for {task_name}"
        ),
        commit_description=hub_cfg.get("commit_description", None),
        token=token,
        repo_type=repo_type,
        revision=hub_cfg.get("revision", None),
        create_pr=hub_cfg.get("create_pr", False),
        allow_patterns=allow_patterns,
        ignore_patterns=_to_container(hub_cfg.get("ignore_patterns", None)),
        delete_patterns=_to_container(hub_cfg.get("delete_patterns", None)),
    )
    logger.info(
        "Uploaded training artifacts from %s to Hugging Face Hub repo %s",
        folder_path,
        repo_id,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        model_cfg=model_cfg,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )
    save_run_settings(cfg, trainer, trainer_args)

    if trainer_args.do_train:
        with track_codecarbon_emissions(
            cfg.get("codecarbon", None),
            trainer=trainer,
            trainer_args=trainer_args,
            task_name=cfg.task_name,
        ):
            trainer.train()
            trainer.save_state()
            trainer.save_model(trainer_args.output_dir)
        push_artifacts_to_hub(
            cfg.get("hub_adapter", None),
            trainer=trainer,
            trainer_args=trainer_args,
            task_name=cfg.task_name,
        )

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
