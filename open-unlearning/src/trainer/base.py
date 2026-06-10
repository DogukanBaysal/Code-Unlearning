# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class ScheduledSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset.get_scheduled_indices(self.batch_size))

    def __len__(self):
        return len(self.dataset)


class ScheduledBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return iter(
            self.dataset.get_scheduled_batches(
                self.batch_size, drop_last=self.drop_last
            )
        )

    def __len__(self):
        return len(
            self.dataset.get_scheduled_batches(
                self.batch_size, drop_last=self.drop_last
            )
        )


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        if (
            self.train_dataset is not None
            and hasattr(self.train_dataset, "get_scheduled_indices")
            and self.train_dataset.get_scheduled_indices(self.args.train_batch_size)
            is not None
        ):
            return ScheduledSampler(self.train_dataset, self.args.train_batch_size)
        return super()._get_train_sampler()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if (
            hasattr(self.train_dataset, "get_scheduled_batches")
            and self.train_dataset.get_scheduled_batches(self._train_batch_size)
            is not None
        ):
            data_collator = self._get_collator_with_removed_columns(
                self.data_collator, description="Training"
            )
            dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=ScheduledBatchSampler(
                    self.train_dataset,
                    self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                ),
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor,
            )
            return self.accelerator.prepare(dataloader)

        return super().get_train_dataloader()

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
                return {}

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)
            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.processing_class,
                }
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
