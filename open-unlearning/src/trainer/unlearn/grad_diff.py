import copy
import os
import torch
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

hf_home = os.getenv("HF_HOME", default=None)


def _resolve_torch_dtype(torch_dtype):
    if torch_dtype is None or isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported reference model torch_dtype: {torch_dtype}")


class GradDiff(UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        reference_model_args=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.reference_model_args = reference_model_args
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        if self.reference_model_args is not None:
            ref_model = self._load_reference_model(self.reference_model_args)
        else:
            ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        if self.reference_model_args is not None:
            return ref_model
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def _load_reference_model(self, reference_model_args):
        reference_model_args = dict(reference_model_args)
        model_path = reference_model_args.pop("pretrained_model_name_or_path", None)
        peft_name = reference_model_args.pop("peft_name", None)
        peft_checkpoint_subfolder = reference_model_args.pop(
            "peft_checkpoint_subfolder", None
        )
        peft_subfolder = reference_model_args.pop("peft_subfolder", None)
        if peft_checkpoint_subfolder is None:
            peft_checkpoint_subfolder = peft_subfolder

        load_in_8bit = reference_model_args.pop("load_in_8bit", False)
        load_in_4bit = reference_model_args.pop("load_in_4bit", False)
        torch_dtype = _resolve_torch_dtype(reference_model_args.pop("torch_dtype", None))

        model_kwargs = {
            **reference_model_args,
            "cache_dir": hf_home,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if load_in_8bit or load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
            if "device_map" not in model_kwargs and torch.cuda.is_available():
                model_kwargs["device_map"] = {"": self.accelerator.local_process_index}

        ref_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if peft_name is not None:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "reference_model_args.peft_name was set, but `peft` is not installed."
                ) from e
            peft_kwargs = {"cache_dir": hf_home}
            if peft_checkpoint_subfolder is not None:
                peft_kwargs["subfolder"] = peft_checkpoint_subfolder
            ref_model = PeftModel.from_pretrained(
                ref_model,
                peft_name,
                is_trainable=False,
                **peft_kwargs,
            )

        if not (load_in_8bit or load_in_4bit):
            ref_model = ref_model.to(self.accelerator.device)

        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def _model_inputs(self, inputs):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def compute_forget_loss(self, model, forget_inputs):
        forget_outputs = model(**forget_inputs)
        return -forget_outputs.loss, forget_outputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = 0.0
        outputs = None

        if "forget" in inputs:
            forget_inputs = self._model_inputs(inputs["forget"])
            forget_loss, outputs = self.compute_forget_loss(
                model=model, forget_inputs=forget_inputs
            )
            loss = loss + self.gamma * forget_loss

        if "retain" in inputs:
            retain_inputs = self._model_inputs(inputs["retain"])
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )
            loss = loss + self.alpha * retain_loss

        if outputs is None:
            outputs = model(**retain_inputs)

        return (loss, outputs) if return_outputs else loss
