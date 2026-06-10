import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
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
