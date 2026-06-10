from trainer.utils import compute_kl_divergence
from trainer.unlearn.grad_diff import GradDiff


class CodeEraser(GradDiff):
    def __init__(self, retain_gd_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_gd_alpha = retain_gd_alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _model_inputs(self, inputs):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if "retain_gd" not in inputs:
            raise ValueError(
                "CodeEraser requires a third dataset under data.retain_gd"
            )

        forget_inputs = self._model_inputs(inputs["forget"])
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = self._model_inputs(inputs["retain"])
        retain_kl_loss, _ = compute_kl_divergence(
            model, self.ref_model, retain_inputs
        )

        retain_gd_inputs = self._model_inputs(inputs["retain_gd"])
        retain_gd_outputs = model(**retain_gd_inputs)
        retain_gd_loss = retain_gd_outputs.loss

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_kl_loss
            + self.retain_gd_alpha * retain_gd_loss
        )

        return (loss, forget_outputs) if return_outputs else loss
