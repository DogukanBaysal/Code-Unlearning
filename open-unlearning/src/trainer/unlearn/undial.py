from trainer.utils import compute_undial_loss
from trainer.unlearn.grad_diff import GradDiff


class UNDIAL(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = 0.0
        outputs = None

        if "forget" in inputs:
            forget_inputs = self._model_inputs(inputs["forget"])
            forget_loss, outputs = compute_undial_loss(
                model, self.ref_model, forget_inputs, self.beta
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
