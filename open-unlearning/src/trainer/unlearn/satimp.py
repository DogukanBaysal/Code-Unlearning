from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_satimp_loss


class SatImp(GradDiff):
    def __init__(
        self, beta1=5.0, beta2=1.0, gamma=1.0, alpha=0.1, *args, **kwargs
    ):  # attention, satimp requires two beta!!!!
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.alpha = alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = 0.0
        outputs = None

        if "forget" in inputs:
            forget_inputs = self._model_inputs(inputs["forget"])
            forget_loss, outputs = compute_satimp_loss(
                model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
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
