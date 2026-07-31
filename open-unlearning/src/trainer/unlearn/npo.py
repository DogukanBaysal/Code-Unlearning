from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
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
            forget_inputs = inputs["forget"]
            forget_loss, outputs = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )
            loss = loss + self.gamma * forget_loss

        retain_outputs = None
        if "retain" in inputs and self.alpha != 0.0:
            retain_inputs = self._model_inputs(inputs["retain"])
            retain_loss, retain_outputs = self.compute_retain_loss(
                model=model,
                retain_inputs=retain_inputs,
                return_outputs=True,
            )
            loss = loss + self.alpha * retain_loss

        if outputs is None:
            outputs = retain_outputs
        if outputs is None:
            raise ValueError("NPO received a batch without an active loss term")
        return (loss, outputs) if return_outputs else loss


class NPOWithoutRetain(NPO):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        loss, outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        loss = self.gamma * loss
        return (loss, outputs) if return_outputs else loss
