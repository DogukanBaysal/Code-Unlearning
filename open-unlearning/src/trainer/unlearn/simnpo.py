import torch.nn.functional as F

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = 0.0
        outputs = None

        if "forget" in inputs:
            forget_inputs = inputs["forget"]
            forget_labels = forget_inputs["labels"]
            loss_mask = forget_labels != -100
            forget_loss, outputs = compute_batch_nll(model, forget_inputs)
            forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
            forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
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
