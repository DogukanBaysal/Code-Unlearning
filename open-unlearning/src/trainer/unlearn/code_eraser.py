import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


class CodeEraser(GradDiff):
    def __init__(
        self,
        select_gamma=0.5,
        control_alpha=1.0,
        control_lambda=0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.select_gamma = select_gamma
        self.control_alpha = control_alpha
        self.control_lambda = control_lambda
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def _model_inputs(self, inputs):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def _masked_mean(self, values, mask):
        if mask.any():
            return values[mask].mean()
        return values.sum() * 0.0

    def _token_ce_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        return loss.view(shift_labels.shape), shift_labels

    def _masked_kl_loss(self, logits, ref_logits, mask):
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous().float()

        if not mask.any():
            return shift_logits.sum() * 0.0

        shift_logits = shift_logits[mask]
        shift_ref_logits = shift_ref_logits[mask]

        probs = F.softmax(
            shift_logits - shift_logits.max(dim=-1, keepdim=True).values,
            dim=-1,
        )
        ref_probs = F.softmax(
            shift_ref_logits - shift_ref_logits.max(dim=-1, keepdim=True).values,
            dim=-1,
        )
        return F.kl_div(
            (probs + 1e-10).log(),
            ref_probs + 1e-10,
            reduction="batchmean",
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._model_inputs(inputs["forget"])
        if "secret_mask" not in inputs["forget"]:
            raise ValueError(
                "CodeEraser requires the forget dataset to provide secret_mask"
            )

        forget_outputs = model(**forget_inputs)
        token_loss, shift_labels = self._token_ce_loss(
            forget_outputs.logits,
            forget_inputs["labels"],
        )
        valid_mask = shift_labels.ne(-100)
        if "attention_mask" in forget_inputs:
            valid_mask = valid_mask & forget_inputs["attention_mask"][..., 1:].bool()

        secret_mask = inputs["forget"]["secret_mask"][..., 1:].bool() & valid_mask
        normal_mask = (~secret_mask) & valid_mask

        secret_loss = self._masked_mean(token_loss, secret_mask)
        normal_loss = self._masked_mean(token_loss, normal_mask)
        selective_loss = -secret_loss + self.select_gamma * normal_loss

        with torch.no_grad():
            forget_ref_outputs = self.ref_model(**forget_inputs)
        forget_secret_kl = self._masked_kl_loss(
            forget_outputs.logits,
            forget_ref_outputs.logits,
            secret_mask,
        )

        retain_inputs = self._model_inputs(inputs["retain"])
        retain_outputs = model(**retain_inputs)
        with torch.no_grad():
            retain_ref_outputs = self.ref_model(**retain_inputs)
        retain_mask = retain_inputs["attention_mask"][..., 1:].bool()
        retain_kl_loss = self._masked_kl_loss(
            retain_outputs.logits,
            retain_ref_outputs.logits,
            retain_mask,
        )

        loss = (
            selective_loss
            + self.control_lambda
            * (-forget_secret_kl + self.control_alpha * retain_kl_loss)
        )

        return (loss, forget_outputs) if return_outputs else loss
