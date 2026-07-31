import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff


def calculate_loss(model_disprefered_logits, ground_truth_distribution, labels=None, attention_mask=None):
    model_disprefered_distribution = F.softmax(model_disprefered_logits, dim=-1)

    model_disprefered_distribution = model_disprefered_distribution[..., :-1, :].contiguous()

    cross_entropy_loss = -torch.sum(
        ground_truth_distribution * torch.log(model_disprefered_distribution + 1e-10),
        dim=-1,
    )

    if labels is not None:
        loss_mask = labels[..., 1:] != -100
        cross_entropy_loss = cross_entropy_loss * loss_mask.float()

        if attention_mask is not None:
            cross_entropy_loss = cross_entropy_loss * attention_mask[..., 1:].float()

        mean_cross_entropy_loss = (
            cross_entropy_loss.sum() / loss_mask.float().sum().clamp_min(1.0)
        )
    else:
        mean_cross_entropy_loss = torch.mean(cross_entropy_loss)

    return mean_cross_entropy_loss


def top_p_filtering(
    logits,
    top_p=0.9,
    filter_value=0.0,
    N=1,
    max_N=10,
    need_softmax=True,
):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    if max_N is None:
        max_N = probs.size(-1)

    max_N_mask = torch.arange(sorted_probs.size(-1), device=logits.device) >= max_N
    sorted_indices_to_remove |= max_N_mask

    remove_mask = torch.zeros_like(probs, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    filtered_logits = logits.masked_fill(remove_mask, filter_value)

    return filtered_logits


def get_output_distribution(
    logits,
    labels,
    top_p=0.8,
    alpha=0.0,
    temperature=0.8,
    N=1,
    max_N=10,
):
    probs = F.softmax(logits, dim=-1)

    with torch.no_grad():
        labels = labels[..., 1:]

        copied_logits = logits[..., :-1, :].clone()

        mask_start_pos = 1
        mask_start = torch.zeros_like(labels, dtype=torch.bool)
        mask_start[:, mask_start_pos:] = 1

        valid_label_mask = labels != -100
        safe_labels = labels.masked_fill(~valid_label_mask, 0).long()

        mask = F.one_hot(
            safe_labels,
            num_classes=copied_logits.size(-1),
        ) & mask_start.unsqueeze(-1)

        copied_logits = copied_logits.masked_fill(mask.bool(), -float("inf"))

        filtered_logit = top_p_filtering(
            copied_logits,
            top_p=top_p,
            N=N,
            max_N=max_N,
            filter_value=-float("inf"),
        )

        if temperature is None:
            scaled_logit = filtered_logit
        else:
            scaled_logit = filtered_logit / temperature

        ground_truth_probs = F.softmax(scaled_logit, dim=-1)

        one_hot = F.one_hot(
            safe_labels,
            num_classes=probs.size(-1),
        ).bool()

        ground_truth_probs = torch.where(
            one_hot,
            -alpha * probs[..., :-1, :],
            ground_truth_probs,
        )

        ground_truth_probs = ground_truth_probs * valid_label_mask.unsqueeze(-1)

    return probs, ground_truth_probs


class PROD(GradDiff):
    def __init__(
        self,
        top_p=0.8,
        temperature=0.8,
        N=1,
        max_N=10,
        prod_alpha=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.top_p = top_p
        self.temperature = temperature
        self.N = N
        self.max_N = max_N


        self.prod_alpha = prod_alpha

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss = 0.0
        outputs = None

        if "forget" in inputs:
            forget_inputs = self._model_inputs(inputs["forget"])
            input_ids = forget_inputs["input_ids"]
            attention_mask = forget_inputs["attention_mask"]
            labels = forget_inputs["labels"]

            with torch.no_grad():
                _, ground_truth_distribution = get_output_distribution(
                    self.ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits,
                    labels,
                    top_p=self.top_p,
                    alpha=self.prod_alpha,
                    temperature=self.temperature,
                    N=self.N,
                    max_N=self.max_N,
                )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            forget_loss = calculate_loss(
                outputs.logits,
                ground_truth_distribution,
                labels=labels,
                attention_mask=attention_mask,
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
            raise ValueError("PROD received a batch without an active loss term")
        return (loss, outputs) if return_outputs else loss


class PRODWithoutRetain(PROD):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_inputs = self._model_inputs(inputs["forget"])
        input_ids = forget_inputs["input_ids"]
        attention_mask = forget_inputs["attention_mask"]
        labels = forget_inputs["labels"]

        with torch.no_grad():
            _, ground_truth_distribution = get_output_distribution(
                self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits,
                labels,
                top_p=self.top_p,
                alpha=self.prod_alpha,
                temperature=self.temperature,
                N=self.N,
                max_N=self.max_N,
            )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = calculate_loss(
            outputs.logits,
            ground_truth_distribution,
            labels=labels,
            attention_mask=attention_mask,
        )
        loss = self.gamma * loss
        return (loss, outputs) if return_outputs else loss
