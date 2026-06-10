import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset


class CodeSecretDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        code_key="code",
        secret_key="secret_suffix",
        max_length=512,
    ):
        super().__init__()
        self.data = load_hf_dataset(**hf_args)
        self.tokenizer = tokenizer
        self.code_key = code_key
        self.secret_key = secret_key
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _iter_secrets(self, secrets):
        if secrets is None:
            return []
        if isinstance(secrets, str):
            return [secrets]
        return [secret for secret in secrets if secret]

    def _secret_spans(self, code, secrets):
        spans = []
        for secret in self._iter_secrets(secrets):
            start = 0
            while True:
                start = code.find(secret, start)
                if start == -1:
                    break
                end = start + len(secret)
                spans.append((start, end))
                start = end
        return spans

    def _mask_from_offsets(self, offsets, spans):
        secret_mask = torch.zeros(len(offsets), dtype=torch.bool)
        for token_idx, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue
            for span_start, span_end in spans:
                if token_start < span_end and token_end > span_start:
                    secret_mask[token_idx] = True
                    break
        return secret_mask

    def _mask_from_token_search(self, input_ids, secrets):
        secret_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        for secret in self._iter_secrets(secrets):
            secret_ids = self.tokenizer(
                secret, add_special_tokens=False
            )["input_ids"]
            if not secret_ids:
                continue
            for start in range(len(input_ids) - len(secret_ids) + 1):
                if input_ids[start : start + len(secret_ids)] == secret_ids:
                    secret_mask[start : start + len(secret_ids)] = True
        return secret_mask

    def __getitem__(self, idx):
        sample = self.data[idx]
        code = sample[self.code_key]
        secrets = sample.get(self.secret_key)

        try:
            encoded = self.tokenizer(
                code,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
            )
            input_ids = encoded["input_ids"]
            spans = self._secret_spans(code, secrets)
            secret_mask = self._mask_from_offsets(encoded["offset_mapping"], spans)
        except NotImplementedError:
            encoded = self.tokenizer(
                code,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            input_ids = encoded["input_ids"]
            secret_mask = self._mask_from_token_search(input_ids, secrets)

        input_ids = torch.tensor(input_ids)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "secret_mask": secret_mask,
        }
