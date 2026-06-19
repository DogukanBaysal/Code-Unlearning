from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class HuggingFaceDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        peft_name: str = None,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: str = None,
        gguf_file: str = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map

        kwargs = {
            "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "gguf_file": gguf_file,
        }

        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.force_base_prompt = force_base_prompt

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {}
        if gguf_file is not None:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if peft_name is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading PEFT adapters requires the optional `peft` package. "
                    "Install it with `pip install peft`."
                ) from exc

            self.model = PeftModel.from_pretrained(self.model, peft_name)
        if device_map is None:
            self.model = self.model.to(self.device)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def _format_prompt(self, prompt: str) -> str:
        return (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )

    def _trim_outputs(self, outputs: List[str]) -> List[str]:
        trimmed = []
        for output in outputs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            trimmed.append(output[:min_index].replace("\t", "    "))
        return trimmed

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        return self.codegen_batch(
            [prompt], do_sample=do_sample, num_samples=num_samples
        )[0]

    @torch.inference_mode()
    def codegen_batch(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 1
    ) -> List[List[str]]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompts = [self._format_prompt(prompt) for prompt in prompts]
        input_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        )
        if self.device_map is None:
            input_tokens = input_tokens.to(self.device)
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        num_return_sequences = min(self.batch_size, num_samples)

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stop_strings=self.eos,
            tokenizer=self.tokenizer,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens["input_ids"].size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        gen_strs = self._trim_outputs(gen_strs)
        return [
            gen_strs[i : i + num_return_sequences]
            for i in range(0, len(gen_strs), num_return_sequences)
        ]
