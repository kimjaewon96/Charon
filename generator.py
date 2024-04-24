import math
from typing import List, Optional, Union

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class BaseGenerator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        logits_processor_list: Optional[List[LogitsProcessor]] = None,
        max_new_tokens: int = 20,
        stop_strings: Union[str, List[str]] = None,
        ban_bos_token: bool = False,
        ban_eos_token: bool = False,
        skip_special_tokens: bool = True,
        streaming: bool = False,
        return_dict: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if logits_processor_list and len(logits_processor_list) > 0:
            self.logits_processor = LogitsProcessorList(logits_processor_list)
        else:
            self.logits_processor = None
        self.max_new_tokens = max_new_tokens
        self.stop_strings = stop_strings
        self.ban_bos_token = ban_bos_token
        self.ban_eos_token = ban_eos_token
        self.skip_special_tokens = skip_special_tokens
        self.streaming = streaming
        self.return_dict = return_dict

    def chat(
        self,
        messages,
        continuation: str = "",
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        return_dict = kwargs.get("return_dict", self.return_dict)
        prompt = (
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            + continuation
        )
        print(prompt)
        output = self.generate(prompt, **kwargs)
        if not return_dict:
            return continuation + output
        messages.append()
        output["messages"]
        return self.generate(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs):
        # settings
        logits_processor = kwargs.get("logits_processor", self.logits_processor)
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        stop_strings = kwargs.get("stop_strings", self.stop_strings)
        ban_bos_token = kwargs.get("ban_bos_token", self.ban_bos_token)
        ban_eos_token = kwargs.get("ban_eos_token", self.ban_eos_token)
        skip_special_tokens = kwargs.get(
            "skip_special_tokens", self.skip_special_tokens
        )
        streaming = kwargs.get("streaming", self.streaming)
        return_dict = kwargs.get("return_dict", self.return_dict)

        if streaming:
            return self.generate_stream(
                prompt=prompt,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                ban_bos_token=ban_bos_token,
                ban_eos_token=ban_eos_token,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            output_text = ""
            stop_reason = {}
            for output in self.generate_stream(
                prompt=prompt,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                ban_bos_token=ban_bos_token,
                ban_eos_token=ban_eos_token,
                skip_special_tokens=skip_special_tokens,
            ):
                if isinstance(output, str):
                    output_text += output
                elif isinstance(output, dict):
                    stop_reason = output
            if return_dict:
                return {
                    "text": output_text,
                    "full_text": prompt + output_text,
                    **stop_reason,
                }
            else:
                return output_text

    def generate_stream(
        self,
        prompt: str,
        logits_processor,
        max_new_tokens,
        stop_strings,
        ban_bos_token,
        ban_eos_token,
        skip_special_tokens,
    ):
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]

        if isinstance(stop_strings, list):
            stop_strings = [x for x in stop_strings]

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            device=self.model.device
        )
        # input_ids_seq_length = len(input_ids)

        # preprocess cache
        output_string = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=skip_special_tokens
        )
        cursor = len(output_string)
        held_string = ""
        generated_tokens = 0
        self.model.update_cache(input_ids)

        while True:
            output = self.generate_one(input_ids, logits_processor=logits_processor)

            input_ids = torch.cat([input_ids, output], dim=-1)
            output_string = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=skip_special_tokens
            )
            # print(output_string)
            held_string = output_string[cursor:]
            if len(held_string) > 0 and held_string[0] == " ":
                held_string = held_string[1:]

            if not ban_eos_token and output[0] == self.tokenizer.eos_token_id:
                yield {"reason": "EOS"}
                break

            if stop_strings and held_string in stop_strings:
                yield {"reason": "stop_string", "stop_string": held_string}
                break

            if not stop_strings or not any(
                [
                    text.startswith(held_string)
                    for text in stop_strings
                    if len(held_string) > 0
                ]
            ):
                yield output_string[cursor:]
                cursor = len(output_string)

            generated_tokens += 1
            if generated_tokens > max_new_tokens:
                print("warning: max new token exceeded!")
                yield {"reason": "max_new_tokens", "max_new_tokens": max_new_tokens}
                break

    def beam_search(
        self,
        prompt,
        num_beams,
        max_length: int = 10,
        logits_processor_list: Optional[List[LogitsProcessor]] = None,
    ):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            device=self.model.device
        )
        logits_processor = None
        if logits_processor_list and len(logits_processor_list) > 0:
            logits_processor = LogitsProcessorList(logits_processor_list)

        beams = [{"sequence": input_ids, "score": 0}]
        # first_try
        for b_idx in range(num_beams):
            output = self.generate_one(
                input_ids, logits_processor=logits_processor, return_prob=True
            )

    def generate_one(
        self, input_ids=None, logits_processor=None, return_prob: bool = False
    ):
        scores = self.model(input_ids)
        if logits_processor:
            scores = logits_processor(
                input_ids, scores, input_ids_seq_length=len(input_ids)
            )
            probs = torch.nn.functional.softmax(scores, dim=-1)
            output = torch.multinomial(probs, 1)
        else:
            output = torch.argmax(scores, dim=-1).unsqueeze(0)

        if not return_prob:
            return output
        elif logits_processor:
            target_prob = probs[:, output].item()
            return output, target_prob
        else:
            probs = torch.nn.functional.softmax(scores, dim=-1)
            target_prob = probs[:, output].item()
            return output, target_prob

    def top_tokens(self, prompt: str, top_tokens: int = 50, **kwargs):
        logits_processor = kwargs.get("logits_processor", self.logits_processor)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            device=self.model.device
        )
        input_ids_seq_length = len(input_ids)
        self.model.update_cache(input_ids)
        scores = self.model(input_ids)
        if logits_processor:
            scores = logits_processor(
                input_ids, scores, input_ids_seq_length=input_ids_seq_length
            )
        probs = torch.nn.functional.softmax(scores, dim=-1)
        sorted_logits, sorted_indices = torch.sort(probs, descending=True)
        sorted_logits = sorted_logits[:, :top_tokens]
        sorted_indices = sorted_indices[:, :top_tokens]
        text = self.tokenizer.batch_decode(sorted_indices.transpose(0, 1))
        token_ids = sorted_indices[0].tolist()
        logits = sorted_logits[0].tolist()
        return [
            {"text": a, "token_id": b, "prob": c}
            for a, b, c in zip(text, token_ids, logits)
            if c > 0
        ]

    def choice(
        self,
        prompt: str,
        choices: List[str] = [],
    ):
        assert len(choices) >= 2
        # create trie
        choice_seqs = self.tokenizer.batch_encode_plus(choices).input_ids
        choice_seqs = [x[1:] for x in choice_seqs]
        prompt = prompt.strip()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            device=self.model.device
        )
        scores = []
        for choice_seq in choice_seqs:
            score = self.calculate_choice(seq_in=input_ids, seq_out=choice_seq)
            scores.append(score)
            # print(choice_seq, score)
        high_score = max(scores)
        high_score_index = scores.index(high_score)
        return choices[high_score_index]

    def calculate_choice(self, seq_in, seq_out):
        logprobs = []
        self.model.update_cache(seq_in)
        for target in seq_out:
            scores = self.model(seq_in)
            # find score
            probs = torch.nn.functional.softmax(scores, dim=-1)
            target_prob = probs[:, target].item()
            if target_prob == 0:
                return -math.inf
            logprobs.append(math.log(target_prob))
            seq_in = torch.cat(
                (seq_in, torch.tensor([[target]]).to(device=self.model.device)), dim=1
            )
        return sum(logprobs) / len(logprobs)
