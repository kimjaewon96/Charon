from pprint import pprint
from typing import List, Optional, Union

from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

from .generator import BaseGenerator
from .models.exl2 import ExllamaV2Model


class Charon:
    def __init__(self, model_path: str, model_loader: str = "exl2"):
        if model_loader == "exl2":
            self.model = ExllamaV2Model(model_path, max_seq_len=2048)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generator = BaseGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.chat_template = self.tokenizer.chat_template
        # self.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    def setup_generator(
        self,
        logits_processor_list: Optional[List[LogitsProcessor]] = None,
        max_new_tokens: int = 20,
        stop_strings: Union[str, List[str]] = None,
        ban_bos_token: bool = False,
        ban_eos_token: bool = False,
        skip_special_tokens: bool = True,
        streaming: bool = False,
        return_dict: bool = False,
    ):
        self.generator = BaseGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            logits_processor_list=logits_processor_list,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            ban_bos_token=ban_bos_token,
            ban_eos_token=ban_eos_token,
            skip_special_tokens=skip_special_tokens,
            streaming=streaming,
            return_dict=return_dict,
        )


class CharonChat(Charon):
    def __init__(self, model_path: str, model_loader: str = "exl2"):
        super().__init__(model_path, model_loader)
        self.chat_history = []

    def chat(self, message: str, prefill_response: str = "", **kwargs):
        if message == "":
            raise Exception("Message should not be empty!")
        new_chat_history = self.chat_history.copy()
        new_chat_history.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(
            new_chat_history,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt += prefill_response
        output = self.generator.generate(prompt, return_dict=False, **kwargs)
        new_chat_history.append(
            {"role": "assistant", "content": prefill_response + output}
        )
        self.chat_history = new_chat_history
        return prefill_response + output

    def clear_chat(self):
        self.chat_history = []


if __name__ == "__main__":
    gen = CharonChat(
        "D:/text-generation-webui/models/bartowski_Meta-Llama-3-8B-Instruct-exl2_6_5"
    )
    gen.chat("Hi, where is South Korea?")
    gen.chat("I love Kimchi.")
    pprint(gen.chat_history)
