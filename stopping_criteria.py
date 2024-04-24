import torch
from transformers import StoppingCriteria, AutoTokenizer
from typing import Union, List


class StopStringsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_strings: Union[str, List[str]]):
        self.stop_strings = stop_strings
        self.stop_seq_list = None
        self.test = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        tokenizer: AutoTokenizer,
        **kwargs,
    ) -> bool:
        if not self.stop_seq_list:
            self.stop_seq_list = tokenizer(self.stop_strings).input_ids
            if all(isinstance(x, int) for x in self.stop_seq_list):
                self.stop_seq_list = [self.stop_seq_list]
            self.stop_seq_list = [x[1:] for x in self.stop_seq_list if len(x) > 1]
        is_done = False
        self.test = True
        for stop_seq in self.stop_seq_list:
            if (
                len(input_ids[0]) >= len(stop_seq)
                and input_ids[0, len(input_ids[0]) - len(stop_seq) :] == stop_seq
            ):
                is_done = True
                break
            elif len(input_ids[0]) >= len(stop_seq):
                print("<", input_ids[0, len(input_ids[0]) - len(stop_seq) :])
                print(">", stop_seq)
        return is_done


class EOSStoppingCriteria(StoppingCriteria):
    def __init__(self, bool: bool = True):
        self.switch = bool

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tokenizer
    ) -> bool:
        pass
