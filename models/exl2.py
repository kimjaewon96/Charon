import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer


class ExllamaV2Model:
    def __init__(self, model_dir, **kwargs):
        config = ExLlamaV2Config(model_dir)
        # config.model_dir = model_dir
        # config.scale_pos_emb = kwargs.pop("scale_pos_emb", config.scale_pos_emb)
        # config.scale_alpha_value = kwargs.pop(
        #     "scale_alpha_value", config.scale_alpha_value
        # )
        # config.no_flash_attn = kwargs.pop("no_flash_attn", config.no_flash_attn)
        # config.num_experts_per_token = int(
        #     kwargs.pop("num_experts_per_token", config.num_experts_per_token)
        # )
        config.max_seq_len = kwargs.pop("max_seq_len", config.max_seq_len)
        # config.max_input_len = min(
        #     kwargs.pop("max_input_len", config.max_input_len), config.max_seq_len
        # )
        self.model = ExLlamaV2(config)
        split = None
        if "gpu_split" in kwargs.keys():
            split = [float(alloc) for alloc in kwargs["gpu_split"].split(",")]
        self.model.load(split)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.cache = ExLlamaV2Cache(self.model)
        self.past_seq = None
        self.device = torch.device(0)
        # self.max_input_len = config.max_input_len

    def update_cache(self, input_ids):
        reset = True
        seq_tensor = input_ids[0]
        if self.past_seq is not None:
            min_length = min(self.past_seq.shape[0], seq_tensor.shape[0])
            indices = torch.nonzero(
                ~torch.eq(self.past_seq[:min_length], seq_tensor[:min_length])
            )
            if len(indices) > 0:
                longest_prefix = indices[0].item()
            else:
                longest_prefix = min_length

            if longest_prefix > 0:
                reset = False
                self.cache.current_seq_len = longest_prefix
                if seq_tensor.shape[0] - longest_prefix > 1:
                    self.model.forward(
                        seq_tensor[longest_prefix:-1].view(1, -1),
                        self.cache,
                        preprocess_only=True,
                    )
                elif seq_tensor.shape[0] == longest_prefix:
                    self.cache.current_seq_len -= 1

        if reset:
            self.cache.current_seq_len = 0
            if seq_tensor.shape[0] > 1:
                self.model.forward(
                    seq_tensor[:-1].view(1, -1),
                    self.cache,
                    preprocess_only=True,
                )
        pass

    def forward(self, input_ids):
        logits = self.model.forward(input_ids[:, -1:], self.cache)
        self.past_seq = input_ids[0]
        return logits[..., -1, :]

    # def forward_label(self, input_ids):
    #     self.cache.current_seq_len = 0
    #     logits = self.model.forward(input_ids, self.cache, last_id_only=False)
    #     self.past_seq = input_ids[0]
    #     return logits[..., -1, :]

    def __call__(self, input_ids):
        return self.forward(input_ids)


if __name__ == "__main__":
    model = ExllamaV2Model(
        r"C:\CODE\text-generation-webui\models\LoneStriker_Mistral-7B-Instruct-v0.2-8.0bpw-h8-exl2-2"
    )
