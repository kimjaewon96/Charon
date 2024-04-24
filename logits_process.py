import math

import torch
from transformers import LogitsWarper
from transformers.generation.logits_process import LogitsProcessor


class MinPLogitsWarper(LogitsWarper):
    def __init__(
        self,
        min_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if min_p < 0 or min_p > 1.0:
            raise ValueError(f"`min_p` has to be a float >= 0 and <= 1, but is {min_p}")
        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(scores, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(
            tokens_to_remove, dim=-1, index=sorted_indices
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(
        self,
        tfs: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(
        self,
        top_a: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(
        self,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if mirostat_mode not in [2]:
            raise ValueError(
                f"`mirostat` has to be a an integer 2, but is {mirostat_mode}"
            )
        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if i == 0:
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0).to(device=input_ids.device)
        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True).to(
            device=input_ids.device
        )

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(
            1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0)
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class RepetitionPenaltyLogitsProcessorWithRange(LogitsProcessor):
    """
    Copied from the transformers library
    """

    def __init__(
        self,
        penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        _range: int,
    ):
        if not (penalty > 0):
            raise ValueError(f"`penalty` has to be strictly positive, but is {penalty}")

        self.penalty = penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self._range = _range

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        input_ids = input_ids[:, -self._range :]

        # We loop here because torch.unique() needs to process each row separately in the
        # case that batch_size > 1.
        for input_ids_row, scores_row in zip(input_ids, scores):
            unique_ids, counts = torch.unique(input_ids_row, return_counts=True)
            score = torch.gather(scores_row, 0, unique_ids)

            # multiplicative repetition penalty
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * self.penalty, score / self.penalty)
            scores_row.scatter_(0, unique_ids, score)

            # presence_penalty and frequency_penalty
            raw_presence_penalty = (counts > 0).to(scores.dtype)
            raw_frequency_penalty = counts.to(scores.dtype)
            additive_penalty = (
                raw_presence_penalty * self.presence_penalty
                + raw_frequency_penalty * self.frequency_penalty
            )
            scores_row.scatter_add_(0, unique_ids, -additive_penalty)

        return scores
