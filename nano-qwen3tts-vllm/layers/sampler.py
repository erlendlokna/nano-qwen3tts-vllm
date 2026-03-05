import torch
from torch import nn
from typing import Optional


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def apply_temperature(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.float().div_(temperatures.unsqueeze(dim=1))

    @torch.compile
    def _sample_compiled(self, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int = 50, top_p: float = 1.0):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        top_k_logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        probs = torch.softmax(top_k_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _sample_seeded(self, logits: torch.Tensor, temperatures: torch.Tensor, generator: torch.Generator, top_k: int = 50, top_p: float = 1.0):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        top_k_logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        probs = torch.softmax(top_k_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int = 50, top_p: float = 1.0, generator: Optional[torch.Generator] = None):
        if generator is not None:
            return self._sample_seeded(logits, temperatures, generator, top_k, top_p)
        return self._sample_compiled(logits, temperatures, top_k, top_p)
