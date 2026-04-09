import torch
class GaussianNoiseSampler:
    def sample_like(self, x: torch.Tensor):
        return torch.randn_like(x)
