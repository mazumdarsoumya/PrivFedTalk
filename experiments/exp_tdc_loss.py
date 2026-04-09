import torch
from privfedtalk.losses.tdc_loss import temporal_denoising_consistency

def test_tdc():
    eps = torch.randn(2,4,3,8,8)
    pred = torch.randn(2,4,3,8,8)
    loss = temporal_denoising_consistency(eps, pred)
    assert loss.dim() == 0
