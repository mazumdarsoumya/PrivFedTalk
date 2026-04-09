import torch
import torch.nn.functional as F
def temporal_denoising_consistency(eps, eps_pred):
    if eps.shape[1] < 2:
        return torch.tensor(0.0, device=eps.device)
    de = eps[:, 1:] - eps[:, :-1]
    dp = eps_pred[:, 1:] - eps_pred[:, :-1]
    return F.l1_loss(dp, de)
