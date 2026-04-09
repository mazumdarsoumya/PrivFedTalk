import torch.nn.functional as F
def diffusion_mse(eps, eps_pred):
    return F.mse_loss(eps_pred, eps)
