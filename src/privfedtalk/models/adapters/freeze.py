import torch.nn as nn

def unfreeze_lora_only(module: nn.Module):
    """Freeze everything except LoRA parameters named *.A and *.B."""
    for name, p in module.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
