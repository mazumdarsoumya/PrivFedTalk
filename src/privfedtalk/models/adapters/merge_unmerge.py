# Optional merge/unmerge utilities for LoRA.
import torch
import torch.nn as nn
from privfedtalk.models.adapters.lora import LoRALinear

def merge_lora(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            delta = (m.B @ m.A) * m.scaling
            with torch.no_grad():
                m.base.weight += delta
    return module

def unmerge_lora(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            delta = (m.B @ m.A) * m.scaling
            with torch.no_grad():
                m.base.weight -= delta
    return module
