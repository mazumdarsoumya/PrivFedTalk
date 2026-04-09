import torch.nn as nn
from typing import Dict, Any, List
from privfedtalk.models.adapters.lora import LoRALinear

def inject_lora(module: nn.Module, lora_cfg: Dict[str, Any]) -> int:
    targets: List[str] = list(lora_cfg.get("target_modules", []))
    r = int(lora_cfg.get("r", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.0))

    injected = 0
    for name in targets:
        if hasattr(module, name):
            base = getattr(module, name)
            if isinstance(base, nn.Linear):
                setattr(module, name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
                injected += 1
    for child in module.children():
        injected += inject_lora(child, lora_cfg)
    return injected
