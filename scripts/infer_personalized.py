from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from privfedtalk.utils.config import load_config
from privfedtalk.models.build_model import build_model
from privfedtalk.utils.adapter_state import load_adapter_state, move_state_to

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_ckpt(path: str):
    obj = torch.load(path, map_location="cpu")
    return obj

def maybe_load_full_model(model: torch.nn.Module, ckpt_obj: Dict[str, Any]) -> None:
    if not isinstance(ckpt_obj, dict):
        return

    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        model.load_state_dict(ckpt_obj["model"], strict=False)
        return

    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        model.load_state_dict(ckpt_obj["state_dict"], strict=False)
        return

def extract_adapter_state(ckpt_obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "adapter_state" in ckpt_obj and isinstance(ckpt_obj["adapter_state"], dict):
            return ckpt_obj["adapter_state"]
        # fallback: if the dict itself already looks like a state dict
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Could not find adapter_state in checkpoint.")

def parse_args():
    ap = argparse.ArgumentParser(description="Reconstruct personalized PrivFedTalk model")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--global-adapter-ckpt", required=True, help="Global adapter checkpoint, e.g. best round")
    ap.add_argument("--adapter-ckpt", required=True, help="Personalized client adapter checkpoint")
    ap.add_argument("--base-ckpt", default="", help="Optional full shared backbone checkpoint")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--out", required=True, help="Output .pt file")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 1234))
    set_seed(seed)

    device = torch.device(args.device)
    model = build_model(cfg)

    # Optional full backbone load if available.
    if args.base_ckpt:
        base_obj = load_ckpt(args.base_ckpt)
        maybe_load_full_model(model, base_obj)

    # Move base model to target device.
    model = model.to(device)

    # Load best global adapter first.
    global_obj = load_ckpt(args.global_adapter_ckpt)
    global_adapter = extract_adapter_state(global_obj)
    load_adapter_state(model, move_state_to(global_adapter, device))

    # Then load personalized adapter on top.
    personal_obj = load_ckpt(args.adapter_ckpt)
    personal_adapter = extract_adapter_state(personal_obj)
    load_adapter_state(model, move_state_to(personal_adapter, device))

    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config_path": args.config,
        "seed": seed,
        "device": str(device),
        "global_adapter_ckpt": args.global_adapter_ckpt,
        "personal_adapter_ckpt": args.adapter_ckpt,
        "base_ckpt": args.base_ckpt,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "status": "personalized_model_reconstructed"
    }
    torch.save(payload, out_path)

    print("Saved reconstructed personalized model to:", out_path)

    meta_path = out_path.with_suffix(".json")
    meta = {
        "config_path": args.config,
        "seed": seed,
        "device": str(device),
        "global_adapter_ckpt": args.global_adapter_ckpt,
        "personal_adapter_ckpt": args.adapter_ckpt,
        "base_ckpt": args.base_ckpt,
        "status": "personalized_model_reconstructed"
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved metadata to:", meta_path)

if __name__ == "__main__":
    main()
