import torch
from typing import Dict, Any, Tuple
from tqdm import tqdm
from privfedtalk.models.build_model import build_model
from privfedtalk.data.datamodule import DataModule
from privfedtalk.utils.seed import set_seed
from privfedtalk.utils.dist import resolve_device
from privfedtalk.fl.client.local_objective import LocalObjective
from privfedtalk.fl.client.score_signals import compute_identity_score, compute_temporal_stability

def extract_lora_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if (".A" in k or ".B" in k)}

def load_lora_state(model: torch.nn.Module, lora_sd: Dict[str, torch.Tensor]):
    sd = model.state_dict()
    for k, v in lora_sd.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
    model.load_state_dict(sd, strict=False)

def state_sub(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (a[k] - b[k]).detach() for k in a.keys() if k in b}

def run_local_training(cfg: Dict[str, Any], model: torch.nn.Module, loader, client_id: int) -> Tuple[Dict[str, torch.Tensor], float, int]:
    device = next(model.parameters()).device
    model.train()
    obj = LocalObjective(cfg, device=device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    lora_before = {k: v.clone().detach() for k, v in extract_lora_state(model).items()}
    steps = int(cfg["train"]["local_steps"])
    grad_clip = float(cfg["train"]["grad_clip"])

    it = iter(loader)
    last = None
    for _ in tqdm(range(steps), desc=f"Client {client_id}", leave=False):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = obj.compute(model, batch)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(opt)
        scaler.update()
        last = out

    if last is None:
        score = 0.5
        n_samples = 0
    else:
        id_score = compute_identity_score(last["id_ref"], last["id_hat"])
        tmp_score = compute_temporal_stability(last["video_hat"])
        alpha = cfg["fl"]["isfa"]["alpha"]
        score = alpha * id_score + (1 - alpha) * tmp_score
        n_samples = int(cfg["data"]["batch_size"] * steps)

    lora_after = extract_lora_state(model)
    delta = state_sub(lora_after, lora_before)
    return delta, float(score), n_samples

def run_single_client_local_training(cfg: Dict[str, Any], client_id: int = 0):
    set_seed(cfg["seed"] + client_id)
    device = resolve_device(cfg.get("device", "auto"))

    dm = DataModule(cfg); dm.setup()
    loader = dm.get_client_loader(client_id, shuffle=True)

    model = build_model(cfg).to(device)
    model.scheduler.to(device)

    delta, score, n = run_local_training(cfg, model, loader, client_id)
    print(f"Client {client_id}: delta_keys={len(delta)} score={score:.3f} samples={n}")
