import os, random, torch
from typing import Dict, Any, List
from privfedtalk.data.datamodule import DataModule
from privfedtalk.models.build_model import build_model
from privfedtalk.fl.client.client_trainer import run_local_training, extract_lora_state, load_lora_state
from privfedtalk.fl.protocol.messages import ClientUpdate
from privfedtalk.fl.server.aggregator_fedavg import aggregate_fedavg
from privfedtalk.fl.server.aggregator_fedprox import aggregate_fedprox
from privfedtalk.fl.server.aggregator_isfa import aggregate_isfa
from privfedtalk.fl.privacy.dp_clip_noise import clip_and_add_noise
from privfedtalk.fl.privacy.secure_aggregation import secure_mask_updates
from privfedtalk.utils.io import ensure_dir
from privfedtalk.utils.seed import set_seed
from privfedtalk.utils.dist import resolve_device
from privfedtalk.trainers.loggers import CSVLogger
from privfedtalk.utils.timers import Timer


def apply_lora_delta(global_lora: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    keys = set(global_lora.keys()).union(delta.keys())
    for k in keys:
        gv = global_lora.get(k, None)
        dv = delta.get(k, None)
        if gv is None and dv is not None:
            out[k] = dv.detach().clone()
        elif gv is not None and dv is None:
            out[k] = gv.detach().clone()
        else:
            out[k] = gv.detach().clone() + dv.detach().clone()
    return out


def select_clients(cfg: Dict[str, Any], round_idx: int) -> List[int]:
    K = cfg["data"]["num_clients"]
    frac = float(cfg["train"]["clients_per_round"])
    m = max(1, int(K * frac))
    rnd = random.Random(cfg["seed"] + 1000 + round_idx)
    return rnd.sample(list(range(K)), m)


def run_single_round(cfg: Dict[str, Any], round_idx: int):
    set_seed(cfg["seed"] + round_idx)
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = cfg.get("output_dir", "outputs")
    ckpt_dir = os.path.join(out_dir, "checkpoints", "server")
    ensure_dir(ckpt_dir)

    dm = DataModule(cfg); dm.setup()
    model = build_model(cfg).to(device); model.scheduler.to(device)

    global_lora = extract_lora_state(model)

    last_ckpt = os.path.join(ckpt_dir, "global_last.pt")
    if os.path.exists(last_ckpt):
        state = torch.load(last_ckpt, map_location="cpu")
        gl = state.get("global_lora", None)
        if gl:
            global_lora = {k: v.to(device) for k, v in gl.items()}
            load_lora_state(model, global_lora)

    client_ids = select_clients(cfg, round_idx)
    updates: List[ClientUpdate] = []

    for cid in client_ids:
        loader = dm.get_client_loader(cid, shuffle=True)
        load_lora_state(model, global_lora)

        delta, score, n = run_local_training(cfg, model, loader, cid)

        # Client-level DP on adapter deltas
        if cfg["privacy"]["enable_client_dp"]:
            dp = cfg["privacy"]["dp"]
            delta = clip_and_add_noise(
                delta,
                dp["clip_norm"],
                dp["noise_mult"],
                seed=dp["seed"] + cid + round_idx
            )

        updates.append(ClientUpdate(client_id=cid, num_samples=n, delta=delta, score=score))

    # Secure aggregation (simulation-grade): pairwise-canceling masks so aggregate remains correct.
    if cfg["privacy"]["enable_secure_agg"] and len(updates) > 1:
        ids = [u.client_id for u in updates]
        masked = secure_mask_updates([u.delta for u in updates], ids, round_seed=777 + round_idx)
        updates = [
            ClientUpdate(client_id=u.client_id, num_samples=u.num_samples, delta=md, score=u.score)
            for u, md in zip(updates, masked)
        ]

    agg_name = cfg["fl"]["aggregator"]
    if agg_name == "fedavg":
        agg = aggregate_fedavg(updates)
    elif agg_name == "fedprox":
        agg = aggregate_fedprox(updates, mu=float(cfg["fl"].get("fedprox_mu", 0.01)))
    else:
        agg = aggregate_isfa(updates, gamma=float(cfg["fl"]["isfa"]["gamma"]))

    global_lora = apply_lora_delta(global_lora, agg)
    load_lora_state(model, global_lora)

    torch.save({
        "round": round_idx,
        "global_lora": {k: v.detach().cpu() for k, v in global_lora.items()},
        "model": model.state_dict(),
        "client_ids": client_ids
    }, last_ckpt)

    return {
        "round": round_idx,
        "clients": client_ids,
        "avg_score": float(sum(u.score for u in updates) / max(1, len(updates)))
    }


def run_federated_training(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    out_dir = cfg.get("output_dir", "outputs")
    log_dir = os.path.join(out_dir, "logs", "csv")
    ensure_dir(log_dir)
    logger = CSVLogger(os.path.join(log_dir, "train_log.csv"))

    rounds = int(cfg["train"]["rounds"])
    save_every = int(cfg["train"]["save_every_rounds"])

    for r in range(rounds):
        tmr = Timer()
        info = run_single_round(cfg, r)
        info["time_sec"] = tmr.elapsed()
        logger.log(info)

        if (r + 1) % save_every == 0:
            ckpt_dir = os.path.join(out_dir, "checkpoints", "server")
            last = os.path.join(ckpt_dir, "global_last.pt")
            snap = os.path.join(ckpt_dir, f"global_round_{r+1:04d}.pt")
            if os.path.exists(last):
                import shutil
                shutil.copy2(last, snap)

    logger.close()
    print("Done. Logs at:", os.path.join(out_dir, "logs", "csv", "train_log.csv"))
