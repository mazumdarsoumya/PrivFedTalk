from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from privfedtalk.utils.config import load_config
from privfedtalk.models.build_model import build_model
from privfedtalk.fl.client.local_objective import LocalObjective
from privfedtalk.data.datasets import build_manifest, get_dataset_and_manifest
from privfedtalk.fl.privacy.dp import clip_and_noise_adapter_delta, adapter_delta_norm
from privfedtalk.fl.privacy.secure_aggregation import secure_mask_updates
from privfedtalk.fl.server.isfa import compute_client_score, aggregate_client_deltas
from privfedtalk.utils.adapter_state import (
    load_adapter_state,
    get_adapter_state,
    load_adapter_state,
    move_state_to,
    subtract_adapter_states,
    add_adapter_delta,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DatasetSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]

def read_manifest_rows(manifest_path: str) -> List[Dict[str, str]]:
    import csv

    p = Path(manifest_path)
    txt = p.read_text(encoding="utf-8-sig").strip()
    if not txt:
        return []

    def looks_like_row(d: Dict[str, str]) -> bool:
        keys = set(d.keys())
        markers = {
            "video", "video_path", "path", "text", "clip_id",
            "group", "identity", "speaker", "client_id", "split"
        }
        return bool(keys & markers)

    def norm_split(name):
        if name is None:
            return None
        s = str(name).strip().lower()
        if s in {"valid", "validation"}:
            return "val"
        if s in {"train", "val", "test"}:
            return s
        return None

    def walk(obj, current_split=None):
        rows = []
        if isinstance(obj, dict):
            if looks_like_row(obj):
                row = dict(obj)
                if current_split is not None and "split" not in row:
                    row["split"] = current_split
                rows.append(row)
            for k, v in obj.items():
                next_split = current_split
                guessed = norm_split(k)
                if guessed is not None:
                    next_split = guessed
                rows.extend(walk(v, next_split))
        elif isinstance(obj, list):
            for item in obj:
                rows.extend(walk(item, current_split))
        return rows

    if txt.startswith("{") or txt.startswith("["):
        try:
            obj = json.loads(txt)
            return walk(obj, None)
        except Exception:
            pass

    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def infer_client_id(row: Dict[str, str]) -> str:
    for key in ("client_id", "identity", "speaker", "group"):
        val = row.get(key, "")
        if val:
            return str(val)

    video = row.get("video", "") or row.get("path", "") or row.get("video_path", "")
    if video:
        p = Path(video)
        if p.parent.name:
            return p.parent.name
        return p.stem.split("_")[0]

    clip_id = row.get("clip_id", "")
    if clip_id:
        return str(clip_id).split("_")[0]

    return "client_0"

def build_client_subsets(cfg, split: str, manifest_path: str, min_client_samples: int = 1):
    rows = read_manifest_rows(manifest_path)
    base_ds = get_dataset_and_manifest(cfg, split, manifest_path)

    client_to_indices: Dict[str, List[int]] = {}
    split_rows = [row for row in rows if row.get("split") == split] if rows else []

    if split_rows:
        for ds_idx, row in enumerate(split_rows):
            cid = infer_client_id(row)
            client_to_indices.setdefault(cid, []).append(ds_idx)

    if not client_to_indices:
        source_items = getattr(base_ds, "items", None)
        if source_items is None:
            for name in ("samples", "rows", "records", "metadata"):
                source_items = getattr(base_ds, name, None)
                if source_items is not None:
                    break
        if source_items is not None:
            for ds_idx, item in enumerate(source_items):
                cid = infer_client_id(item) if isinstance(item, dict) else "client_0"
                client_to_indices.setdefault(cid, []).append(ds_idx)

    client_to_ds = {}
    for cid, idxs in client_to_indices.items():
        if len(idxs) >= min_client_samples:
            client_to_ds[cid] = DatasetSubset(base_ds, idxs)
    return base_ds, client_to_ds

def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device,
                persistent_workers: bool = True, prefetch_factor: int = 2) -> DataLoader:
    kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)

def choose_amp_dtype(cfg: Dict[str, Any]) -> Optional[torch.dtype]:
    if not bool(cfg["train"].get("amp", True)):
        return None
    name = str(cfg["train"].get("amp_dtype", "bf16")).lower()
    if name == "fp16":
        return torch.float16
    return torch.bfloat16

def make_scaler(amp_dtype: Optional[torch.dtype]):
    enabled = amp_dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)

@torch.no_grad()
def estimate_client_factors(model, ds, cfg, device, stage: str, max_batches: int = 2):
    batch_size = int(cfg["federated"].get("factor_batch_size", min(4, int(cfg["data"].get("batch_size", 8)))))
    num_workers = int(cfg["federated"].get("factor_num_workers", 0))
    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        persistent_workers=False,
        prefetch_factor=2,
    )

    objective = LocalObjective(cfg, device)
    model.eval()
    id_sims, temp_stabs, losses = [], [], []

    for bi, batch in enumerate(loader):
        loss, _acc, stats = objective(model, batch, epoch=1, stage=stage)
        id_sims.append(float(stats.get("identity_sim", 0.0)))
        temp_stabs.append(float(stats.get("temporal_stability", 0.0)))
        losses.append(float(loss.detach().item()))
        if (bi + 1) >= max_batches:
            break

    if len(id_sims) == 0:
        return {"identity_sim": 0.0, "temporal_stability": 0.0, "eval_loss": 0.0}

    return {
        "identity_sim": float(sum(id_sims) / len(id_sims)),
        "temporal_stability": float(sum(temp_stabs) / len(temp_stabs)),
        "eval_loss": float(sum(losses) / len(losses)),
    }

def run_local_adapter_training(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    client_ds,
    device: torch.device,
    stage: str,
    global_adapter_state_cpu: Dict[str, torch.Tensor],
    round_idx: int,
    client_id: str,
):
    federated_cfg = cfg.get("federated", {})
    batch_size = int(federated_cfg.get("local_batch_size", cfg["data"].get("batch_size", 8)))
    num_workers = int(federated_cfg.get("local_num_workers", 4))
    persistent_workers = bool(federated_cfg.get("local_persistent_workers", True))
    prefetch_factor = int(federated_cfg.get("local_prefetch_factor", 2))
    local_epochs = int(federated_cfg.get("local_epochs", 1))
    local_lr = float(federated_cfg.get("local_lr", cfg["train"].get("lr", 2e-4)))
    local_wd = float(federated_cfg.get("local_weight_decay", cfg["train"].get("weight_decay", 1e-4)))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    amp_dtype = choose_amp_dtype(cfg)
    scaler = make_scaler(amp_dtype)

    load_adapter_state(model, move_state_to(global_adapter_state_cpu, device))

    loader = make_loader(
        client_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=local_lr, weight_decay=local_wd)
    objective = LocalObjective(cfg, device)
    model.train()

    last_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(local_epochs):
        for step, batch in enumerate(loader, start=1):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _acc, _stats = objective(model, batch, epoch=epoch + 1, stage=stage)
                loss = loss / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % accum_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            last_loss = float(loss.detach().item()) * accum_steps

    if any(p.grad is not None for p in params):
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    local_adapter_gpu = get_adapter_state(model, trainable_only=True)
    delta_gpu = subtract_adapter_states(local_adapter_gpu, move_state_to(global_adapter_state_cpu, device))

    clip_norm = float(federated_cfg.get("dp_clip_norm", 1.0))
    noise_multiplier = float(federated_cfg.get("dp_noise_multiplier", 0.0))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.get("seed", 1234)) + int(round_idx) * 100003 + hash(str(client_id)) % 10007)
    delta_priv_gpu = clip_and_noise_adapter_delta(
        delta=delta_gpu,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        generator=gen,
    )

    factors = estimate_client_factors(
        model=model,
        ds=client_ds,
        cfg=cfg,
        device=device,
        stage=stage,
        max_batches=int(federated_cfg.get("factor_max_batches", 2)),
    )

    alpha = float(federated_cfg.get("isfa_alpha", 0.5))
    score = compute_client_score(
        identity_sim=factors["identity_sim"],
        temporal_stability=factors["temporal_stability"],
        alpha=alpha,
    )

    return {
        "client_id": client_id,
        "n_samples": len(client_ds),
        "train_loss": last_loss,
        "identity_sim": factors["identity_sim"],
        "temporal_stability": factors["temporal_stability"],
        "score": float(score),
        "delta_norm": float(adapter_delta_norm(move_state_to(delta_priv_gpu, "cpu"))),
        "delta": move_state_to(delta_priv_gpu, "cpu"),
        "personalized_adapter": move_state_to(local_adapter_gpu, "cpu"),
    }

def build_worker_context(cfg: Dict[str, Any], manifest_path: str, stage: str, base_ckpt: str, gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = build_model(cfg).to(device)
    if base_ckpt:
        state = torch.load(base_ckpt, map_location="cpu")
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=False)

    min_client_samples = int(cfg.get("federated", {}).get("min_client_samples", 1))
    _, train_clients = build_client_subsets(cfg, "train", manifest_path, min_client_samples=min_client_samples)
    val_ds = get_dataset_and_manifest(cfg, "val", manifest_path)
    return device, model, train_clients, val_ds

def worker_main(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, config_path: str,
                manifest_path: str, stage: str, base_ckpt: str):
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 1234)) + gpu_id)
    device, model, train_clients, val_ds = build_worker_context(cfg, manifest_path, stage, base_ckpt, gpu_id)

    while True:
        task = task_queue.get()
        if task is None:
            break

        task_type = task["type"]
        if task_type == "train_client":
            client_id = task["client_id"]
            pkg = run_local_adapter_training(
                cfg=cfg,
                model=model,
                client_ds=train_clients[client_id],
                device=device,
                stage=stage,
                global_adapter_state_cpu=task["global_adapter_state_cpu"],
                round_idx=int(task["round_idx"]),
                client_id=str(client_id),
            )
            pkg["gpu_id"] = gpu_id
            result_queue.put(pkg)

        elif task_type == "eval_global":
            load_adapter_state(model, move_state_to(task["global_adapter_state_cpu"], device))
            factors = estimate_client_factors(
                model=model,
                ds=val_ds,
                cfg=cfg,
                device=device,
                stage=stage,
                max_batches=int(cfg.get("federated", {}).get("eval_max_batches", 8)),
            )
            result_queue.put({
                "type": "eval_result",
                "val_identity_sim": factors["identity_sim"],
                "val_temporal_stability": factors["temporal_stability"],
                "val_loss": factors["eval_loss"],
                "gpu_id": gpu_id,
            })
        else:
            result_queue.put({"type": "error", "msg": f"Unknown task type: {task_type}", "gpu_id": gpu_id})

def sample_clients(client_ids: List[str], clients_per_round: int, rng: random.Random) -> List[str]:
    if clients_per_round >= len(client_ids):
        return list(client_ids)
    return rng.sample(client_ids, clients_per_round)

def save_personalized_adapter(pkg: Dict[str, Any], out_dir: Path, round_idx: int) -> None:
    client_dir = out_dir / "personalized" / str(pkg["client_id"])
    client_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_idx,
        "client_id": pkg["client_id"],
        "adapter_state": pkg["personalized_adapter"],
        "score": pkg["score"],
    }
    torch.save(payload, client_dir / "latest.pt")

def run_server(config_path: str, stage: str, base_ckpt: str = "") -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 1234)))

    out_dir = Path(cfg.get("federated", {}).get("output_dir", cfg.get("output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / "federated_2gpu"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg["data"].get("split_csv")
    if not manifest_path:
        manifest_path = str(out_dir / "manifests" / f"{cfg['data']['name']}_split.csv")
    build_manifest(cfg, manifest_path)

    _, train_clients = build_client_subsets(
        cfg,
        split="train",
        manifest_path=manifest_path,
        min_client_samples=int(cfg.get("federated", {}).get("min_client_samples", 1)),
    )
    client_ids = sorted(train_clients.keys())
    if len(client_ids) == 0:
        raise RuntimeError("No train clients were found in the manifest.")

    global_model = build_model(cfg)
    if base_ckpt:
        state = torch.load(base_ckpt, map_location="cpu")
        state_dict = state.get("model", state)
        global_model.load_state_dict(state_dict, strict=False)
    global_adapter_state_cpu = move_state_to(get_adapter_state(global_model, trainable_only=True), "cpu")

    ctx = mp.get_context("spawn")
    num_gpus = int(cfg.get("runtime", {}).get("num_gpus", torch.cuda.device_count()))
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))
    task_queues = [ctx.Queue() for _ in range(num_gpus)]
    result_queue = ctx.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=worker_main,
            args=(gpu_id, task_queues[gpu_id], result_queue, config_path, manifest_path, stage, base_ckpt),
        )
        p.start()
        workers.append(p)

    rounds = int(cfg.get("federated", {}).get("rounds", 10))
    clients_per_round = int(cfg.get("federated", {}).get("clients_per_round", 2))
    gamma = float(cfg.get("federated", {}).get("isfa_gamma", 1.0))
    server_lr = float(cfg.get("federated", {}).get("server_lr", 1.0))
    enable_secure_agg = bool(cfg.get("privacy", {}).get("enable_secure_agg", False))
    rng = random.Random(int(cfg.get("seed", 1234)))
    history = []

    try:
        for round_idx in range(1, rounds + 1):
            start_t = time.time()
            selected = sample_clients(client_ids, clients_per_round, rng)

            results = []
            wave_size = len(task_queues)
            for wave_start in range(0, len(selected), wave_size):
                wave = selected[wave_start: wave_start + wave_size]
                for worker_slot, client_id in enumerate(wave):
                    task_queues[worker_slot].put({
                        "type": "train_client",
                        "client_id": client_id,
                        "global_adapter_state_cpu": global_adapter_state_cpu,
                        "round_idx": round_idx,
                    })
                for _ in wave:
                    results.append(result_queue.get())

            if enable_secure_agg and len(results) > 1:
                client_numeric_ids = list(range(len(results)))
                masked_deltas = secure_mask_updates(
                    [pkg["delta"] for pkg in results],
                    client_ids=client_numeric_ids,
                    round_seed=777 + round_idx,
                )
                for pkg, masked in zip(results, masked_deltas):
                    pkg["delta"] = masked

            deltas = [pkg["delta"] for pkg in results]
            infos = [
                {
                    "client_id": pkg["client_id"],
                    "n_samples": pkg["n_samples"],
                    "identity_sim": pkg["identity_sim"],
                    "temporal_stability": pkg["temporal_stability"],
                    "score": pkg["score"],
                }
                for pkg in results
            ]
            agg_mode = str(cfg.get("federated", {}).get("agg", "isfa")).lower()
            agg_delta_cpu, weights = aggregate_client_deltas(
                deltas=deltas,
                client_infos=infos,
                mode=agg_mode,
                gamma=gamma,
            )
            global_adapter_state_cpu = add_adapter_delta(global_adapter_state_cpu, agg_delta_cpu, scale=server_lr)

            for pkg in results:
                save_personalized_adapter(pkg, out_dir, round_idx)

            task_queues[0].put({
                "type": "eval_global",
                "global_adapter_state_cpu": global_adapter_state_cpu,
            })
            eval_result = result_queue.get()

            round_info = {
                "round": round_idx,
                "selected_clients": selected,
                "weights": {pkg["client_id"]: float(w) for pkg, w in zip(results, weights)},
                "client_stats": [
                    {
                        "client_id": pkg["client_id"],
                        "n_samples": pkg["n_samples"],
                        "train_loss": pkg["train_loss"],
                        "identity_sim": pkg["identity_sim"],
                        "temporal_stability": pkg["temporal_stability"],
                        "score": pkg["score"],
                        "delta_norm": pkg["delta_norm"],
                        "gpu_id": pkg["gpu_id"],
                    }
                    for pkg in results
                ],
                "val_loss": eval_result["val_loss"],
                "val_identity_sim": eval_result["val_identity_sim"],
                "val_temporal_stability": eval_result["val_temporal_stability"],
                "time_sec": time.time() - start_t,
            }
            history.append(round_info)

            torch.save(
                {
                    "round": round_idx,
                    "adapter_state": global_adapter_state_cpu,
                    "cfg": cfg,
                    "history_tail": history[-5:],
                },
                ckpt_dir / f"global_adapter_round_{round_idx:04d}.pt",
            )
            torch.save(
                {
                    "round": round_idx,
                    "adapter_state": global_adapter_state_cpu,
                    "cfg": cfg,
                    "history_tail": history[-5:],
                },
                ckpt_dir / "global_adapter_last.pt",
            )
            (out_dir / "federated_history_2gpu.json").write_text(json.dumps(history, indent=2))

            print(
                f"Round {round_idx:03d}/{rounds} | clients={len(selected)} | "
                f"val_loss={round_info['val_loss']:.6f} | "
                f"val_id={round_info['val_identity_sim']:.4f} | "
                f"val_temp={round_info['val_temporal_stability']:.4f} | "
                f"time={round_info['time_sec']:.2f}s",
                flush=True,
            )
    finally:
        for q in task_queues:
            q.put(None)
        for p in workers:
            p.join(timeout=20)

def parse_args():
    ap = argparse.ArgumentParser(description="2-GPU PrivFedTalk federated training")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--stage", default="finetune", choices=["pretrain", "finetune"])
    ap.add_argument("--base-ckpt", default="", help="Shared backbone checkpoint")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_server(config_path=args.config, stage=args.stage, base_ckpt=args.base_ckpt)
