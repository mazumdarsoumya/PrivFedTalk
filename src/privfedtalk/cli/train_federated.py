#/home/vineet/PycharmProjects/PrivFedTalk/src/privfedtalk/cli/train_federated.py

from __future__ import annotations

import argparse
from privfedtalk.utils.config import load_config
from privfedtalk.fl.server.orchestrator import run_federated_training
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from privfedtalk.data.datasets import build_manifest, get_dataset_and_manifest
from privfedtalk.fl.client.local_objective import LocalObjective
from privfedtalk.fl.privacy.dp import clip_and_noise_adapter_delta, adapter_delta_norm
from privfedtalk.fl.server.isfa import compute_client_score, aggregate_client_deltas
from privfedtalk.fl.utils.adapter_state import (
    get_adapter_state,
    load_adapter_state,
    subtract_adapter_states,
    add_adapter_delta,
    move_state_to,
)
from privfedtalk.models.build_model import build_model
from privfedtalk.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DatasetSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.indices[idx]]


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device, persistent_workers: bool = True, prefetch_factor: int = 2) -> DataLoader:
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



def read_manifest_rows(manifest_path: str) -> List[Dict[str, str]]:
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

    # JSON manifest support
    if txt.startswith("{") or txt.startswith("["):
        try:
            obj = json.loads(txt)
            rows = walk(obj, None)
            return rows
        except Exception:
            pass

    # CSV manifest support
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

    # --------------------------------------------------------
    # Primary path: use manifest rows if available
    # --------------------------------------------------------
    split_rows = [row for row in rows if row.get("split") == split] if rows else []
    if split_rows:
        for ds_idx, row in enumerate(split_rows):
            cid = infer_client_id(row)
            client_to_indices.setdefault(cid, []).append(ds_idx)

    # --------------------------------------------------------
    # Fallback path: use dataset items directly
    # --------------------------------------------------------
    if not client_to_indices:
        source_items = getattr(base_ds, "items", None)

        # sometimes dataset stores metadata under a different field
        if source_items is None:
            for name in ("samples", "rows", "records", "metadata"):
                source_items = getattr(base_ds, name, None)
                if source_items is not None:
                    break

        if source_items is not None:
            for ds_idx, item in enumerate(source_items):
                if isinstance(item, dict):
                    cid = infer_client_id(item)
                else:
                    cid = "client_0"
                client_to_indices.setdefault(cid, []).append(ds_idx)

    client_to_ds = {}
    for cid, idxs in client_to_indices.items():
        if len(idxs) >= min_client_samples:
            client_to_ds[cid] = DatasetSubset(base_ds, idxs)

    return base_ds, client_to_ds



def load_ckpt(model, ckpt_path: str) -> None:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=False)


def maybe_freeze_for_finetune(model) -> None:
    if hasattr(model, "vae"):
        for p in model.vae.parameters():
            p.requires_grad_(False)


def run_loader_epoch(model, loader, objective, optimizer, device, use_amp: bool, train_mode: bool, epoch: int, stage: str, prox_ref=None, prox_mu: float = 0.0):
    model.train(train_mode)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    for batch in loader:
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                loss, acc, _ = objective(model, batch, epoch=epoch, stage=stage)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().item())
        total_acc += float(acc.detach().item())
        n_steps += 1

    if n_steps == 0:
        return 0.0, 0.0

    return total_loss / n_steps, total_acc / n_steps


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

    id_sims = []
    temp_stabs = []
    losses = []
    accs = []

    for bi, batch in enumerate(loader):
        loss, acc, stats = objective(model, batch, epoch=1, stage=stage)
        id_sims.append(float(stats.get("identity_sim", 0.0)))
        temp_stabs.append(float(stats.get("temporal_stability", 0.0)))
        losses.append(float(loss.detach().item()))
        accs.append(float(acc.detach().item()))
        if (bi + 1) >= max_batches:
            break

    if len(id_sims) == 0:
        return {
            "identity_sim": 0.0,
            "temporal_stability": 0.0,
            "eval_loss": 0.0,
            "eval_acc": 0.0,
        }

    return {
        "identity_sim": float(sum(id_sims) / len(id_sims)),
        "temporal_stability": float(sum(temp_stabs) / len(temp_stabs)),
        "eval_loss": float(sum(losses) / len(losses)),
        "eval_acc": float(sum(accs) / len(accs)),
    }


def local_client_update(
    cfg,
    stage: str,
    client_id: str,
    client_ds,
    global_model_ckpt_state: Dict[str, torch.Tensor],
    global_adapter_state_cpu: Dict[str, torch.Tensor],
    device: torch.device,
    round_idx: int,
):
    local_model = build_model(cfg).to(device)
    local_model.load_state_dict(global_model_ckpt_state, strict=False)

    if stage == "finetune":
        maybe_freeze_for_finetune(local_model)

    federated_cfg = cfg.get("federated", {})
    local_epochs = int(federated_cfg.get("local_epochs", 1))
    local_lr = float(federated_cfg.get("local_lr", cfg["train"].get("lr", 2e-4)))
    local_wd = float(federated_cfg.get("local_weight_decay", cfg["train"].get("weight_decay", 1e-4)))
    use_amp = bool(cfg["train"].get("amp", True))
    batch_size = int(federated_cfg.get("local_batch_size", cfg["data"].get("batch_size", 8)))
    num_workers = int(federated_cfg.get("local_num_workers", 0))
    persistent_workers = bool(federated_cfg.get("local_persistent_workers", False))
    prefetch_factor = int(federated_cfg.get("local_prefetch_factor", 2))

    loader = make_loader(
        client_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    params = [p for p in local_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=local_lr, weight_decay=local_wd)
    objective = LocalObjective(cfg, device)

    use_cosine = bool(federated_cfg.get("use_cosine_local_lr", False))
    min_lr = float(federated_cfg.get("local_min_lr", local_lr * 0.1))
    scheduler = None
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(local_epochs, 1),
            eta_min=min_lr,
        )

    prox_mu = float(federated_cfg.get("prox_mu", 0.0))
    prox_ref = {
        name: p.detach().clone()
        for name, p in local_model.named_parameters()
        if p.requires_grad
    }

    last_train_loss = 0.0
    last_train_acc = 0.0

    for ep in range(1, local_epochs + 1):
        last_train_loss, last_train_acc = run_loader_epoch(
            model=local_model,
            loader=loader,
            objective=objective,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            train_mode=True,
            epoch=ep,
            stage=stage,
            prox_ref=prox_ref,
            prox_mu=prox_mu,
        )
        if scheduler is not None:
            scheduler.step()

    # Client quality factors for ISFA
    factors = estimate_client_factors(
        model=local_model,
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

    # Adapter delta
    local_adapter_gpu = get_adapter_state(local_model, trainable_only=True)
    global_adapter_gpu = move_state_to(global_adapter_state_cpu, device)
    delta_gpu = subtract_adapter_states(local_adapter_gpu, global_adapter_gpu)

    clip_norm = float(federated_cfg.get("dp_clip_norm", 1.0))
    noise_multiplier = float(federated_cfg.get("dp_noise_multiplier", 0.0))
    delta_priv_gpu = clip_and_noise_adapter_delta(
        delta=delta_gpu,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
    )

    delta_priv_cpu = move_state_to(delta_priv_gpu, "cpu")

    pkg = {
        "client_id": client_id,
        "n_samples": len(client_ds),
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "identity_sim": factors["identity_sim"],
        "temporal_stability": factors["temporal_stability"],
        "score": score,
        "delta_norm": adapter_delta_norm(delta_priv_cpu),
        "delta": delta_priv_cpu,
    }

    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pkg


@torch.no_grad()
def evaluate_global_model(cfg, stage: str, model, val_ds, device):
    eval_cfg = cfg.get("federated", {})
    batch_size = int(eval_cfg.get("eval_batch_size", cfg["data"].get("batch_size", 8)))
    num_workers = int(eval_cfg.get("eval_num_workers", 0))
    max_batches = int(eval_cfg.get("eval_max_batches", 8))
    use_amp = bool(cfg["train"].get("amp", True))

    loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        persistent_workers=False,
        prefetch_factor=2,
    )

    objective = LocalObjective(cfg, device)
    model.eval()

    losses = []
    accs = []

    for bi, batch in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            loss, acc, _ = objective(model, batch, epoch=1, stage=stage)
        losses.append(float(loss.detach().item()))
        accs.append(float(acc.detach().item()))
        if (bi + 1) >= max_batches:
            break

    if len(losses) == 0:
        return 0.0, 0.0

    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs))


def sample_clients(client_ids: List[str], clients_per_round: int, rng: random.Random) -> List[str]:
    if clients_per_round >= len(client_ids):
        return list(client_ids)
    return rng.sample(client_ids, clients_per_round)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--stage", required=True, choices=["pretrain", "finetune"])
    parser.add_argument("--init_ckpt", default=None, type=str)
    parser.add_argument("--agg", default=None, choices=["fedavg", "isfa"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1234)))

    device = choose_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    fed_out_dir = Path(cfg.get("federated", {}).get("output_dir", str(out_dir / "federated")))
    ckpt_dir = fed_out_dir / "checkpoints" / "federated"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    federated_cfg = cfg.setdefault("federated", {})
    agg_mode = args.agg or federated_cfg.get("agg", "isfa")

    manifest_path = cfg["data"].get("split_csv")
    if not manifest_path:
        manifest_path = str(out_dir / "manifests" / f'{cfg["data"]["name"]}_split.csv')

    build_manifest(cfg, manifest_path)

    min_client_samples = int(federated_cfg.get("min_client_samples", 1))
    _, train_clients = build_client_subsets(cfg, "train", manifest_path, min_client_samples=min_client_samples)
    val_ds = get_dataset_and_manifest(cfg, "val", manifest_path)

    client_ids = sorted(train_clients.keys())
    rounds = int(federated_cfg.get("rounds", 10))
    patience = int(federated_cfg.get("patience", 8))
    clients_per_round = int(federated_cfg.get("clients_per_round", max(1, min(5, len(client_ids)))))
    gamma = float(federated_cfg.get("isfa_gamma", 1.0))
    server_lr = float(federated_cfg.get("server_lr", 1.0))

    print(f"dataset       : {cfg['data']['name']}")
    print(f"root          : {cfg['data']['root']}")
    print(f"manifest      : {manifest_path}")
    print(f"num_clients   : {len(client_ids)}")
    print(f"clients/round : {clients_per_round}")
    print(f"rounds        : {rounds}")
    print(f"agg           : {agg_mode}")
    print(f"device        : {device}")
    if torch.cuda.is_available():
        print(f"gpu count     : {torch.cuda.device_count()}")
        print(f"gpu name      : {torch.cuda.get_device_name(0)}")

    # Save client summary
    client_summary = {cid: len(ds) for cid, ds in train_clients.items()}
    (fed_out_dir / "client_summary.json").write_text(json.dumps(client_summary, indent=2))

    # Build global model
    global_model = build_model(cfg).to(device)
    if args.init_ckpt:
        load_ckpt(global_model, args.init_ckpt)

    if args.stage == "finetune":
        maybe_freeze_for_finetune(global_model)

    history = []
    best_val = float("inf")
    best_round = 0
    bad_rounds = 0
    rng = random.Random(int(cfg.get("seed", 1234)))

    for rnd in range(1, rounds + 1):
        t0 = time.time()

        selected = sample_clients(client_ids, clients_per_round, rng)

        # Snapshot global model and adapter
        global_model_ckpt_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        global_adapter_state_cpu = move_state_to(get_adapter_state(global_model, trainable_only=True), "cpu")

        client_pkgs = []
        for cid in selected:
            pkg = local_client_update(
                cfg=cfg,
                stage=args.stage,
                client_id=cid,
                client_ds=train_clients[cid],
                global_model_ckpt_state=global_model_ckpt_state,
                global_adapter_state_cpu=global_adapter_state_cpu,
                device=device,
                round_idx=rnd,
            )
            client_pkgs.append(pkg)

        deltas = [pkg["delta"] for pkg in client_pkgs]
        client_infos = [
            {
                "client_id": pkg["client_id"],
                "n_samples": pkg["n_samples"],
                "identity_sim": pkg["identity_sim"],
                "temporal_stability": pkg["temporal_stability"],
                "score": pkg["score"],
            }
            for pkg in client_pkgs
        ]

        agg_delta_cpu, weights = aggregate_client_deltas(
            deltas=deltas,
            client_infos=client_infos,
            mode=agg_mode,
            gamma=gamma,
        )

        new_adapter_cpu = add_adapter_delta(global_adapter_state_cpu, agg_delta_cpu, scale=server_lr)
        load_adapter_state(global_model, new_adapter_cpu)

        val_loss, val_acc = evaluate_global_model(
            cfg=cfg,
            stage=args.stage,
            model=global_model,
            val_ds=val_ds,
            device=device,
        )

        elapsed = time.time() - t0

        round_info = {
            "round": rnd,
            "time_sec": elapsed,
            "selected_clients": selected,
            "weights": {pkg["client_id"]: float(w) for pkg, w in zip(client_pkgs, weights)},
            "client_stats": [
                {
                    "client_id": pkg["client_id"],
                    "n_samples": pkg["n_samples"],
                    "train_loss": pkg["train_loss"],
                    "train_acc": pkg["train_acc"],
                    "identity_sim": pkg["identity_sim"],
                    "temporal_stability": pkg["temporal_stability"],
                    "score": pkg["score"],
                    "delta_norm": pkg["delta_norm"],
                }
                for pkg in client_pkgs
            ],
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(round_info)

        print(
            f"Round {rnd:03d}/{rounds} | time={elapsed:.2f}s | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.6f} | "
            f"clients={len(selected)}",
            flush=True,
        )

        round_ckpt = ckpt_dir / f"round_{rnd:03d}.pt"
        torch.save(
            {
                "model": global_model.state_dict(),
                "round": rnd,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "cfg": cfg,
                "history_tail": history[-5:],
            },
            round_ckpt,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_round = rnd
            bad_rounds = 0
            torch.save(
                {
                    "model": global_model.state_dict(),
                    "round": rnd,
                    "best_val_loss": best_val,
                    "cfg": cfg,
                },
                ckpt_dir / "best_federated.pt",
            )
        else:
            bad_rounds += 1

        (fed_out_dir / "federated_history.json").write_text(json.dumps(history, indent=2))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if patience > 0 and bad_rounds >= patience:
            print(
                f"Early stopping at round {rnd:03d}; "
                f"best_round={best_round:03d}; best_val_loss={best_val:.6f}",
                flush=True,
            )
            break

    print(f"Best round: {best_round:03d} | best_val_loss={best_val:.6f}", flush=True)
    print(f"Best checkpoint: {ckpt_dir / 'best_federated.pt'}", flush=True)


if __name__ == "__main__":
    main()
