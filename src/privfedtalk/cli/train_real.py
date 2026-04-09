#/home/vineet/PycharmProjects/PrivFedTalk/src/privfedtalk/cli/train_real.py

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from privfedtalk.data.datasets import build_manifest, get_dataset_and_manifest
from privfedtalk.fl.client.local_objective import LocalObjective
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


def make_loader(ds, cfg, split: str, device: torch.device) -> DataLoader:
    workers = int(cfg["data"].get("num_workers", 4))
    kwargs = {
        "dataset": ds,
        "batch_size": int(cfg["data"].get("batch_size", 8)),
        "shuffle": split == "train",
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": split == "train",
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def load_ckpt(model, ckpt_path: str) -> None:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=False)


def maybe_freeze_for_finetune(model) -> None:
    if hasattr(model, "vae"):
        for p in model.vae.parameters():
            p.requires_grad_(False)


def run_one_epoch(model, loader, objective, optimizer, device, use_amp, train_mode, epoch, stage):
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
                grad_clip = float(objective.cfg["train"].get("grad_clip", 1.0))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().item())
        total_acc += float(acc.detach().item())
        n_steps += 1

    if n_steps == 0:
        return 0.0, 0.0
    return total_loss / n_steps, total_acc / n_steps


def train_stage(cfg, stage: str, init_ckpt: str | None = None) -> str:
    device = choose_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints" / "central"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cfg["data"].get("split_csv")
    if not manifest_path:
        manifest_path = str(out_dir / "manifests" / f'{cfg["data"]["name"]}_split.csv')

    rows = build_manifest(cfg, manifest_path)
    train_ds = get_dataset_and_manifest(cfg, "train", manifest_path)
    val_ds = get_dataset_and_manifest(cfg, "val", manifest_path)
    test_ds = get_dataset_and_manifest(cfg, "test", manifest_path)

    print(f"dataset     : {cfg['data']['name']}")
    print(f"root        : {cfg['data']['root']}")
    print(f"cache_root  : {cfg['data'].get('cache_root', 'NA')}")
    print(f"frames      : {cfg['data']['video_frames']}")
    print(f"video_size  : {cfg['data']['video_size']}")
    print(f"batch_size  : {cfg['data']['batch_size']}")
    print(f"workers     : {cfg['data'].get('num_workers', 0)}")
    print(f"cuda        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu count   : {torch.cuda.device_count()}")
        print(f"gpu name    : {torch.cuda.get_device_name(0)}")
    print(f"manifest    : {manifest_path}")
    if isinstance(rows, dict):
        print(f"rows        : {rows}")
    elif isinstance(rows, (list, tuple)):
        print(f"rows        : {len(rows)}")
    else:
        print(f"rows        : {rows}")
    print(f"train       : {len(train_ds)}")
    print(f"val         : {len(val_ds)}")
    print(f"test        : {len(test_ds)}")

    train_loader = make_loader(train_ds, cfg, "train", device)
    val_loader = make_loader(val_ds, cfg, "val", device)

    model = build_model(cfg).to(device)
    if init_ckpt:
        load_ckpt(model, init_ckpt)

    if stage == "finetune":
        maybe_freeze_for_finetune(model)

    lr = float(cfg["train"].get(f"{stage}_lr", cfg["train"].get("lr", 2e-4)))
    epochs = int(cfg["train"].get(f"{stage}_epochs", cfg["train"].get("epochs", 100)))
    patience = int(cfg["train"].get("patience", 8))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    use_amp = bool(cfg["train"].get("amp", True))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    objective = LocalObjective(cfg, device)

    best_val = float("inf")
    best_epoch = 0
    best_path = ckpt_dir / f"{stage}_best.pt"
    history = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            objective=objective,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            train_mode=True,
            epoch=epoch,
            stage=stage,
        )

        with torch.no_grad():
            val_loss, val_acc = run_one_epoch(
                model=model,
                loader=val_loader,
                objective=objective,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                train_mode=False,
                epoch=epoch,
                stage=stage,
            )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | time={elapsed:.2f}s | "
            f"val_loss={val_loss:.6f} | train_loss={train_loss:.6f} | "
            f"val_acc={val_acc:.6f} | train_acc={train_acc:.6f}",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch,
                "time_sec": elapsed,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "stage": stage,
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "cfg": cfg,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bad_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch:03d}; "
                f"best_epoch={best_epoch:03d}; best_val_loss={best_val:.6f}",
                flush=True,
            )
            break

    hist_path = out_dir / f"{stage}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Best checkpoint: {best_path}", flush=True)
    return str(best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--stage", required=True, choices=["pretrain", "finetune"])
    parser.add_argument("--init_ckpt", default=None, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    train_stage(cfg, stage=args.stage, init_ckpt=args.init_ckpt)


if __name__ == "__main__":
    main()
