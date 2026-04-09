from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from privfedtalk.utils.config import load_config
from privfedtalk.data.datasets import get_dataset_and_manifest
from privfedtalk.models.build_model import build_model
from privfedtalk.fl.client.local_objective import LocalObjective
from privfedtalk.fl.utils.adapter_state import get_adapter_state


def load_ckpt(model, ckpt_path: str) -> None:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=False)


def maybe_build_fid(device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        return FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    except Exception:
        return None


@torch.no_grad()
def evaluate_one(cfg, split: str, ckpt_path: str, name: str, device: torch.device):
    manifest = cfg["data"].get("split_csv")
    if not manifest:
        manifest = str(Path(cfg["output_dir"]) / "manifests" / f'{cfg["data"]["name"]}_split.csv')

    ds = get_dataset_and_manifest(cfg, split, manifest)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["data"].get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(cfg).to(device).eval()
    load_ckpt(model, ckpt_path)
    objective = LocalObjective(cfg, device)
    fid = maybe_build_fid(device)

    id_sum = 0.0
    sync_sum = 0.0
    lpips_sum = 0.0
    tjit_sum = 0.0
    n = 0

    for batch in loader:
        video = batch["video"].to(device).float()
        ref = batch["ref"].to(device).float()
        audio = batch.get("audio", None)
        if audio is not None:
            audio = audio.to(device).float()

        out = model(video=video, audio=audio, ref=ref)
        x0_pred = model.scheduler.predict_x0(out["zt"], out["t"], out["eps_pred"])
        video_hat = model.vae.decode_video(x0_pred).clamp(0.0, 1.0)

        center_gt = objective._center_frame(video)
        center_hat = objective._center_frame(video_hat)

        id_sim = objective._identity_similarity(center_hat, ref)
        sync_err = objective._sync_proxy_loss(video_hat, audio)
        lpips = objective._perceptual_loss(center_hat, center_gt)
        tjit = objective._temporal_jitter(video_hat, video)

        id_sum += float(id_sim.item())
        sync_sum += float(sync_err.item())
        lpips_sum += float(lpips.item())
        tjit_sum += float(tjit.item())
        n += 1

        if fid is not None:
            real = (center_gt.clamp(0, 1) * 255.0).to(torch.uint8)
            fake = (center_hat.clamp(0, 1) * 255.0).to(torch.uint8)
            fid.update(real, real=True)
            fid.update(fake, real=False)

    adapter_state = get_adapter_state(model, trainable_only=True)
    comm_bytes = sum(v.numel() * v.element_size() for v in adapter_state.values())

    row = {
        "method": name,
        "checkpoint": ckpt_path,
        "identity": id_sum / max(n, 1),
        "sync": sync_sum / max(n, 1),
        "lpips": lpips_sum / max(n, 1),
        "temporal_jitter": tjit_sum / max(n, 1),
        "fid": float(fid.compute().item()) if fid is not None else "",
        "comm_bytes_per_round": comm_bytes,
        "split": split,
        "steps": n,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="Format: method_name=/absolute/path/to/checkpoint.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: List[Dict[str, object]] = []
    for item in args.ckpt:
        name, path = item.split("=", 1)
        rows.append(evaluate_one(cfg, args.split, path, name, device))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method", "checkpoint", "identity", "sync", "lpips",
        "temporal_jitter", "fid", "comm_bytes_per_round", "split", "steps"
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
