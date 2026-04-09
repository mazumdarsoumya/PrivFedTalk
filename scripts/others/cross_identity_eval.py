from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List
import torch
from PIL import Image

from privfedtalk.utils.config import load_config
from privfedtalk.data.datasets import get_dataset_and_manifest
from privfedtalk.models.build_model import build_model
from privfedtalk.fl.client.local_objective import LocalObjective

def load_ckpt(model, ckpt_path: str):
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model.load_state_dict(state, strict=False)

def to_pil(x: torch.Tensor):
    x = x.detach().clamp(0, 1).cpu()
    if x.dim() == 3:
        x = x.permute(1, 2, 0)
    arr = (x.numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)

def make_strip(images: List[Image.Image], out_path: Path):
    w = sum(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (w, h))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = cfg["data"].get("split_csv")
    ds = get_dataset_and_manifest(cfg, "test", manifest)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_identity_metrics.csv"

    model = build_model(cfg).to(device).eval()
    load_ckpt(model, args.ckpt)
    objective = LocalObjective(cfg, device)

    rows = []
    used = 0

    for i in range(len(ds)):
        if used >= args.n_pairs:
            break
        a = ds[i]
        for j in range(i + 1, len(ds)):
            b = ds[j]
            ida = a.get("identity", "")
            idb = b.get("identity", "")
            if ida and idb and ida == idb:
                continue

            ref = a["ref"].unsqueeze(0).to(device).float()
            audio = b["audio"].unsqueeze(0).to(device).float()

            with torch.no_grad():
                video_hat = model.sample(
                    audio=audio,
                    ref=ref,
                    num_frames=cfg["data"]["video_frames"],
                    height=cfg["data"]["video_size"],
                    width=cfg["data"]["video_size"],
                ).clamp(0.0, 1.0)

            center_hat = objective._center_frame(video_hat)
            id_score = float(objective._identity_similarity(center_hat, ref).item())
            sync_err = float(objective._sync_proxy_loss(video_hat, audio).item())

            ref_center = a["video"][a["video"].shape[0] // 2]
            audio_center = b["video"][b["video"].shape[0] // 2]
            gen_center = center_hat[0]

            out_img = img_dir / f"pair_{used:03d}.png"
            make_strip(
                [to_pil(a["ref"]), to_pil(ref_center), to_pil(audio_center), to_pil(gen_center)],
                out_img
            )

            rows.append({
                "pair_id": used,
                "reference_identity": ida,
                "audio_identity": idb,
                "identity_score": id_score,
                "sync_error": sync_err,
                "image_file": str(out_img),
            })
            used += 1
            break

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["pair_id","reference_identity","audio_identity","identity_score","sync_error","image_file"])
        w.writeheader()
        if rows:
            w.writerows(rows)

    print("Wrote:", csv_path)
    print("Saved images to:", img_dir)

if __name__ == "__main__":
    main()
