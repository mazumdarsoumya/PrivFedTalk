from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from torchvision.io import read_video


def save_frame(frame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame.numpy())
    img.save(out_path)


def find_client_clip(data_root: Path, client_id: str) -> Path:
    client_dir = data_root / client_id
    if not client_dir.exists():
        raise FileNotFoundError(f"Client folder not found: {client_dir}")

    clips = sorted(client_dir.rglob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No .mp4 clips found under {client_dir}")

    return clips[0]


def main():
    ap = argparse.ArgumentParser(description="Create qualitative figure metadata sheet")
    ap.add_argument("--data-root", required=True, help="LRS3 root, e.g. data/LRS3")
    ap.add_argument("--personalized-root", required=True, help="Root of personalized adapters")
    ap.add_argument("--global-adapter-ckpt", required=True, help="Best global adapter ckpt")
    ap.add_argument("--out-dir", required=True, help="Output folder for metadata + images")
    ap.add_argument("--clients", nargs="+", required=True, help="Client IDs to include")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    personalized_root = Path(args.personalized_root)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for row_idx, client_id in enumerate(args.clients, start=1):
        personal_ckpt = personalized_root / client_id / "latest.pt"
        if not personal_ckpt.exists():
            raise FileNotFoundError(f"Missing personalized adapter: {personal_ckpt}")

        clip_path = find_client_clip(data_root, client_id)

        video, _, info = read_video(str(clip_path), pts_unit="sec")
        if video is None or len(video) == 0:
            raise RuntimeError(f"No frames decoded from {clip_path}")

        num_frames = int(video.shape[0])
        ref_idx = 0
        gt_idx = max(0, num_frames // 2)

        ref_path = fig_dir / f"ref_{client_id}.png"
        gt_path = fig_dir / f"gt_{client_id}_t{gt_idx}.png"

        save_frame(video[ref_idx], ref_path)
        save_frame(video[gt_idx], gt_path)

        baseline_path = fig_dir / f"baseline_{client_id}_t{gt_idx}.png"
        privfedtalk_path = fig_dir / f"privfedtalk_{client_id}_t{gt_idx}.png"

        row = {
            "row": row_idx,
            "client_id": client_id,
            "clip_path": str(clip_path),
            "num_frames": num_frames,
            "video_fps": info.get("video_fps", None) if isinstance(info, dict) else None,
            "reference_frame_index": ref_idx,
            "gt_frame_index": gt_idx,
            "reference_image": str(ref_path),
            "ground_truth_image": str(gt_path),
            "baseline_output_image": str(baseline_path),
            "privfedtalk_output_image": str(privfedtalk_path),
            "global_adapter_ckpt": str(Path(args.global_adapter_ckpt)),
            "personalized_adapter_ckpt": str(personal_ckpt),
            "notes": "baseline_output_image and privfedtalk_output_image are placeholders to be filled after generation/export"
        }
        rows.append(row)

    csv_path = out_dir / "qualitative_rows.csv"
    json_path = out_dir / "qualitative_rows.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("Saved CSV :", csv_path)
    print("Saved JSON:", json_path)
    print("Saved images:")
    for r in rows:
        print(" -", r["reference_image"])
        print(" -", r["ground_truth_image"])


if __name__ == "__main__":
    main()
