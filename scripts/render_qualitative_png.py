from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms.functional import pil_to_tensor

from privfedtalk.utils.config import load_config
from privfedtalk.models.build_model import build_model
from privfedtalk.utils.adapter_state import load_adapter_state, move_state_to
from privfedtalk.utils.audio import pad_or_trim


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ckpt(path: str):
    return torch.load(path, map_location="cpu")


def maybe_load_full_model(model: torch.nn.Module, ckpt_obj: Dict[str, Any]) -> None:
    if not isinstance(ckpt_obj, dict):
        return
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        model.load_state_dict(ckpt_obj["model"], strict=False)
    elif "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        model.load_state_dict(ckpt_obj["state_dict"], strict=False)


def extract_adapter_state(ckpt_obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "adapter_state" in ckpt_obj and isinstance(ckpt_obj["adapter_state"], dict):
            return ckpt_obj["adapter_state"]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Could not find adapter_state in checkpoint.")


def square_resize_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    # img: [C,H,W], float in [0,1]
    _, h, w = img.shape
    side = min(h, w)
    top = max(0, (h - side) // 2)
    left = max(0, (w - side) // 2)
    img = img[:, top:top+side, left:left+side]
    img = F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    return img


def load_ref_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = pil_to_tensor(img).float() / 255.0
    x = square_resize_tensor(x, size)
    return x


def load_audio_from_video(path: str, target_len: int, target_sr: int) -> torch.Tensor:
    _v, audio, info = read_video(path, pts_unit="sec")
    if audio.numel() == 0:
        return torch.zeros(target_len, dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    audio = audio.float()
    src_sr = int(info.get("audio_fps", target_sr))
    if src_sr != target_sr:
        audio = AF.resample(audio.unsqueeze(0), src_sr, target_sr).squeeze(0)
    return pad_or_trim(audio, target_len).float()


def save_frame_png(frame_chw: torch.Tensor, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = frame_chw.detach().cpu().clamp(0, 1)
    img = (x * 255.0).byte().permute(1, 2, 0).numpy()
    Image.fromarray(img).save(out)


def pick_row(rows: List[Dict[str, Any]], client_id: str) -> Dict[str, Any]:
    for r in rows:
        if r["client_id"] == client_id:
            return r
    raise RuntimeError(f"client_id not found in rows json: {client_id}")


def parse_args():
    ap = argparse.ArgumentParser(description="Render baseline/PrivFedTalk PNG for qualitative figure")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows-json", required=True)
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--global-adapter-ckpt", required=True)
    ap.add_argument("--personalized-adapter-ckpt", default="")
    ap.add_argument("--base-ckpt", default="")
    ap.add_argument("--kind", required=True, choices=["baseline", "privfedtalk"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1234)))

    rows = json.loads(Path(args.rows_json).read_text())
    row = pick_row(rows, args.client_id)

    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()

    if args.base_ckpt:
        base_obj = load_ckpt(args.base_ckpt)
        maybe_load_full_model(model, base_obj)

    global_obj = load_ckpt(args.global_adapter_ckpt)
    global_adapter = extract_adapter_state(global_obj)
    load_adapter_state(model, move_state_to(global_adapter, device))

    if args.personalized_adapter_ckpt:
        personal_obj = load_ckpt(args.personalized_adapter_ckpt)
        personal_adapter = extract_adapter_state(personal_obj)
        load_adapter_state(model, move_state_to(personal_adapter, device))

    video, _, _ = read_video(row["clip_path"], pts_unit="sec")
    if video is None or len(video) == 0:
        raise RuntimeError(f"Could not decode video: {row['clip_path']}")

    num_frames = int(row["num_frames"])
    gt_idx = int(row["gt_frame_index"])
    raw_h = int(video.shape[1])
    raw_w = int(video.shape[2])

    ref = load_ref_image(row["reference_image"], int(cfg["data"]["video_size"])).unsqueeze(0).to(device)
    audio = load_audio_from_video(
        row["clip_path"],
        int(cfg["data"]["audio_len"]),
        int(cfg["data"].get("audio_sr", 16000)),
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        sample = model.sample(
            audio=audio,
            ref=ref,
            num_frames=num_frames,
            height=raw_h,
            width=raw_w,
        )

    idx = min(max(gt_idx, 0), sample.shape[1] - 1)
    frame = sample[0, idx]

    if args.out:
        out_path = args.out
    else:
        if args.kind == "baseline":
            out_path = row["baseline_output_image"]
        else:
            out_path = row["privfedtalk_output_image"]

    save_frame_png(frame, out_path)
    print("Saved:", out_path)
    print("client_id:", row["client_id"])
    print("clip:", row["clip_path"])
    print("gt_frame_index:", gt_idx)
    print("used_frame_index:", idx)
    print("kind:", args.kind)


if __name__ == "__main__":
    main()
