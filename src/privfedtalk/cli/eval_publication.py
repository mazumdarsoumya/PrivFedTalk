from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import lpips
from facenet_pytorch import InceptionResnetV1
from privfedtalk.utils.adapter_state import load_adapter_state

from privfedtalk.utils.config import load_config
from privfedtalk.utils.seed import set_seed
from privfedtalk.utils.dist import resolve_device
from privfedtalk.utils.io import ensure_dir
from privfedtalk.models.build_model import build_model
from privfedtalk.metrics.temporal_jitter import temporal_jitter
from privfedtalk.data.datasets import get_dataset_and_manifest, build_manifest


def main():
    ap = argparse.ArgumentParser(description="Publication-style evaluation on leak-free LRS3 test split.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    set_seed(int(cfg["seed"]))

    # generation stays on GPU
    device = resolve_device(cfg.get("device", "auto"))
    # heavy metrics go to CPU to avoid CUDA OOM
    metric_device = torch.device("cpu")

    out_dir = cfg.get("output_dir", "outputs")
    report_dir = os.path.join(out_dir, "reports")
    ensure_dir(report_dir)

    data_name = str(cfg["data"]["name"]).lower()

    if data_name == "hdtf":
        manifest_path = cfg["data"].get(
            "split_csv",
            os.path.join(out_dir, "splits", f"hdtf_split_seed_{int(cfg['seed'])}.csv"),
        )
    else:
        manifest_path = os.path.join(
            out_dir, "splits", f"lrs3_split_seed_{int(cfg['seed'])}.json"
        )

    build_manifest(cfg, manifest_path)

    test_ds = get_dataset_and_manifest(cfg, split="test", manifest_path=manifest_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(cfg).to(device).eval()
    model.scheduler.to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "adapter_state" in state:
        load_adapter_state(model, state["adapter_state"])
    elif isinstance(state, dict):
        state_dict = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # keep evaluation models on CPU
    lpips_metric = lpips.LPIPS(net="alex").to(metric_device).eval()
    face_model = InceptionResnetV1(pretrained="vggface2").eval().to(metric_device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(metric_device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(metric_device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    kid_metric = KernelInceptionDistance(subset_size=50, normalize=False).to(metric_device)

    lpips_sum = 0.0
    id_sum = 0.0
    jitter_sum = 0.0
    sync_proxy_sum = 0.0
    n_items = 0

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            video = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            ref = batch["ref"].to(device, non_blocking=True)

            fake = model.sample(
                audio=audio,
                ref=ref,
                num_frames=video.size(1),
                height=video.size(-2),
                width=video.size(-1),
            )

            # move generated / real tensors to CPU for metrics
            real_video_cpu = video.detach().cpu().clamp(0, 1)
            fake_video_cpu = fake.detach().cpu().clamp(0, 1)
            ref_cpu = ref.detach().cpu().clamp(0, 1)
            audio_cpu = audio.detach().cpu()

            # center-frame metrics for FID/KID and identity
            center_idx = fake_video_cpu.size(1) // 2
            real_center = real_video_cpu[:, center_idx]
            fake_center = fake_video_cpu[:, center_idx]

            # PSNR / SSIM on center frames
            psnr_metric.update(fake_center, real_center)
            ssim_metric.update(fake_center, real_center)

            # LPIPS on center frames
            lpips_sum += float(lpips_metric(fake_center * 2 - 1, real_center * 2 - 1).mean().item())

            # FID / KID on center frames only
            real_u8 = (real_center * 255.0).to(torch.uint8)
            fake_u8 = (fake_center * 255.0).to(torch.uint8)
            fid_metric.update(real_u8, real=True)
            fid_metric.update(fake_u8, real=False)
            kid_metric.update(real_u8, real=True)
            kid_metric.update(fake_u8, real=False)

            # identity on center frame
            ref_face = F.interpolate(ref_cpu, size=(160, 160), mode="bilinear", align_corners=False)
            fake_face = F.interpolate(fake_center, size=(160, 160), mode="bilinear", align_corners=False)
            ref_emb = F.normalize(face_model(ref_face), dim=-1)
            fake_emb = F.normalize(face_model(fake_face), dim=-1)
            id_sum += float((ref_emb * fake_emb).sum(dim=-1).mean().item())

            # temporal jitter on full video
            jitter_sum += float(temporal_jitter(fake_video_cpu))

            # internal sync proxy
            audio_emb = model.audio_enc(audio).detach().cpu()
            vid_emb = model.id_enc(fake[:, fake.size(1) // 2]).detach().cpu()
            sync_proxy_sum += float(
                (F.normalize(audio_emb, dim=-1) * F.normalize(vid_emb, dim=-1)).sum(dim=-1).mean().item()
            )

            n_items += 1

            # free GPU memory each loop
            del video, audio, ref, fake
            if device.type == "cuda":
                torch.cuda.empty_cache()

    kid_mean, kid_std = kid_metric.compute()
    summary = {
        "split": "test",
        "checkpoint": args.checkpoint,
        "num_batches": n_items,
        "metrics": {
            "psnr_center_frame": float(psnr_metric.compute().item()),
            "ssim_center_frame": float(ssim_metric.compute().item()),
            "lpips_center_frame": float(lpips_sum / max(1, n_items)),
            "fid_center_frame": float(fid_metric.compute().item()),
            "kid_mean_center_frame": float(kid_mean.item()),
            "kid_std_center_frame": float(kid_std.item()),
            "identity_face_embedding_center_frame": float(id_sum / max(1, n_items)),
            "temporal_jitter_full_video": float(jitter_sum / max(1, n_items)),
            "sync_proxy_model_emb": float(sync_proxy_sum / max(1, n_items)),
        },
        "note": "FID/KID/PSNR/SSIM/LPIPS/identity are computed on center frames to keep evaluation memory-safe. temporal_jitter is computed on full videos. sync_proxy_model_emb is an internal proxy and should be replaced by SyncNet/LSE-C/LSE-D before final IEEE/ACM submission.",
    }

    out_path = os.path.join(report_dir, "publication_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
