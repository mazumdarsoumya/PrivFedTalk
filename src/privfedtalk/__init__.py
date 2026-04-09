from __future__ import annotations

from typing import Dict, Any

from privfedtalk.data.datasets.lrs3 import (
    LRS3ClipDataset,
    build_or_load_manifest as build_lrs3_manifest,
)
from privfedtalk.data.datasets.hdtf import (
    HDTFClipDataset,
    build_or_load_manifest as build_hdtf_manifest,
)


def get_dataset_and_manifest(cfg: Dict[str, Any], split: str, manifest_path: str):
    name = str(cfg["data"]["name"]).lower()

    if name == "lrs3":
        return LRS3ClipDataset(cfg, split=split, manifest_path=manifest_path)

    if name == "hdtf":
        return HDTFClipDataset(
            root_dir=cfg["data"]["root"],
            split_csv=manifest_path,
            split=split,
            num_frames=int(cfg["data"]["video_frames"]),
            video_size=int(cfg["data"]["video_size"]),
            audio_len=int(cfg["data"].get("audio_len", 16000)),
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def build_manifest(cfg: Dict[str, Any], manifest_path: str):
    name = str(cfg["data"]["name"]).lower()

    if name == "lrs3":
        return build_lrs3_manifest(
            root=cfg["data"]["root"],
            output_json=manifest_path,
            seed=int(cfg["seed"]),
            ratios=(
                float(cfg["data"]["train_ratio"]),
                float(cfg["data"]["val_ratio"]),
                float(cfg["data"]["test_ratio"]),
            ),
        )

    if name == "hdtf":
        return build_hdtf_manifest(
            root_dir=cfg["data"]["root"],
            manifest_path=manifest_path,
            train_ratio=float(cfg["data"].get("train_ratio", 0.8)),
            val_ratio=float(cfg["data"].get("val_ratio", 0.1)),
            test_ratio=float(cfg["data"].get("test_ratio", 0.1)),
            seed=int(cfg["seed"]),
        )

    raise ValueError(f"Unsupported dataset name: {name}")