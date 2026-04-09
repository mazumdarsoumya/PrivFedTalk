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
            audio_len=int(cfg["data"]["audio_len"]),
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
            train_ratio=float(cfg["data"]["train_ratio"]),
            val_ratio=float(cfg["data"]["val_ratio"]),
            test_ratio=float(cfg["data"]["test_ratio"]),
            seed=int(cfg["seed"]),
        )

    raise ValueError(f"Unsupported dataset name: {name}")