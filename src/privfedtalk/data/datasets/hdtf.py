from __future__ import annotations

from pathlib import Path
import csv
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _identity_from_path(p: Path) -> str:
    stem = p.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_or_load_manifest(
    root_dir: str,
    manifest_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    root = Path(root_dir)
    manifest = Path(manifest_path)

    if manifest.exists():
        rows = []
        with open(manifest, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "split ratios must sum to 1"

    mp4s = sorted(root.rglob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"No .mp4 files found under {root}")

    by_identity = defaultdict(list)
    for p in mp4s:
        ident = _identity_from_path(p)
        by_identity[ident].append(p)

    identities = sorted(by_identity.keys())
    rng = random.Random(seed)
    rng.shuffle(identities)

    n = len(identities)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else max(0, n - n_train)
    n_test = max(0, n - n_train - n_val)

    if n_test == 0 and n >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train:n_train + n_val])
    test_ids = set(identities[n_train + n_val:])

    rows = []
    for ident, paths in by_identity.items():
        if ident in train_ids:
            split = "train"
        elif ident in val_ids:
            split = "val"
        else:
            split = "test"

        for p in sorted(paths):
            rows.append({
                "path": str(p.resolve()),
                "identity": ident,
                "split": split,
            })

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "identity", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


build_manifest = build_or_load_manifest


class HDTFClipDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split_csv: str,
        split: str,
        num_frames: int = 32,
        video_size: int = 128,
        audio_len: int = 16000,
    ):
        self.root_dir = Path(root_dir)
        self.split_csv = Path(split_csv)
        self.split = split
        self.num_frames = num_frames
        self.video_size = video_size
        self.audio_len = audio_len

        self.items = []
        with open(self.split_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append(row)

        if not self.items:
            raise RuntimeError(f"No HDTF items found for split={split}")

    def __len__(self):
        return len(self.items)

    def _read_video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Could not decode video: {path}")

        return frames

    def _sample_indices(self, n_frames: int):
        if n_frames >= self.num_frames:
            idx = np.linspace(0, n_frames - 1, self.num_frames).round().astype(int)
        else:
            idx = list(range(n_frames))
            while len(idx) < self.num_frames:
                idx.append(idx[-1])
            idx = np.array(idx, dtype=int)
        return idx

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = Path(item["path"])
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()

        frames = self._read_video(path)
        sel = self._sample_indices(len(frames))

        out = []
        for i in sel:
            fr = frames[int(i)]
            fr = cv2.resize(fr, (self.video_size, self.video_size), interpolation=cv2.INTER_AREA)
            out.append(fr)

        video = np.stack(out, axis=0)  # T,H,W,C
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0  # T,C,H,W

        ref = video[video.size(0) // 2]
        audio = torch.zeros(self.audio_len, dtype=torch.float32)

        return {
            "video": video,
            "ref": ref,
            "audio": audio,
            "identity": item["identity"],
        }
