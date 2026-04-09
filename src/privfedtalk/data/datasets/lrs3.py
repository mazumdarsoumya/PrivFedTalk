from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchaudio.functional as AF

from privfedtalk.utils.audio import pad_or_trim


def _collect_groups(root: Path) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        items: List[Dict[str, str]] = []
        for mp4 in sorted(folder.glob("*.mp4")):
            txt = mp4.with_suffix(".txt")
            if txt.exists():
                items.append({
                    "group": str(folder.name),
                    "video": str(mp4.resolve()),
                    "text": str(txt.resolve()),
                    "clip_id": str(mp4.stem),
                })
        if items:
            groups[str(folder.name)] = items
    return groups


def build_or_load_manifest(root: str, output_json: str, seed: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, Any]:
    output_path = Path(output_json)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    root_path = Path(root)
    groups = _collect_groups(root_path)
    if not groups:
        raise RuntimeError(f"No valid LRS3 groups found under: {root}")

    group_ids = sorted(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = max(1, int(n_groups * ratios[0]))
    n_val = max(1, int(n_groups * ratios[1]))
    if n_train + n_val >= n_groups:
        n_val = max(1, n_groups - n_train - 1)
    n_test = n_groups - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_groups = group_ids[:n_train]
    val_groups = group_ids[n_train:n_train + n_val]
    test_groups = group_ids[n_train + n_val:]

    manifest = {
        "root": str(root_path.resolve()),
        "seed": seed,
        "ratios": list(ratios),
        "splits": {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        },
        "samples": {
            "train": [item for gid in train_groups for item in groups[gid]],
            "val": [item for gid in val_groups for item in groups[gid]],
            "test": [item for gid in test_groups for item in groups[gid]],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class LRS3ClipDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str, manifest_path: str):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_cfg = cfg["data"]
        self.manifest = build_or_load_manifest(
            root=self.data_cfg["root"],
            output_json=manifest_path,
            seed=int(cfg["seed"]),
            ratios=(
                float(self.data_cfg["train_ratio"]),
                float(self.data_cfg["val_ratio"]),
                float(self.data_cfg["test_ratio"]),
            ),
        )
        self.items: List[Dict[str, str]] = self.manifest["samples"][split]
        self.num_frames = int(self.data_cfg["video_frames"])
        self.video_size = int(self.data_cfg["video_size"])
        self.audio_len = int(self.data_cfg["audio_len"])
        self.audio_sr = int(self.data_cfg.get("audio_sr", 16000))
        self.base_seed = int(cfg["seed"])
        self.train_mode = split == "train"

        self.use_cache = bool(self.data_cfg.get("use_cache", True))
        self.cache_root = Path(str(self.data_cfg.get("cache_root", "data/LRS3_cache"))).expanduser().resolve()

    def __len__(self) -> int:
        return len(self.items)

    def _safe_name(self, x: Any) -> str:
        s = str(x)
        s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return s

    def _cache_path(self, item: Dict[str, str]) -> Path:
        group = self._safe_name(item.get("group", "unknown_group"))
        clip_id = self._safe_name(item.get("clip_id", "unknown_clip"))
        fname = f"{clip_id}_f{self.num_frames}_s{self.video_size}_a{self.audio_len}.pt"
        return self.cache_root / group / fname

    def _square_resize(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, h, w = frames.shape
        side = min(h, w)
        top = max(0, (h - side) // 2)
        left = max(0, (w - side) // 2)
        frames = frames[:, :, top:top + side, left:left + side]
        frames = F.interpolate(frames, size=(self.video_size, self.video_size), mode="bilinear", align_corners=False)
        return frames

    def _sample_frames(self, frames: torch.Tensor, idx: int) -> torch.Tensor:
        t = frames.size(0)
        if t <= 0:
            raise RuntimeError("Video contains zero frames")
        if t >= self.num_frames:
            if self.train_mode:
                rnd = random.Random(self.base_seed + idx)
                start = rnd.randint(0, max(0, t - self.num_frames))
            else:
                start = max(0, (t - self.num_frames) // 2)
            frames = frames[start:start + self.num_frames]
        else:
            reps = (self.num_frames + t - 1) // t
            frames = frames.repeat(reps, 1, 1, 1)[:self.num_frames]
        return frames

    def _load_audio(self, path: str) -> torch.Tensor:
        _v, audio, info = read_video(path, pts_unit="sec")
        if audio.numel() == 0:
            return torch.zeros(self.audio_len, dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        audio = audio.float()
        sr = int(info.get("audio_fps", self.audio_sr))
        if sr != self.audio_sr:
            audio = AF.resample(audio.unsqueeze(0), sr, self.audio_sr).squeeze(0)
        return pad_or_trim(audio, self.audio_len).float()

    def _decode_item(self, item: Dict[str, str], idx: int) -> Dict[str, Any]:
        frames, _, _ = read_video(item["video"], pts_unit="sec")
        if frames.numel() == 0:
            raise RuntimeError(f"Empty video: {item['video']}")
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = self._sample_frames(frames, idx)
        frames = self._square_resize(frames)
        ref = frames[self.num_frames // 2]
        audio = self._load_audio(item["video"])
        return {
            "video": frames,
            "audio": audio,
            "ref": ref,
            "group": str(item["group"]),
            "clip_id": str(item["clip_id"]),
            "video_path": str(item["video"]),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        n = len(self.items)
        max_tries = min(32, n)
        last_err = None

        for offset in range(max_tries):
            j = (int(idx) + offset) % n
            item = self.items[j]
            try:
                cache_path = self._cache_path(item)

                if self.use_cache and cache_path.exists():
                    try:
                        return torch.load(cache_path, map_location="cpu")
                    except Exception:
                        try:
                            cache_path.unlink()
                        except Exception:
                            pass

                pack = self._decode_item(item, j)

                if self.use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = cache_path.parent / f"{cache_path.name}.{os.getpid()}.tmp"
                    torch.save(pack, tmp_path)
                    os.replace(tmp_path, cache_path)

                return pack

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"Failed to load sample after {max_tries} attempts in split={self.split}. "
            f"Last error: {repr(last_err)}"
        )
