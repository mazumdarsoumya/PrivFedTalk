from __future__ import annotations

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from privfedtalk.utils.config import load_config
from privfedtalk.data.datasets.lrs3 import LRS3ClipDataset, build_or_load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("output_dir", "outputs")
    manifest_path = f'{out_dir}/splits/lrs3_split_seed_{int(cfg["seed"])}.json'

    build_or_load_manifest(
        root=cfg["data"]["root"],
        output_json=manifest_path,
        seed=int(cfg["seed"]),
        ratios=(
            float(cfg["data"]["train_ratio"]),
            float(cfg["data"]["val_ratio"]),
            float(cfg["data"]["test_ratio"]),
        ),
    )

    for split in ["train", "val", "test"]:
        ds = LRS3ClipDataset(cfg, split=split, manifest_path=manifest_path)
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=False,
            persistent_workers=bool(cfg["data"].get("persistent_workers", True)) if int(cfg["data"]["num_workers"]) > 0 else False,
            prefetch_factor=int(cfg["data"].get("prefetch_factor", 4)) if int(cfg["data"]["num_workers"]) > 0 else None,
        )
        for _ in tqdm(dl, desc=f"caching {split}", total=len(ds)):
            pass

    print("Caching complete.", flush=True)


if __name__ == "__main__":
    main()
