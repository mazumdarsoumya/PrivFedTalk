from __future__ import annotations
import csv
import json
import random
from pathlib import Path
from copy import deepcopy

BASE_MANIFEST = Path("/DATA/vineet/PrivFedTalk/outputs_lrs3/manifests/lrs3_split.csv")
OUT_DIR = Path("/DATA/vineet/PrivFedTalk/paper_manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1234
CLIENT_COUNTS = [50, 100, 200]

def client_key(row):
    for k in ("group", "identity", "speaker", "client_id"):
        v = row.get(k, "")
        if v:
            return v
    p = row.get("video", "") or row.get("path", "") or row.get("video_path", "")
    if p:
        pp = Path(p)
        return pp.parent.name or pp.stem.split("_")[0]
    return "client_0"

def load_manifest(path: Path):
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        raise RuntimeError(f"Empty manifest: {path}")

    # Case 1: true JSON manifest
    if txt.startswith("{") or txt.startswith("["):
        obj = json.loads(txt)

        # Already the proper structured manifest
        if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], dict):
            return obj

        # Raw list -> convert into structured manifest
        if isinstance(obj, list):
            samples = {"train": [], "val": [], "test": []}
            for r in obj:
                sp = r.get("split", "")
                if sp in samples:
                    samples[sp].append(r)
            return {
                "root": "",
                "seed": SEED,
                "ratios": [0.8, 0.1, 0.1],
                "samples": samples,
            }

        raise RuntimeError(f"Unsupported JSON manifest structure in {path}")

    # Case 2: CSV rows -> convert into structured manifest
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    samples = {"train": [], "val": [], "test": []}
    for r in rows:
        sp = r.get("split", "")
        if sp in samples:
            samples[sp].append(r)

    return {
        "root": "",
        "seed": SEED,
        "ratios": [0.8, 0.1, 0.1],
        "samples": samples,
    }

manifest = load_manifest(BASE_MANIFEST)

train_rows = manifest["samples"].get("train", [])
val_rows = manifest["samples"].get("val", [])
test_rows = manifest["samples"].get("test", [])

clients = sorted(set(client_key(r) for r in train_rows if client_key(r)))
rng = random.Random(SEED)
rng.shuffle(clients)

print("Base manifest       :", BASE_MANIFEST)
print("Train rows          :", len(train_rows))
print("Val rows            :", len(val_rows))
print("Test rows           :", len(test_rows))
print("Total train clients :", len(clients))

for k in CLIENT_COUNTS:
    chosen = set(clients[:k])
    subset_train = [r for r in train_rows if client_key(r) in chosen]

    out_manifest = deepcopy(manifest)
    out_manifest["samples"]["train"] = subset_train
    out_manifest["samples"]["val"] = val_rows
    out_manifest["samples"]["test"] = test_rows

    out = OUT_DIR / f"lrs3_clients_{k}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f)

    print("Wrote:", out)
    print("  train rows   :", len(subset_train))
    print("  val rows     :", len(val_rows))
    print("  test rows    :", len(test_rows))
