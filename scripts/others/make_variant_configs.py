from __future__ import annotations
from pathlib import Path
import copy
import yaml

CONFIG_DIR = Path("configs/paper")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

BASE = yaml.safe_load(Path("configs/lrs3_privfedtalk.yaml").read_text())

BASE_MANIFEST = "/DATA/vineet/PrivFedTalk/outputs_lrs3/manifests/lrs3_split.csv"
MANIFEST_DIR = "/DATA/vineet/PrivFedTalk/paper_manifests"

def clone():
    return copy.deepcopy(BASE)

def set_common(cfg, out_name, manifest=BASE_MANIFEST):
    cfg["data"]["split_csv"] = manifest
    cfg["federated"]["output_dir"] = f"/DATA/vineet/PrivFedTalk/{out_name}"
    cfg["federated"]["rounds"] = 30
    cfg["federated"]["patience"] = 8
    cfg["federated"]["clients_per_round"] = 10
    cfg["federated"]["local_epochs"] = 1
    cfg["federated"]["local_lr"] = 5.0e-06
    cfg["federated"]["server_lr"] = 0.5
    cfg["federated"]["dp_clip_norm"] = 1.0
    cfg["federated"]["dp_noise_multiplier"] = 0.0
    cfg["federated"]["isfa_alpha"] = 0.7
    cfg["federated"]["isfa_gamma"] = 0.5
    cfg["federated"]["prox_mu"] = 0.0
    cfg["loss"]["lambda_tdc"] = 0.05
    cfg["loss"]["lambda_id"] = 0.10
    cfg["loss"]["lambda_perc"] = 0.10
    cfg["loss"]["lambda_sync"] = 0.05
    return cfg

def write_cfg(name, cfg):
    p = CONFIG_DIR / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Wrote:", p)

# ------------------------------------------------------------
# Ablations
# ------------------------------------------------------------
cfg = set_common(clone(), "outputs_paper_abl_adapters")
cfg["federated"]["agg"] = "fedavg"
cfg["loss"]["lambda_tdc"] = 0.0
cfg["federated"]["dp_noise_multiplier"] = 0.0
write_cfg("abl_adapters_only", cfg)

cfg = set_common(clone(), "outputs_paper_abl_dp")
cfg["federated"]["agg"] = "fedavg"
cfg["loss"]["lambda_tdc"] = 0.0
cfg["federated"]["dp_noise_multiplier"] = 0.02
write_cfg("abl_dp_only", cfg)

cfg = set_common(clone(), "outputs_paper_abl_tdc")
cfg["federated"]["agg"] = "fedavg"
cfg["loss"]["lambda_tdc"] = 0.05
cfg["federated"]["dp_noise_multiplier"] = 0.0
write_cfg("abl_tdc_only", cfg)

cfg = set_common(clone(), "outputs_paper_abl_isfa")
cfg["federated"]["agg"] = "isfa"
cfg["loss"]["lambda_tdc"] = 0.0
cfg["federated"]["dp_noise_multiplier"] = 0.0
write_cfg("abl_isfa_only", cfg)

cfg = set_common(clone(), "outputs_paper_abl_full")
cfg["federated"]["agg"] = "isfa"
cfg["loss"]["lambda_tdc"] = 0.05
cfg["federated"]["dp_noise_multiplier"] = 0.02
write_cfg("abl_full", cfg)

# ------------------------------------------------------------
# Privacy sweep
# ------------------------------------------------------------
for sigma in [0.0, 0.02, 0.05, 0.10, 0.20]:
    cfg = set_common(clone(), f"outputs_paper_privacy_sigma_{str(sigma).replace('.','p')}")
    cfg["federated"]["agg"] = "isfa"
    cfg["federated"]["dp_noise_multiplier"] = sigma
    write_cfg(f"privacy_sigma_{str(sigma).replace('.','p')}", cfg)

# ------------------------------------------------------------
# Non-IID stress via limited client pools
# ------------------------------------------------------------
for k in [50, 100, 200]:
    cfg = set_common(clone(), f"outputs_paper_noniid_k_{k}", manifest=f"{MANIFEST_DIR}/lrs3_clients_{k}.csv")
    cfg["federated"]["agg"] = "isfa"
    cfg["federated"]["clients_per_round"] = min(10, k)
    write_cfg(f"noniid_k_{k}", cfg)

# ------------------------------------------------------------
# Partial participation (using K=100 manifest)
# p in {0.1, 0.2, 0.5} => clients_per_round = 10, 20, 50
# ------------------------------------------------------------
for p, cpr in [("0p1", 10), ("0p2", 20), ("0p5", 50)]:
    cfg = set_common(clone(), f"outputs_paper_participation_{p}", manifest=f"{MANIFEST_DIR}/lrs3_clients_100.csv")
    cfg["federated"]["agg"] = "isfa"
    cfg["federated"]["clients_per_round"] = cpr
    write_cfg(f"participation_{p}", cfg)
