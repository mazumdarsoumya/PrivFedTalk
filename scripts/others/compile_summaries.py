from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path("/DATA/vineet/PrivFedTalk/paper_results")

def read_first_row(csv_path: Path):
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None

# ---------------------------
# Ablation summary
# ---------------------------
abl_map = {
    "FedAvg adapters": ROOT / "ablation" / "abl_adapters_only" / "metrics.csv",
    "SecureAgg (system-level)": ROOT / "ablation" / "abl_adapters_only" / "metrics.csv",
    "+ DP": ROOT / "ablation" / "abl_dp_only" / "metrics.csv",
    "+ TDC": ROOT / "ablation" / "abl_tdc_only" / "metrics.csv",
    "+ ISFA": ROOT / "ablation" / "abl_isfa_only" / "metrics.csv",
    "PrivFedTalk (full)": ROOT / "ablation" / "abl_full" / "metrics.csv",
}
abl_out = ROOT / "ablation_table_filled.csv"
rows = []
for variant, p in abl_map.items():
    row = read_first_row(p)
    if row is None:
        row = {"method": "", "identity": "", "sync": "", "lpips": "", "temporal_jitter": "", "fid": ""}
    row["variant"] = variant
    if variant == "SecureAgg (system-level)":
        row["note"] = "Quality-transparent system feature; same metrics as nearest training-equivalent row."
    else:
        row["note"] = ""
    rows.append(row)

fields = ["variant", "method", "identity", "sync", "lpips", "temporal_jitter", "fid", "note"]
with open(abl_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("Wrote:", abl_out)

# ---------------------------
# Privacy summary
# ---------------------------
privacy_out = ROOT / "privacy_sweep_filled.csv"
fields = ["sigma", "method", "identity", "sync", "lpips", "temporal_jitter", "fid"]
rows = []
for sigma in ["0p0", "0p02", "0p05", "0p1", "0p2"]:
    p = ROOT / "privacy" / f"privacy_sigma_{sigma}" / "metrics.csv"
    row = read_first_row(p)
    if row is None:
        row = {}
    row["sigma"] = sigma.replace("p", ".")
    rows.append(row)
with open(privacy_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("Wrote:", privacy_out)

# ---------------------------
# Non-IID summary
# ---------------------------
noniid_out = ROOT / "noniid_stress_filled.csv"
fields = ["k_clients", "method", "identity", "sync", "lpips", "temporal_jitter", "fid"]
rows = []
for k in [50, 100, 200]:
    p = ROOT / "noniid" / f"noniid_k_{k}" / "metrics.csv"
    row = read_first_row(p)
    if row is None:
        row = {}
    row["k_clients"] = k
    rows.append(row)
with open(noniid_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("Wrote:", noniid_out)

# ---------------------------
# Participation summary
# ---------------------------
part_out = ROOT / "partial_participation_filled.csv"
fields = ["participation", "method", "identity", "sync", "lpips", "temporal_jitter", "fid"]
rows = []
for ptxt in ["0p1", "0p2", "0p5"]:
    p = ROOT / "participation" / f"participation_{ptxt}" / "metrics.csv"
    row = read_first_row(p)
    if row is None:
        row = {}
    row["participation"] = ptxt.replace("p", ".")
    rows.append(row)
with open(part_out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
print("Wrote:", part_out)
