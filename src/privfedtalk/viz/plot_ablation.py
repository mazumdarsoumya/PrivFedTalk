import json
import matplotlib.pyplot as plt

def plot_ablation_from_reports(summary_json: str, out_pdf: str):
    with open(summary_json, "r", encoding="utf-8") as f:
        s = json.load(f)
    methods = list(s.get("methods", {}).keys())
    vals = [s["methods"][m].get("identity", 0.0) for m in methods]
    plt.figure()
    plt.bar(methods, vals)
    plt.ylabel("Identity (cosine)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
