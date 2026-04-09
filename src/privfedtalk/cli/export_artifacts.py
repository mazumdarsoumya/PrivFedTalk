import argparse, os, json
import pandas as pd
from privfedtalk.utils.config import load_config
from privfedtalk.utils.io import ensure_dir
from privfedtalk.viz.plot_convergence import plot_convergence_from_csv
from privfedtalk.viz.plot_ablation import plot_ablation_from_reports
from privfedtalk.viz.plot_privacy_tradeoff import plot_privacy_tradeoff
from privfedtalk.utils.seed import set_seed
from privfedtalk.data.preprocess.build_clients_partition import build_synthetic_client_partition

def export_latex_tables(cfg):
    out_dir = cfg.get("output_dir", "outputs")
    reports_dir = os.path.join(out_dir, "reports")
    tables_dir = os.path.join(reports_dir, "tables")
    ensure_dir(tables_dir)

    summary_path = os.path.join(reports_dir, "metrics_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("Run eval first to create metrics_summary.json")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = []
    for method, vals in summary.get("methods", {}).items():
        rows.append({
            "method": method,
            "identity": vals.get("identity", None),
            "sync": vals.get("sync", None),
            "lpips": vals.get("lpips", None),
            "temporal_jitter": vals.get("temporal_jitter", None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, "main_results.csv"), index=False)

    paper_tables = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "paper", "tables"))
    ensure_dir(paper_tables)
    tex = df.to_latex(index=False, float_format="%.4f", caption="Main quantitative results (synthetic demo).", label="tab:main_results")
    with open(os.path.join(paper_tables, "main_results.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print("Exported tables to outputs/reports/tables and paper/tables")

def make_paper_artifacts(cfg):
    out_dir = cfg.get("output_dir", "outputs")
    reports_dir = os.path.join(out_dir, "reports")
    fig_dir = os.path.join(reports_dir, "figures")
    ensure_dir(fig_dir)

    csv_log = os.path.join(out_dir, "logs", "csv", "train_log.csv")
    if os.path.exists(csv_log):
        plot_convergence_from_csv(csv_log, os.path.join(fig_dir, "convergence.pdf"))

    summary_path = os.path.join(reports_dir, "metrics_summary.json")
    if os.path.exists(summary_path):
        plot_ablation_from_reports(summary_path, os.path.join(fig_dir, "ablation.pdf"))

    plot_privacy_tradeoff(os.path.join(fig_dir, "privacy_tradeoff.pdf"))
    print("Figures exported to:", fig_dir)

def make_synthetic_data(cfg):
    set_seed(cfg["seed"])
    part_dir = os.path.join("data", "partitions")
    ensure_dir(part_dir)
    part_path = os.path.join(part_dir, f"synthetic_clients_k{cfg['data']['num_clients']}.json")
    partition = build_synthetic_client_partition(
        num_clients=cfg["data"]["num_clients"],
        samples_per_client=cfg["data"]["samples_per_client"],
        non_iid=cfg["data"]["non_iid"],
        iid_fraction=cfg["data"]["iid_fraction"],
    )
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(partition, f, indent=2)
    print("Wrote:", part_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="make_paper_artifacts",
                    choices=["make_paper_artifacts", "make_synthetic_data", "export_latex_tables"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "make_paper_artifacts":
        make_paper_artifacts(cfg)
    elif args.mode == "make_synthetic_data":
        make_synthetic_data(cfg)
    else:
        export_latex_tables(cfg)

if __name__ == "__main__":
    main()
