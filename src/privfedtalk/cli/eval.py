import argparse
from privfedtalk.utils.config import load_config
from privfedtalk.metrics.report import evaluate_and_report

def main():
    ap = argparse.ArgumentParser(description="Evaluate PrivFedTalk.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate_and_report(cfg)

if __name__ == "__main__":
    main()
