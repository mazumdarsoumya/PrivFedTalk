#!/usr/bin/env python
from privfedtalk.fl.server.orchestrator import run_single_round
from privfedtalk.utils.config import load_config
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_single_round(cfg, round_idx=args.round)

if __name__ == "__main__":
    main()
