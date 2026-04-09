#!/usr/bin/env python
from privfedtalk.fl.server.orchestrator import run_federated_training
from privfedtalk.utils.config import load_config
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_federated_training(cfg)

if __name__ == "__main__":
    main()
