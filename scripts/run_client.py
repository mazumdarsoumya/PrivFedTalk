#!/usr/bin/env python
from privfedtalk.fl.client.client_trainer import run_single_client_local_training
from privfedtalk.utils.config import load_config
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--client_id", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_single_client_local_training(cfg, client_id=args.client_id)

if __name__ == "__main__":
    main()
