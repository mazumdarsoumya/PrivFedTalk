#!/usr/bin/env python
from privfedtalk.cli.export_artifacts import export_latex_tables
from privfedtalk.utils.config import load_config
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    export_latex_tables(cfg)

if __name__ == "__main__":
    main()
