#!/usr/bin/env bash
set -e
python -m privfedtalk.cli.export_artifacts --config configs/default.yaml --mode make_synthetic_data
