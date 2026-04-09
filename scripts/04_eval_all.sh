#!/usr/bin/env bash
set -e
python -m privfedtalk.cli.eval --config configs/default.yaml
python -m privfedtalk.cli.export_artifacts --config configs/default.yaml --mode export_latex_tables
