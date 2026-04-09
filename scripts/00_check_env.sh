#!/usr/bin/env bash
set -e
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('cuda:', torch.version.cuda)"
