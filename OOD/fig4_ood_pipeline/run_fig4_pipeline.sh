#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/src/process_fig4_ood.py" --config "$HERE/configs/fig4_ood_config.yaml"
