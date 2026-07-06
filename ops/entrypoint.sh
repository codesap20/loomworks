#!/bin/bash
# Standardized trainer entrypoint: the validator passes the task CLI here.
set -u
cd /workspace
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
exec python /workspace/src/main.py "$@"
