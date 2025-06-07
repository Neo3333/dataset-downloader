#!/bin/sh
set -e

# No longer needed
# if [ -n "$HF_HUB_TOKEN" ]; then
#   huggingface-cli login --token "$HF_HUB_TOKEN"
# fi

exec python worker/main.py "$@"
