#!/bin/sh
set -e

if [ -n "$HF_HUB_TOKEN" ]; then
  huggingface-cli login --token "$HF_HUB_TOKEN"
fi

exec python main.py "$@"
