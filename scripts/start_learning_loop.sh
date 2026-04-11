#!/usr/bin/env bash
# start_learning_loop.sh
# 
# Launches the Claudio Autonomous Critic & Trainer Daemon on the background.
# It watches data/training_ingest/ for new audio stems, calculates fidelity,
# and triggers DDSP parameter fine-tuning to hot-swap into the running server.
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"

echo "╔════════════════════════════════════════════════════════╗"
echo "║ Launching Claudio Autonomous Intelligence Daemon       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo "Watch Directory: $PROJECT_ROOT/data/training_ingest"
echo "Hot-Swap Target: $PROJECT_ROOT/checkpoints/forge_model_best.pt"

# Ensure data directory exists
mkdir -p "$PROJECT_ROOT/data/training_ingest"

cd "$PROJECT_ROOT"
PYTHONPATH="src" ./.venv/bin/python src/claudio/intelligence/autonomous_trainer.py
