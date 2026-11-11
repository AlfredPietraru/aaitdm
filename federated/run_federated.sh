#!/usr/bin/env bash
# Start a local Flower deployment: one SuperLink (server), NUM_CLIENTS SuperNodes
# (hospitals), then submit the app and stream the server log until it finishes.
#
#   ./run_federated.sh                                  # defaults from pyproject.toml
#   NUM_CLIENTS=4 ./run_federated.sh num-server-rounds=10 proximal-mu=0.01
#
# Per-process logs are written to outputs/logs/.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$APP_DIR")"
NUM_CLIENTS="${NUM_CLIENTS:-3}"
SUPERLINK_PORT="${SUPERLINK_PORT:-8000}"
LOG_DIR="$APP_DIR/outputs/logs"

export PATH="${VENV_BIN:-$PROJECT_DIR/.venv/bin}:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Must stay outside APP_DIR: flwr run scans APP_DIR when packaging the app.
export FLWR_HOME="${FLWR_HOME:-$HOME/.flwr-aitdm}"
mkdir -p "$LOG_DIR" "$FLWR_HOME"

cat > "$FLWR_HOME/config.toml" <<EOF
[superlink]
default = "local-deployment"

[superlink.local-deployment]
address = "127.0.0.1:$SUPERLINK_PORT"
insecure = true
EOF

pids=()
cleanup() {
    kill "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() {
    for _ in $(seq 60); do
        (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0
        sleep 1
    done
    echo "Port $1 did not open; see $LOG_DIR" >&2
    return 1
}

echo "Starting SuperLink (server) on port $SUPERLINK_PORT"
# Without --disable-runtime-dependency-installation the SuperLink builds a fresh
# multi-GB environment from pyproject.toml for every run instead of using this venv.
flower-superlink --insecure --port "$SUPERLINK_PORT" --disable-runtime-dependency-installation \
    > "$LOG_DIR/superlink.log" 2>&1 &
pids+=($!)
wait_for_port "$SUPERLINK_PORT"

for ((i = 0; i < NUM_CLIENTS; i++)); do
    echo "Starting SuperNode (hospital $i)"
    flower-supernode --insecure \
        --superlink "127.0.0.1:$SUPERLINK_PORT" \
        --port "$((9094 + i))" \
        --node-config "partition-id=$i num-partitions=$NUM_CLIENTS" \
        > "$LOG_DIR/hospital-$i.log" 2>&1 &
    pids+=($!)
done

overrides=(--run-config "min-nodes=$NUM_CLIENTS")
for override in "$@"; do
    overrides+=(--run-config "$override")
done
flwr run "$APP_DIR" local-deployment --stream "${overrides[@]}"
