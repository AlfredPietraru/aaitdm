#!/usr/bin/env bash
# Start a local Flower deployment: one SuperLink (server), NUM_CLIENTS SuperNodes
# (hospitals), then submit the app and stream the server log until it finishes.
#
#   ./run_federated.sh                                  # defaults from app-state/federated_configuration.yaml
#   NUM_CLIENTS=4 ./run_federated.sh
#
# Per-process logs are written to outputs/logs/.
set -euo pipefail

if (( $# > 0 )); then
    echo "This script takes no arguments. Edit app-state/federated_configuration.yaml; use NUM_CLIENTS to override the hospital count." >&2
    exit 2
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$APP_DIR")"
SUPERLINK_PORT="${SUPERLINK_PORT:-8000}"
LOG_DIR="$APP_DIR/outputs/logs"

export PATH="${VENV_BIN:-$PROJECT_DIR/.venv/bin}:$PATH"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
FEDERATED_CONFIG="$PROJECT_DIR/app-state/federated_configuration.yaml"
# Read settings before starting any processes; a malformed YAML must fail here.
config_text="$(python3 "$APP_DIR/fl_isic/configuration.py" "$FEDERATED_CONFIG")"
mapfile -t config_overrides <<< "$config_text"
for setting in "${config_overrides[@]}"; do
    if [[ "$setting" == min-nodes=* ]]; then
        NUM_CLIENTS="${NUM_CLIENTS:-${setting#min-nodes=}}"
    fi
done
if [[ ! "$NUM_CLIENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_CLIENTS must be a positive integer" >&2
    exit 1
fi
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
RUN_APP_DIR=""
cleanup() {
    kill "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
    if [[ -n "$RUN_APP_DIR" ]]; then
        rm -rf -- "$RUN_APP_DIR"
    fi
}
trap cleanup EXIT INT TERM

# Declare all YAML settings in a generated manifest before Flower packages it.
# Staging outside APP_DIR also excludes output logs and other mutable files.
RUN_APP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fl-isic.XXXXXX")"
echo "Generating Flower app configuration from $FEDERATED_CONFIG"
python3 "$APP_DIR/fl_isic/configuration.py" "$FEDERATED_CONFIG" \
    --prepare-app "$APP_DIR" --output-dir "$RUN_APP_DIR" --num-clients "$NUM_CLIENTS"

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

flwr run "$RUN_APP_DIR" local-deployment --stream
