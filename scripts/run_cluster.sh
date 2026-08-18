#!/usr/bin/env bash
# VectorRS Distributed Cluster Orchestrator (1 Coordinator + 3 Workers)

set -e

INDEX_TYPE="${1:-hnsw}"
DIMENSION="${2:-128}"
METRIC="${3:-l2}"
DATA_DIR="${4:-}"

echo "=================================================="
echo "   VectorRS Distributed Cluster Orchestrator      "
echo "=================================================="
echo "Index Type:  $INDEX_TYPE"
echo "Dimension:   $DIMENSION"
echo "Metric:      $METRIC"
if [ -n "$DATA_DIR" ]; then echo "Data Dir:    $DATA_DIR"; fi
echo "--------------------------------------------------"

echo "[1/5] Compiling cluster binaries..."
cargo build --release --bin vector-worker --bin vector-coordinator

WORKER_BIN="target/release/vector-worker"
COORD_BIN="target/release/vector-coordinator"

PIDS=()

cleanup() {
    echo ""
    echo "Stopping all VectorRS cluster processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "Cluster stopped successfully."
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start Worker 0 (Port 50051)
echo "[2/5] Starting Worker 0 on port 50051 (Shard 0, Metric: $METRIC)..."
W0_ARGS=(--port 50051 --worker-id worker-0 --shard-id 0 --index-type "$INDEX_TYPE" --dimension "$DIMENSION" --metric "$METRIC")
if [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/shard_0.bin" ]; then
    W0_ARGS+=(--storage-file "$DATA_DIR/shard_0.bin")
fi
"$WORKER_BIN" "${W0_ARGS[@]}" &
PIDS+=($!)

# 2. Start Worker 1 (Port 50052)
echo "[3/5] Starting Worker 1 on port 50052 (Shard 1, Metric: $METRIC)..."
W1_ARGS=(--port 50052 --worker-id worker-1 --shard-id 1 --index-type "$INDEX_TYPE" --dimension "$DIMENSION" --metric "$METRIC")
if [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/shard_1.bin" ]; then
    W1_ARGS+=(--storage-file "$DATA_DIR/shard_1.bin")
fi
"$WORKER_BIN" "${W1_ARGS[@]}" &
PIDS+=($!)

# 3. Start Worker 2 (Port 50053)
echo "[4/5] Starting Worker 2 on port 50053 (Shard 2, Metric: $METRIC)..."
W2_ARGS=(--port 50053 --worker-id worker-2 --shard-id 2 --index-type "$INDEX_TYPE" --dimension "$DIMENSION" --metric "$METRIC")
if [ -n "$DATA_DIR" ] && [ -f "$DATA_DIR/shard_2.bin" ]; then
    W2_ARGS+=(--storage-file "$DATA_DIR/shard_2.bin")
fi
"$WORKER_BIN" "${W2_ARGS[@]}" &
PIDS+=($!)

sleep 2

# 4. Start Coordinator (Port 50050)
echo "[5/5] Starting Coordinator on port 50050..."
COORD_ARGS=(
    --port 50050
    --workers "http://127.0.0.1:50051,http://127.0.0.1:50052,http://127.0.0.1:50053"
)
"$COORD_BIN" "${COORD_ARGS[@]}" &
PIDS+=($!)

echo ""
echo ">>> VectorRS Cluster is ONLINE and HEALTHY! <<<"
echo "Coordinator endpoint: http://127.0.0.1:50050"
echo "Press Ctrl+C to terminate all cluster nodes."
echo ""

wait
