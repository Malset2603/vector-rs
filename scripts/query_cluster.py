#!/usr/bin/env python3
"""Distributed Cluster Query Utility for VectorRS.

This script generates a synthetic query vector with the specified dimension and
dispatches an ANN search request to the VectorRS Coordinator service (127.0.0.1:50050)
via `grpcurl`. It parses the response JSON and displays formatted search results,
including cluster shard distribution, query latency, and ranked nearest neighbors.

Usage Examples:
    # Run a query with default parameters (128-dim, top-5, L2 distance)
    python scripts/query_cluster.py

    # Query using Cosine Similarity with top-10 neighbors on 128-dimensional vectors
    python scripts/query_cluster.py --metric cosine --dimension 128 --k 10

    # Query using Manhattan (L1) distance
    python scripts/query_cluster.py --metric manhattan --dimension 128 --k 5

    # Query using Weighted Jaccard distance with custom dimension
    python scripts/query_cluster.py --metric jaccard --dimension 256 --k 10
"""

import argparse
import io
import json
import random
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

METRICS_MAP = {
    "l2": "DISTANCE_METRIC_L2_SQUARED",
    "l2_squared": "DISTANCE_METRIC_L2_SQUARED",
    "euclidean": "DISTANCE_METRIC_L2_SQUARED",
    "dot": "DISTANCE_METRIC_DOT_PRODUCT",
    "dot_product": "DISTANCE_METRIC_DOT_PRODUCT",
    "inner_product": "DISTANCE_METRIC_DOT_PRODUCT",
    "cosine": "DISTANCE_METRIC_COSINE_SIMILARITY",
    "cosine_similarity": "DISTANCE_METRIC_COSINE_SIMILARITY",
    "cos": "DISTANCE_METRIC_COSINE_SIMILARITY",
    "manhattan": "DISTANCE_METRIC_MANHATTAN",
    "l1": "DISTANCE_METRIC_MANHATTAN",
    "minkowski": "DISTANCE_METRIC_MINKOWSKI",
    "lp": "DISTANCE_METRIC_MINKOWSKI",
    "chebyshev": "DISTANCE_METRIC_CHEBYSHEV",
    "linf": "DISTANCE_METRIC_CHEBYSHEV",
    "hamming": "DISTANCE_METRIC_HAMMING",
    "l0": "DISTANCE_METRIC_HAMMING",
    "mahalanobis": "DISTANCE_METRIC_MAHALANOBIS",
    "jaccard": "DISTANCE_METRIC_JACCARD",
    "tanimoto": "DISTANCE_METRIC_JACCARD",
    "hellinger": "DISTANCE_METRIC_HELLINGER",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the VectorRS distributed cluster coordinator using grpcurl."
    )
    parser.add_argument(
        "--dimension",
        "-d",
        type=int,
        default=128,
        help="Dimension of the query vector (default: 128)",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="l2",
        choices=list(METRICS_MAP.keys()),
        help="Distance metric to evaluate (default: l2)",
    )
    parser.add_argument(
        "--coordinator",
        "-c",
        type=str,
        default="127.0.0.1:50050",
        help="Coordinator endpoint address (default: 127.0.0.1:50050)",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=50,
        help="Size of dynamic candidate list for HNSW search (default: 50)",
    )
    args = parser.parse_args()

    proto_metric = METRICS_MAP.get(args.metric.lower(), "DISTANCE_METRIC_L2_SQUARED")

    print(
        f"[*] Generating random query vector "
        f"(Dimension: {args.dimension}, Top-K: {args.k}, Metric: {proto_metric})..."
    )
    query_vector = [random.uniform(-1.0, 1.0) for _ in range(args.dimension)]

    payload = {
        "query_vector": query_vector,
        "k": args.k,
        "ef_search": args.ef_search,
        "metric": proto_metric,
    }

    payload_file = "temp_query.json"
    with open(payload_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"[*] Payload saved to {payload_file}")
    print(f"[*] Dispatching query to Coordinator ({args.coordinator}) via grpcurl...\n")

    grpcurl_cmd = r".\grpcurl" if sys.platform == "win32" else "grpcurl"

    cmd = [
        grpcurl_cmd,
        "-plaintext",
        "-import-path",
        "proto",
        "-proto",
        "vector_service.proto",
        "-d",
        "@",
        args.coordinator,
        "vector_proto.VectorCoordinatorService/SearchCluster",
    ]

    try:
        with open(payload_file, encoding="utf-8") as f:
            result = subprocess.run(
                cmd, stdin=f, capture_output=True, text=True, check=False
            )

        if result.returncode == 0:
            print("=== SEARCH RESULTS ===")
            try:
                data = json.loads(result.stdout)
                results = data.get("results", [])
                successful_shards = data.get("successfulShards", 0)
                total_shards = data.get("totalQueriedShards", 0)
                latency = data.get("queryLatencyMicros", "0")
                print(
                    f"[*] Successful shards: {successful_shards}/{total_shards} | "
                    f"Latency: {latency} µs | Total matches: {len(results)}"
                )
                if results:
                    print(
                        "\n{:<6} {:<12} {:<16} {:<8}".format(
                            "Rank", "Vector ID", "Score/Distance", "Shard"
                        )
                    )
                    print("-" * 46)
                    for idx, item in enumerate(results, start=1):
                        print(
                            "{:<6} {:<12} {:<16.6f} {:<8}".format(
                                idx,
                                item.get("id", 0),
                                float(item.get("distance", 0.0)),
                                item.get("shardId", 0),
                            )
                        )
                else:
                    print("[!] Warning: No vector results returned.")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(result.stdout)
        else:
            print("=== REQUEST REJECTED / ERROR ===")
            print(result.stderr.strip())
    except FileNotFoundError:
        print(
            f"Error: Executable '{grpcurl_cmd}' not found. Please ensure grpcurl is available in your PATH or current directory."
        )
    finally:
        Path(payload_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
