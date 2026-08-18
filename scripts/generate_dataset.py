#!/usr/bin/env python3
"""Synthetic Binary Dataset & Ground Truth Generator for VectorRS.

This script generates synthetic multi-dimensional vector datasets and partitions
them into binary shard files formatted for VectorRS zero-copy memory-mapped storage
(`MmapStorage`). It also generates a standalone query vector set and computes exact
brute-force K-NN ground truth matches in JSON format for recall evaluation.

Supported Distributions:
    - gaussian : Clustered Gaussian distribution around randomized cluster centroids.
    - sphere   : Uniformly sampled points projected onto the unit hypersphere surface.
    - uniform  : Independent uniformly distributed coordinates in [-1.0, 1.0].

File Format Specification (`.bin` files):
    - Bytes 0..8   : Magic number 0x56454352_53544F52 ("VECRSTOR", uint64 little-endian)
    - Bytes 8..16  : Number of vectors N (uint64 little-endian)
    - Bytes 16..24 : Vector dimension D (uint64 little-endian)
    - Bytes 24..   : Contiguous IEEE-754 float32 vector data (N x D floats, little-endian)

Usage Examples:
    # Generate default dataset (10,000 vectors, 128-dim, 3 shards in ./data)
    python scripts/generate_dataset.py

    # Generate 50,000 vectors with 256 dimensions split across 4 worker shards
    python scripts/generate_dataset.py -n 50000 -d 256 -s 4 -o data

    # Generate normalized vectors on unit sphere for Cosine similarity benchmarks
    python scripts/generate_dataset.py -n 20000 -d 128 --distribution sphere -q 200 -k 10 -o data
"""

import argparse
import io
import json
import math
import operator
import os
import random
import struct
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MAGIC_VECRSTOR = 0x56454352_53544F52


def write_vecrstor_binary(
    filepath: str, vectors: list[list[float]], dimension: int
) -> None:
    """Writes a list of float vectors to a VectorRS .bin file."""
    num_vectors = len(vectors)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        # Write 24-byte header
        header = struct.pack("<QQQ", MAGIC_VECRSTOR, num_vectors, dimension)
        f.write(header)

        # Write vector data
        f.writelines(struct.pack(f"<{dimension}f", *vec) for vec in vectors)


def generate_vectors(
    num_vectors: int, dimension: int, distribution: str
) -> list[list[float]]:
    """Generates synthetic vectors according to the chosen distribution."""
    vectors = []

    if distribution == "gaussian":
        # Generate 8 distinct cluster centroids
        num_clusters = 8
        centroids = [
            [random.uniform(-10.0, 10.0) for _ in range(dimension)]
            for _ in range(num_clusters)
        ]

        for _ in range(num_vectors):
            c = random.choice(centroids)
            vec = [c[d] + random.gauss(0.0, 1.0) for d in range(dimension)]
            vectors.append(vec)

    elif distribution == "sphere":
        # Uniformly distributed on the unit sphere
        for _ in range(num_vectors):
            raw = [random.gauss(0.0, 1.0) for _ in range(dimension)]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0.0:
                vec = [x / norm for x in raw]
            else:
                vec = [0.0] * dimension
            vectors.append(vec)

    else:  # uniform
        for _ in range(num_vectors):
            vec = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
            vectors.append(vec)

    return vectors


def compute_exact_knn(
    queries: list[list[float]], dataset: list[list[float]], k: int
) -> list[dict]:
    """Computes exact brute-force L2^2 top-K ground truth."""
    ground_truth = []

    for q_idx, q in enumerate(queries):
        distances = []
        for d_idx, d in enumerate(dataset):
            dist_sq = sum((q[i] - d[i]) ** 2 for i in range(len(q)))
            distances.append((d_idx, dist_sq))

        distances.sort(key=operator.itemgetter(1))
        top_k = distances[:k]

        ground_truth.append(
            {
                "query_id": q_idx,
                "top_k": [{"id": d_idx, "distance": dist} for d_idx, dist in top_k],
            }
        )

    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VectorRS Synthetic Dataset & Shard Generator"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data",
        help="Output directory for generated .bin files and ground truth (default: data)",
    )
    parser.add_argument(
        "--num-vectors",
        "-n",
        type=int,
        default=10000,
        help="Total number of dataset vectors to generate (default: 10000)",
    )
    parser.add_argument(
        "--dimension",
        "-d",
        type=int,
        default=128,
        help="Vector dimensionality (default: 128)",
    )
    parser.add_argument(
        "--shards",
        "-s",
        type=int,
        default=3,
        help="Number of shards to partition the dataset into (default: 3)",
    )
    parser.add_argument(
        "--num-queries",
        "-q",
        type=int,
        default=100,
        help="Number of test query vectors to generate (default: 100)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["uniform", "gaussian", "sphere"],
        default="gaussian",
        help="Vector distribution pattern (default: gaussian)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Top-K nearest neighbors for ground truth computation (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset generation (default: 42)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=== VectorRS Dataset Generator ===")
    print(f"Vectors:     {args.num_vectors:,}")
    print(f"Dimension:   {args.dimension}")
    print(f"Shards:      {args.shards}")
    print(f"Queries:     {args.num_queries}")
    print(f"Pattern:     {args.distribution}")
    print(f"Output Dir:  {args.output_dir}")
    print("-" * 35)

    start_time = time.time()

    # 1. Generate full dataset
    print(f"[1/4] Generating {args.num_vectors:,} dataset vectors...")
    dataset = generate_vectors(args.num_vectors, args.dimension, args.distribution)

    # 2. Write full dataset & partitioned shards
    full_path = os.path.join(args.output_dir, "dataset_full.bin")
    write_vecrstor_binary(full_path, dataset, args.dimension)
    print(
        f"  -> Wrote full dataset: {full_path} ({Path(full_path).stat().st_size / 1024:.1f} KB)"
    )

    if args.shards > 1:
        chunk_size = (args.num_vectors + args.shards - 1) // args.shards
        for s in range(args.shards):
            start = s * chunk_size
            end = min(start + chunk_size, args.num_vectors)
            shard_data = dataset[start:end]
            shard_path = os.path.join(args.output_dir, f"shard_{s}.bin")
            write_vecrstor_binary(shard_path, shard_data, args.dimension)
            print(f"  -> Wrote shard {s}: {shard_path} ({len(shard_data)} vectors)")

    # 3. Generate query vectors
    print(f"[2/4] Generating {args.num_queries} query vectors...")
    queries = generate_vectors(args.num_queries, args.dimension, args.distribution)
    query_path = os.path.join(args.output_dir, "queries.bin")
    write_vecrstor_binary(query_path, queries, args.dimension)
    print(f"  -> Wrote queries: {query_path}")

    # 4. Compute ground truth
    print(f"[3/4] Computing exact brute-force Top-{args.top_k} ground truth...")
    gt = compute_exact_knn(queries, dataset, args.top_k)
    gt_path = os.path.join(args.output_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dimension": args.dimension, "k": args.top_k, "queries": gt}, f, indent=2
        )
    print(f"  -> Wrote ground truth: {gt_path}")

    elapsed = time.time() - start_time
    print(f"[4/4] Dataset generation completed in {elapsed:.2f}s!")


if __name__ == "__main__":
    main()
