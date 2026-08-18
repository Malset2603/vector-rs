# VectorRS

<p align="center">
  <strong>A High-Performance, Distributed Vector Search Engine & Clustering Framework built in Rust.</strong>
</p>

VectorRS is an end-to-end distributed Approximate Nearest Neighbor (ANN) search engine and HPC training framework. It bridges the gap between AI information retrieval and High-Performance Computing (HPC) by leveraging raw metal performance through SIMD intrinsics, NVIDIA CUDA multi-GPU acceleration, and Distributed Memory MPI synchronization.

## Key Features

* **Extreme Hardware Acceleration**: 
  * **CPU SIMD Engine**: Exploits AVX-512, AVX2+FMA, and NEON instruction sets for ultra-low latency vector math.
  * **NVIDIA CUDA DDP**: Multi-GPU Distributed Data Parallel (DDP) for batch k-NN search and k-Means clustering using pinned memory and asynchronous streams.
* **State-of-the-Art Indexing**: Features multi-layer **HNSW** (Hierarchical Navigable Small World), **IVF-PQ** (Inverted File with Product Quantization), and exact **Flat** brute-force search.
* **10 Mathematical Distance Metrics**: Fully supports Squared L2, Dot Product, Cosine Similarity, Manhattan ($L_1$), Minkowski ($L_3$), Chebyshev ($L_\infty$), Hamming ($L_0$), Mahalanobis, Weighted Jaccard, and Hellinger distances natively across CPU, GPU, and Network boundaries.
* **Distributed Microservices**: A resilient **gRPC Scatter-Gather** topology distributing queries across horizontally scalable Shard Worker Nodes to a centralized Coordinator.
* **CUDA-Aware MPI Training**: Horizontally scaling k-Means clustering using `MPI_Allreduce` with Direct VRAM-to-NIC DMA (Zero Host CPU Staging) for ultra-fast offline index building.
* **Zero-Copy Storage**: High-throughput memory-mapped (`MmapStorage`) backend natively aligned to 64-byte CPU cache lines.

## Project Architecture (Workspace Crates)

| Crate | Description |
|---|---|
| `vector-simd` | CPU intrinsics vectorization engine (AVX2/AVX-512) for lightning-fast distance evaluations. |
| `vector-cuda` | NVIDIA CUDA C++ kernels, FFI bindings, and MultiGpuContext for DDP acceleration. |
| `vector-index` | ANN algorithms (HNSW, IVF-PQ) and storage abstractions (Mmap/Heap). |
| `vector-mpi` | Distributed k-Means training using `rsmpi` and collective synchronization. |
| `vector-proto` | Protocol Buffer (`.proto`) gRPC schemas and auto-generated Rust bindings. |
| `vector-worker` | Sharded worker microservice hosting the index, SIMD engine, and CUDA streams. |
| `vector-coordinator` | Ingress router, scattering queries to workers, and merging Top-K results globally. |

## Quick Start

### 1. Prerequisites
* **Rust Toolchain**: 1.80+ (Edition 2024).
* **Protobuf Compiler**: `protoc` installed in `PATH`.
* **NVIDIA CUDA Toolkit**: 12.x+ (Optional, for GPU acceleration).
* **MPI SDK**: OpenMPI / MS-MPI (Optional, for distributed training).

### 2. Manual CUDA PTX Compilation (Optional, for Cloud/Headless Environments)
If your execution environment fails to compile `.cu` files automatically during `cargo build` because it cannot locate `nvcc`, you should manually compile them first. This ensures the environment uses the compiled `.ptx` files with maximum hardware optimization. Run the following commands:

```bash
nvcc --ptx -O3 --use_fast_math --extra-device-vectorization crates/vector-cuda/src/kernels/kmeans.cu -o crates/vector-cuda/src/kernels/kmeans.ptx
nvcc --ptx -O3 --use_fast_math --extra-device-vectorization crates/vector-cuda/src/kernels/knn.cu -o crates/vector-cuda/src/kernels/knn.ptx
```

### 3. Generate Synthetic Dataset
Generate a random dataset of 10,000 vectors (128-dimensional) split into 3 shards:
```bash
python scripts/generate_dataset.py -n 10000 -d 128 -s 3 -o data
```

### 4. Spin Up the Cluster
Run the distributed cluster using the provided scripts (automatically spawning 1 Coordinator and 3 Workers).

**Windows (PowerShell):**
```powershell
.\scripts\run_cluster.ps1 -IndexType hnsw -Dimension 128 -Metric cosine -DataDir data -Release
```

**Linux / macOS (Bash):**
```bash
./scripts/run_cluster.sh hnsw 128 cosine data
```

### 5. Execute a Vector Search Query
You can query the cluster using our Python client:
```bash
python scripts/query_cluster.py --metric cosine --dimension 128 --k 5
```
Or use `grpcurl`:
```bash
grpcurl -plaintext -import-path proto -proto vector_service.proto \
  -d '{ "query_vector": [0.12, -0.34, 0.56], "k": 5, "ef_search": 50, "metric": "DISTANCE_METRIC_COSINE_SIMILARITY" }' \
  127.0.0.1:50050 vector_proto.VectorCoordinatorService/SearchCluster
```

## Benchmarks

VectorRS is rigorously benchmarked to ensure maximum HPC efficiency. The benchmark suite includes Multi-Core Scaling, CUDA Hardware Acceleration, and MPI-CPU vs CUDA-Aware MPI comparisons.

To automatically execute all 6 benchmark suites, you can run the Jupyter Notebook:
* `notebooks/kaggle_run_benchmark.ipynb`

Or run them individually via Cargo:
```bash
# CPU SIMD Throughput
cargo bench -p vector-simd

# Multi-GPU CUDA Performance
cargo bench -p vector-cuda --bench cuda_bench
```
*Note: Visual benchmark results (SVG plots) are available in the `scripts/benchmarks/` directory.*

## Advanced Usage (Rust Native APIs)

### HNSW Graph Construction
```rust
use vector_index::{DistanceMetric, HnswIndex, HnswConfig};
use vector_index::storage::HeapStorage;

let storage = HeapStorage::new(128); // Load your vectors
let index = HnswIndex::build_parallel_with_config(
    storage, 
    HnswConfig::default(), 
    DistanceMetric::CosineSimilarity // Choose from 10 metrics
);

let query = vec![0.5; 128];
let nearest_neighbors = index.search(&query, 10, 64);
```

### Multi-GPU Distributed Data Parallel (DDP) k-NN
```rust
use vector_cuda::{DistributedKnnEngine, GpuShardMode, CudaError};
use vector_index::DistanceMetric;

// Try to initialize DDP Engine, fails fast if physical GPUs are missing
let ddp_knn = DistributedKnnEngine::try_new(
    &dataset,
    128,          // dimensions
    2,            // request 2 physical GPUs
    GpuShardMode::Sharded,
    DistanceMetric::Manhattan,
)?;

let query = vec![0.1; 128];
let top_k = ddp_knn.search(&query, 10);
```

### CPU SIMD Distance Acceleration
```rust
use vector_simd::{DistanceEngine, DistanceMetric};

let engine = DistanceEngine::auto(); // Automatically detects AVX-512 / AVX2
let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let dist = engine.l2_squared(&a, &b);
let sim = engine.cosine_similarity(&a, &b);
```

## License
This project is licensed under the MIT License.
