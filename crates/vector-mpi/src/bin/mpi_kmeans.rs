//! CLI binary for VectorRS distributed k-Means index training.
//!
//! Supports two execution modes:
//! - **Local mode** (default): Reads all shard files, concatenates, and trains locally.
//! - **MPI mode** (`--features mpi`): Each MPI rank processes its own shard with
//!   global centroid synchronization via `MPI_Allreduce`.
//!
//! # Usage
//!
//! ```bash
//! # Local single-process baseline:
//! cargo run --release -p vector-mpi --bin mpi_kmeans -- --data-dir data -k 64
//!
//! # MPI distributed (requires MS-MPI / OpenMPI):
//! cargo build --release -p vector-mpi --features mpi
//! mpiexec -n 3 target/release/mpi_kmeans --data-dir data -k 64
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tracing::info;
use vector_index::DistanceMetric;
use vector_index::ivf::IvfPqSerializer;
use vector_index::storage::{MmapStorage, VectorStorage};

#[derive(Parser, Debug)]
#[command(
    name = "mpi_kmeans",
    about = "VectorRS Distributed k-Means Index Trainer (MPI + SIMD + CUDA)"
)]
struct Args {
    /// Directory containing shard_0.bin, shard_1.bin, ... files.
    #[arg(short, long, default_value = "data")]
    data_dir: PathBuf,

    /// Number of coarse clusters (centroids) to train.
    #[arg(short = 'k', long, default_value_t = 64)]
    num_clusters: usize,

    /// Maximum number of Lloyd's iterations.
    #[arg(long, default_value_t = 30)]
    max_iters: usize,

    /// Convergence tolerance on maximum centroid shift (L2²).
    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,

    /// Distance metric for centroid assignment.
    #[arg(short = 'm', long, default_value = "l2")]
    metric: String,

    /// Output filename for the centroid binary file (relative to data_dir).
    #[arg(short, long, default_value = "centroids_global.bin")]
    output: String,

    /// Enable CUDA GPU acceleration for k-Means training (requires --features cuda).
    #[arg(long)]
    use_cuda: bool,

    /// Number of CUDA GPU devices to use for CUDA-Aware training.
    #[arg(short = 'g', long, alias = "gpus", default_value_t = 1)]
    num_gpus: usize,
}

fn parse_metric(s: &str) -> DistanceMetric {
    match s.to_lowercase().as_str() {
        "l2" | "l2_squared" | "euclidean" => DistanceMetric::L2Squared,
        "dot" | "dot_product" | "inner_product" => DistanceMetric::DotProduct,
        "cosine" | "cosine_similarity" | "cos" => DistanceMetric::CosineSimilarity,
        "manhattan" | "l1" => DistanceMetric::Manhattan,
        "minkowski" | "lp" => DistanceMetric::Minkowski,
        "chebyshev" | "linf" => DistanceMetric::Chebyshev,
        "hamming" | "l0" => DistanceMetric::Hamming,
        "mahalanobis" => DistanceMetric::Mahalanobis,
        "jaccard" | "tanimoto" => DistanceMetric::Jaccard,
        "hellinger" => DistanceMetric::Hellinger,
        _ => {
            eprintln!("Unknown metric '{}', defaulting to L2Squared", s);
            DistanceMetric::L2Squared
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let metric = parse_metric(&args.metric);
    let output_path = args.data_dir.join(&args.output);

    // CUDA-Aware mode takes priority when --use-cuda is specified
    #[cfg(feature = "cuda")]
    if args.use_cuda {
        run_cuda_aware_mode(&args, metric, &output_path);
        return;
    }

    #[cfg(not(feature = "cuda"))]
    if args.use_cuda {
        eprintln!("Error: --use-cuda requires building with --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "mpi")]
    {
        run_mpi_mode(&args, metric, &output_path);
    }

    #[cfg(not(feature = "mpi"))]
    {
        run_local_mode(&args, metric, &output_path);
    }
}

/// MPI distributed mode: each rank opens its own shard and collaborates
/// via Allreduce to produce globally consistent centroids.
#[cfg(feature = "mpi")]
fn run_mpi_mode(args: &Args, metric: DistanceMetric, output_path: &std::path::Path) {
    use mpi::traits::*;

    let universe = mpi::initialize().expect("Failed to initialize MPI runtime");
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let shard_path = args.data_dir.join(format!("shard_{}.bin", rank));
    info!(rank, size, shard_path = ?shard_path, "MPI rank starting");

    let storage = MmapStorage::open(&shard_path)
        .unwrap_or_else(|e| panic!("Rank {}: failed to open {:?}: {}", rank, shard_path, e));

    let dimension = storage.dimension();
    let n_local = storage.len();

    info!(
        rank,
        n_local,
        dimension,
        k = args.num_clusters,
        "Loaded shard data"
    );

    let start = Instant::now();
    let result = vector_mpi::fit_distributed(
        &world,
        storage.as_raw_slice(),
        dimension,
        args.num_clusters,
        args.max_iters,
        args.tolerance,
        metric,
    );
    let elapsed = start.elapsed();

    info!(
        rank,
        iterations = result.iterations,
        inertia = result.inertia,
        elapsed_ms = elapsed.as_millis(),
        "k-Means training complete on rank"
    );

    // Only Rank 0 writes the output centroid file
    if rank == 0 {
        IvfPqSerializer::save_centroid_file(
            output_path,
            &result.centroids,
            result.dimension,
            result.k,
            metric,
            result.iterations,
            result.inertia,
        )
        .expect("Failed to save centroid file");

        println!("=== MPI Distributed k-Means Training Complete ===");
        println!("  Ranks:      {}", size);
        println!("  Clusters:   {}", result.k);
        println!("  Dimension:  {}", result.dimension);
        println!("  Iterations: {}", result.iterations);
        println!("  Inertia:    {:.6}", result.inertia);
        println!("  Elapsed:    {:.3}s", elapsed.as_secs_f64());
        println!("  Output:     {}", output_path.display());
    }
}

/// Local single-process mode: reads all shards, concatenates data,
/// and trains centroids using the standard single-node k-Means.
#[cfg(not(feature = "mpi"))]
fn run_local_mode(args: &Args, metric: DistanceMetric, output_path: &std::path::Path) {
    // Auto-discover shard files (shard_0.bin, shard_1.bin, ...)
    let mut shard_files: Vec<PathBuf> = Vec::new();
    for i in 0.. {
        let path = args.data_dir.join(format!("shard_{}.bin", i));
        if path.exists() {
            shard_files.push(path);
        } else {
            break;
        }
    }

    if shard_files.is_empty() {
        eprintln!(
            "Error: No shard files (shard_0.bin, shard_1.bin, ...) found in {:?}",
            args.data_dir
        );
        std::process::exit(1);
    }

    println!("=== Local Single-Process k-Means Training ===");
    println!("  Shards found: {}", shard_files.len());

    // Load and concatenate all shards into a single contiguous buffer
    let mut all_data: Vec<f32> = Vec::new();
    let mut dimension = 0usize;

    for shard_path in &shard_files {
        let storage = MmapStorage::open(shard_path)
            .unwrap_or_else(|e| panic!("Failed to open {:?}: {}", shard_path, e));

        if dimension == 0 {
            dimension = storage.dimension();
        } else {
            assert_eq!(
                dimension,
                storage.dimension(),
                "Dimension mismatch across shard files"
            );
        }

        all_data.extend_from_slice(storage.as_raw_slice());
        info!(shard = ?shard_path, vectors = storage.len(), "Loaded shard");
    }

    let n_total = all_data.len() / dimension;
    println!("  Total vectors: {}", n_total);
    println!("  Dimension:     {}", dimension);
    println!("  Clusters (k):  {}", args.num_clusters);
    println!("  Max iters:     {}", args.max_iters);
    println!("  Metric:        {:?}", metric);
    println!("  ---");

    let start = Instant::now();
    let result = vector_mpi::fit_local(
        &all_data,
        dimension,
        args.num_clusters,
        args.max_iters,
        args.tolerance,
        metric,
    );
    let elapsed = start.elapsed();

    IvfPqSerializer::save_centroid_file(
        output_path,
        &result.centroids,
        result.dimension,
        result.k,
        metric,
        result.iterations,
        result.inertia,
    )
    .expect("Failed to save centroid file");

    println!("  Iterations:    {}", result.iterations);
    println!("  Inertia:       {:.6}", result.inertia);
    println!("  Elapsed:       {:.3}s", elapsed.as_secs_f64());
    println!("  Output:        {}", output_path.display());
    println!("=== Training Complete ===");
}

/// CUDA-Aware MPI mode: GPU-accelerated k-Means index training.
///
/// Uses GPU hardware kernels via `vector-cuda` (with DDP multi-GPU scaling
/// when `num_gpus > 1`) for high-throughput centroid fitting.
#[cfg(feature = "cuda")]
fn run_cuda_aware_mode(args: &Args, metric: DistanceMetric, output_path: &std::path::Path) {
    let mut shard_files: Vec<PathBuf> = Vec::new();
    for i in 0.. {
        let path = args.data_dir.join(format!("shard_{}.bin", i));
        if path.exists() {
            shard_files.push(path);
        } else {
            break;
        }
    }

    if shard_files.is_empty() {
        eprintln!(
            "Error: No shard files (shard_0.bin, shard_1.bin, ...) found in {:?}",
            args.data_dir
        );
        std::process::exit(1);
    }

    println!("=== CUDA-Aware k-Means Training ===");
    println!("  Shards found:  {}", shard_files.len());
    println!("  CUDA GPUs:     {}", args.num_gpus);

    let mut all_data: Vec<f32> = Vec::new();
    let mut dimension = 0usize;

    for shard_path in &shard_files {
        let storage = MmapStorage::open(shard_path)
            .unwrap_or_else(|e| panic!("Failed to open {:?}: {}", shard_path, e));

        if dimension == 0 {
            dimension = storage.dimension();
        } else {
            assert_eq!(
                dimension,
                storage.dimension(),
                "Dimension mismatch across shard files"
            );
        }

        all_data.extend_from_slice(storage.as_raw_slice());
        info!(shard = ?shard_path, vectors = storage.len(), "Loaded shard");
    }

    let n_total = all_data.len() / dimension;
    println!("  Total vectors: {}", n_total);
    println!("  Dimension:     {}", dimension);
    println!("  Clusters (k):  {}", args.num_clusters);
    println!("  Max iters:     {}", args.max_iters);
    println!("  Metric:        {:?}", metric);
    println!("  ---");

    let start = Instant::now();
    let result = vector_mpi::fit_cuda_aware(
        &all_data,
        dimension,
        args.num_clusters,
        args.max_iters,
        args.tolerance,
        metric,
        args.num_gpus,
    );
    let elapsed = start.elapsed();

    IvfPqSerializer::save_centroid_file(
        output_path,
        &result.centroids,
        result.dimension,
        result.k,
        metric,
        result.iterations,
        result.inertia,
    )
    .expect("Failed to save centroid file");

    println!("  Iterations:    {}", result.iterations);
    println!("  Inertia:       {:.6}", result.inertia);
    println!("  Elapsed:       {:.3}s", elapsed.as_secs_f64());
    println!("  Output:        {}", output_path.display());
    println!("=== Training Complete ===");
}
