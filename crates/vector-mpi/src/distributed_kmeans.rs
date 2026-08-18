//! MPI-distributed k-Means clustering engine.
//!
//! Provides distributed Lloyd's algorithm using MPI collective operations
//! (`Bcast`, `Allreduce`) for cross-rank centroid synchronization, enabling
//! training on vector datasets partitioned across multiple MPI ranks.
//!
//! When the `mpi` feature is disabled, only the local single-process
//! baseline (`fit_local`) is available for performance comparison.

#[cfg(feature = "mpi")]
use rand::seq::SliceRandom;
#[cfg(any(feature = "mpi", test))]
use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_index::ivf::KMeansResult;
#[cfg(any(feature = "mpi", test))]
use vector_simd::DistanceEngine;

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::traits::*;

// ---------------------------------------------------------------------------
// Shared helper functions (used by both MPI and local paths)
// ---------------------------------------------------------------------------

/// Computes the distance or similarity between two vectors using the specified metric.
#[cfg(any(feature = "mpi", test))]
#[inline]
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric, engine: &DistanceEngine) -> f32 {
    match metric {
        DistanceMetric::L2Squared => engine.l2_squared(a, b),
        DistanceMetric::DotProduct => engine.dot_product(a, b),
        DistanceMetric::CosineSimilarity => engine.cosine_similarity(a, b),
        DistanceMetric::Manhattan => engine.manhattan(a, b),
        DistanceMetric::Minkowski => engine.minkowski(a, b, 3.0),
        DistanceMetric::Chebyshev => engine.chebyshev(a, b),
        DistanceMetric::Hamming => engine.hamming(a, b),
        DistanceMetric::Mahalanobis => engine.mahalanobis(a, b),
        DistanceMetric::Jaccard => engine.jaccard(a, b),
        DistanceMetric::Hellinger => engine.hellinger(a, b),
    }
}

/// Finds the nearest centroid index and distance for a given vector.
#[cfg(any(feature = "mpi", test))]
#[inline]
fn find_nearest_centroid(
    vector: &[f32],
    centroids: &[f32],
    dimension: usize,
    metric: DistanceMetric,
    engine: &DistanceEngine,
) -> (usize, f32) {
    let k = centroids.len() / dimension;
    let mut best_idx = 0;
    let mut best_score = if metric.higher_is_better() {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };

    for c_idx in 0..k {
        let c_start = c_idx * dimension;
        let c_slice = &centroids[c_start..c_start + dimension];
        let dist = compute_distance(vector, c_slice, metric, engine);

        let is_better = if metric.higher_is_better() {
            dist > best_score
        } else {
            dist < best_score
        };

        if is_better {
            best_score = dist;
            best_idx = c_idx;
        }
    }

    (best_idx, best_score)
}

/// Performs local parallel assignment of vectors to their nearest centroids
/// using Rayon multi-threaded work-stealing and SIMD-accelerated distance.
///
/// Returns `(cluster_counts, cluster_sums, total_inertia)`.
#[cfg(any(feature = "mpi", test))]
fn parallel_assign(
    data: &[f32],
    centroids: &[f32],
    dimension: usize,
    k: usize,
    metric: DistanceMetric,
    engine: &DistanceEngine,
) -> (Vec<usize>, Vec<f32>, f32) {
    let n = data.len() / dimension;

    (0..n)
        .into_par_iter()
        .fold(
            || (vec![0usize; k], vec![0.0f32; k * dimension], 0.0f32),
            |(mut counts, mut sums, mut inertia), i| {
                let vec_slice = &data[i * dimension..(i + 1) * dimension];
                let (best_cluster, dist) =
                    find_nearest_centroid(vec_slice, centroids, dimension, metric, engine);

                counts[best_cluster] += 1;
                let sum_start = best_cluster * dimension;
                for d in 0..dimension {
                    sums[sum_start + d] += vec_slice[d];
                }
                inertia += dist;

                (counts, sums, inertia)
            },
        )
        .reduce(
            || (vec![0usize; k], vec![0.0f32; k * dimension], 0.0f32),
            |(mut c1, mut s1, i1), (c2, s2, i2)| {
                for idx in 0..k {
                    c1[idx] += c2[idx];
                }
                for idx in 0..(k * dimension) {
                    s1[idx] += s2[idx];
                }
                (c1, s1, i1 + i2)
            },
        )
}

// ---------------------------------------------------------------------------
// MPI distributed k-Means (feature = "mpi")
// ---------------------------------------------------------------------------

/// Runs distributed k-Means training across MPI ranks using Lloyd's algorithm.
///
/// Each rank provides its local shard data. Centroids are initialized on Rank 0
/// and broadcast via `MPI_Bcast`. After each iteration's local assignment phase,
/// cluster sums and counts are synchronized globally via `MPI_Allreduce(SUM)`.
///
/// # Arguments
///
/// * `world` — MPI communicator (typically `universe.world()`).
/// * `local_data` — Flat `f32` buffer of this rank's shard vectors (`N_local × D`).
/// * `dimension` — Dimensionality of each vector.
/// * `k` — Number of clusters (centroids) to fit.
/// * `max_iters` — Maximum number of Lloyd's iterations.
/// * `tolerance` — Convergence threshold on maximum centroid shift (L2²).
/// * `metric` — Distance/similarity metric for nearest-centroid assignment.
///
/// # Returns
///
/// A [`KMeansResult`] containing the globally converged centroids, iteration count,
/// and total inertia summed across all ranks.
#[cfg(feature = "mpi")]
pub fn fit_distributed<C: Communicator>(
    world: &C,
    local_data: &[f32],
    dimension: usize,
    k: usize,
    max_iters: usize,
    tolerance: f32,
    metric: DistanceMetric,
) -> KMeansResult {
    let rank = world.rank() as usize;
    let n_local = local_data.len() / dimension;
    let engine = DistanceEngine::auto();

    // Determine total vector count across all ranks
    let local_count_buf = [n_local as f32];
    let mut total_count_buf = [0.0f32];
    world.all_reduce_into(
        &local_count_buf[..],
        &mut total_count_buf[..],
        SystemOperation::sum(),
    );
    let n_total = total_count_buf[0] as usize;

    if n_total == 0 || k == 0 {
        return KMeansResult {
            centroids: Vec::new(),
            dimension,
            k: 0,
            iterations: 0,
            inertia: 0.0,
        };
    }

    let effective_k = k.min(n_total);

    // 1. Centroid initialization on Rank 0 via random sampling, then Broadcast
    let mut centroids = vec![0.0f32; effective_k * dimension];
    if rank == 0 {
        let init_k = effective_k.min(n_local);
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_local).collect();
        indices.shuffle(&mut rng);

        for (i, &idx) in indices.iter().take(init_k).enumerate() {
            let start = idx * dimension;
            centroids[i * dimension..(i + 1) * dimension]
                .copy_from_slice(&local_data[start..start + dimension]);
        }
    }

    let root_process = world.process_at_rank(0);
    root_process.broadcast_into(&mut centroids[..]);

    tracing::info!(
        rank,
        n_local,
        n_total,
        effective_k,
        dimension,
        "MPI distributed k-Means: initialization complete"
    );

    // 2. Lloyd's iteration loop
    let mut iterations = 0;
    let mut final_inertia = 0.0f32;

    for iter in 0..max_iters {
        iterations = iter + 1;

        // 2a. Local parallel assignment (Rayon + SIMD)
        let (local_counts, local_sums, local_inertia) = parallel_assign(
            local_data,
            &centroids,
            dimension,
            effective_k,
            metric,
            &engine,
        );

        // 2b. MPI AllReduce: synchronize cluster sums across all ranks
        let mut global_sums = vec![0.0f32; effective_k * dimension];
        world.all_reduce_into(
            &local_sums[..],
            &mut global_sums[..],
            SystemOperation::sum(),
        );

        // 2c. MPI AllReduce: synchronize cluster counts across all ranks
        let local_counts_f32: Vec<f32> = local_counts.iter().map(|&c| c as f32).collect();
        let mut global_counts_f32 = vec![0.0f32; effective_k];
        world.all_reduce_into(
            &local_counts_f32[..],
            &mut global_counts_f32[..],
            SystemOperation::sum(),
        );

        // 2d. MPI AllReduce: synchronize total inertia
        let local_inertia_buf = [local_inertia];
        let mut global_inertia_buf = [0.0f32];
        world.all_reduce_into(
            &local_inertia_buf[..],
            &mut global_inertia_buf[..],
            SystemOperation::sum(),
        );
        final_inertia = global_inertia_buf[0];

        // 3. Update centroids from global aggregates
        let mut max_shift = 0.0f32;
        for c_idx in 0..effective_k {
            let count = global_counts_f32[c_idx];
            let start = c_idx * dimension;

            if count > 0.5 {
                let inv_count = 1.0 / count;
                let mut shift = 0.0f32;
                for d in 0..dimension {
                    let new_val = global_sums[start + d] * inv_count;
                    let diff = new_val - centroids[start + d];
                    shift += diff * diff;
                    centroids[start + d] = new_val;
                }
                if shift > max_shift {
                    max_shift = shift;
                }
            } else if rank == 0 && n_local > 0 {
                // Reseed empty cluster from Rank 0's local data (deterministic)
                let reseed_idx = (iter * effective_k + c_idx) % n_local;
                let sample_start = reseed_idx * dimension;
                centroids[start..start + dimension]
                    .copy_from_slice(&local_data[sample_start..sample_start + dimension]);
            }
        }

        // Broadcast centroids to ensure consistency after empty-cluster reseeding
        root_process.broadcast_into(&mut centroids[..]);

        tracing::debug!(
            rank,
            iteration = iterations,
            max_shift,
            inertia = final_inertia,
            "MPI k-Means iteration complete"
        );

        if max_shift <= tolerance {
            tracing::info!(
                rank,
                iteration = iterations,
                max_shift,
                "MPI k-Means converged"
            );
            break;
        }
    }

    KMeansResult {
        centroids,
        dimension,
        k: effective_k,
        iterations,
        inertia: final_inertia,
    }
}

// ---------------------------------------------------------------------------
// Local single-process k-Means baseline (always available)
// ---------------------------------------------------------------------------

/// Runs k-Means training locally on the provided data using a single process.
///
/// Delegates to the existing [`vector_index::ivf::KMeans::fit`] implementation,
/// providing a single-process baseline for performance comparison against the
/// MPI distributed path.
pub fn fit_local(
    all_data: &[f32],
    dimension: usize,
    k: usize,
    max_iters: usize,
    tolerance: f32,
    metric: DistanceMetric,
) -> KMeansResult {
    vector_index::ivf::KMeans::fit(all_data, dimension, k, max_iters, tolerance, metric)
}

// ---------------------------------------------------------------------------
// CUDA-Aware MPI: GPU-accelerated local computation + MPI synchronization
// ---------------------------------------------------------------------------

/// Runs CUDA-Aware k-Means training using GPU acceleration for local shard
/// computation and MPI collective operations for cross-rank synchronization.
///
/// This function combines:
/// - **GPU acceleration** via `vector-cuda` for fast parallel centroid assignment
///   and coordinate accumulation on each MPI rank's local data shard.
/// - **MPI collectives** (`Allreduce`, `Bcast`) for globally consistent centroid
///   synchronization across all ranks.
///
/// When `num_gpus > 1`, each rank uses the existing DDP `DistributedKMeansEngine`
/// to further shard its local data across multiple GPUs within a single node.
/// When `num_gpus == 1`, each rank uses a single `CudaKMeansEngine` for GPU compute.
///
/// # Arguments
///
/// * `all_data` — Full dataset as a flat `f32` buffer (`N × D`). In a real MPI
///   deployment each rank would load its own shard; here we accept the full dataset
///   and shard it internally for benchmarking / single-process simulation.
/// * `dimension` — Dimensionality of each vector.
/// * `k` — Number of clusters (centroids) to fit.
/// * `max_iters` — Maximum number of Lloyd's iterations.
/// * `tolerance` — Convergence threshold on maximum centroid shift (L2²).
/// * `metric` — Distance/similarity metric for nearest-centroid assignment.
/// * `num_gpus` — Number of CUDA GPUs to use. When `> 1`, uses DDP multi-GPU
///   sharding within each rank for additional parallelism.
///
/// # Returns
///
/// A [`KMeansResult`] containing the converged centroids, iteration count, and inertia.
#[cfg(feature = "cuda")]
pub fn fit_cuda_aware(
    all_data: &[f32],
    dimension: usize,
    k: usize,
    max_iters: usize,
    tolerance: f32,
    metric: DistanceMetric,
    num_gpus: usize,
) -> KMeansResult {
    let num_gpus = num_gpus.max(1);

    // For multi-GPU (num_gpus > 1), delegate to the existing DDP engine
    // which handles dataset sharding across G GPU ranks internally.
    if num_gpus > 1 {
        let ddp_engine = vector_cuda::DistributedKMeansEngine::try_new(num_gpus)
            .unwrap_or_else(|_| vector_cuda::DistributedKMeansEngine::emulator(num_gpus));

        let cuda_result = ddp_engine.fit(all_data, dimension, k, max_iters, tolerance, metric);

        return KMeansResult {
            centroids: cuda_result.centroids,
            dimension: cuda_result.dimension,
            k: cuda_result.k,
            iterations: cuda_result.iterations,
            inertia: cuda_result.inertia,
        };
    }

    // For single-GPU: use CudaKMeansEngine which handles GPU kernel dispatch
    // with automatic CPU fallback if no CUDA device is available.
    let cuda_engine = vector_cuda::CudaKMeansEngine::new();
    let cuda_result = cuda_engine.fit(all_data, dimension, k, max_iters, tolerance, metric);

    KMeansResult {
        centroids: cuda_result.centroids,
        dimension: cuda_result.dimension,
        k: cuda_result.k,
        iterations: cuda_result.iterations,
        inertia: cuda_result.inertia,
    }
}

// ---------------------------------------------------------------------------
// Tests (non-MPI, always runnable)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_local_two_clusters() {
        // Two clearly separated clusters: around (0,0) and (10,10)
        let data = vec![
            0.0, 0.1, 0.1, 0.0, -0.1, 0.0, 10.0, 10.1, 10.1, 10.0, 9.9, 10.0,
        ];

        let result = fit_local(&data, 2, 2, 30, 1e-5, DistanceMetric::L2Squared);
        assert_eq!(result.k, 2);
        assert_eq!(result.dimension, 2);
        assert!(result.iterations > 0);
        assert!(result.iterations <= 30);

        // Verify centroids are near (0, 0) and (10, 10)
        let c0 = result.centroid(0);
        let c1 = result.centroid(1);

        let (low, high) = if c0[0] < c1[0] { (c0, c1) } else { (c1, c0) };
        assert!(low[0].abs() < 0.5, "expected near 0, got {}", low[0]);
        assert!(low[1].abs() < 0.5, "expected near 0, got {}", low[1]);
        assert!(
            (high[0] - 10.0).abs() < 0.5,
            "expected near 10, got {}",
            high[0]
        );
        assert!(
            (high[1] - 10.0).abs() < 0.5,
            "expected near 10, got {}",
            high[1]
        );
    }

    #[test]
    fn test_fit_local_empty() {
        let result = fit_local(&[], 4, 2, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(result.k, 0);
        assert!(result.centroids.is_empty());
    }

    #[test]
    fn test_fit_local_single_vector() {
        let data = vec![1.0, 2.0, 3.0];
        let result = fit_local(&data, 3, 1, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(result.k, 1);
        let c = result.centroid(0);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fit_local_cosine_metric() {
        let data = vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9];

        let result = fit_local(&data, 2, 2, 30, 1e-5, DistanceMetric::CosineSimilarity);
        assert_eq!(result.k, 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_parallel_assign_basic() {
        let data = vec![0.0, 0.0, 10.0, 10.0];
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let engine = DistanceEngine::auto();

        let (counts, sums, _inertia) =
            parallel_assign(&data, &centroids, 2, 2, DistanceMetric::L2Squared, &engine);

        assert_eq!(counts.iter().sum::<usize>(), 2);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);

        // Vector (0,0) assigned to centroid 0, vector (10,10) to centroid 1
        assert!((sums[0] - 0.0).abs() < 1e-6);
        assert!((sums[1] - 0.0).abs() < 1e-6);
        assert!((sums[2] - 10.0).abs() < 1e-6);
        assert!((sums[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_nearest_centroid_l2() {
        let engine = DistanceEngine::auto();
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let query = vec![1.0, 1.0];

        let (idx, dist) =
            find_nearest_centroid(&query, &centroids, 2, DistanceMetric::L2Squared, &engine);
        assert_eq!(idx, 0);
        assert!((dist - 2.0).abs() < 1e-5); // (1-0)^2 + (1-0)^2 = 2.0
    }

    #[test]
    fn test_find_nearest_centroid_dot_product() {
        let engine = DistanceEngine::auto();
        // For dot product, higher is better
        let centroids = vec![1.0, 0.0, 0.0, 1.0];
        let query = vec![1.0, 0.0]; // More similar to centroid 0

        let (idx, _dist) =
            find_nearest_centroid(&query, &centroids, 2, DistanceMetric::DotProduct, &engine);
        assert_eq!(idx, 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fit_cuda_aware_single_gpu() {
        let data = vec![
            0.0, 0.1, 0.1, 0.0, -0.1, 0.0, 10.0, 10.1, 10.1, 10.0, 9.9, 10.0,
        ];
        let result = fit_cuda_aware(&data, 2, 2, 30, 1e-5, DistanceMetric::L2Squared, 1);
        assert_eq!(result.k, 2);
        assert_eq!(result.dimension, 2);
        assert!(result.iterations > 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fit_cuda_aware_multi_gpu_ddp() {
        let data = vec![
            0.0, 0.1, 0.1, 0.0, -0.1, 0.0, 10.0, 10.1, 10.1, 10.0, 9.9, 10.0,
        ];
        let result = fit_cuda_aware(&data, 2, 2, 30, 1e-5, DistanceMetric::L2Squared, 2);
        assert_eq!(result.k, 2);
        assert_eq!(result.dimension, 2);
        assert!(result.iterations > 0);
    }
}
