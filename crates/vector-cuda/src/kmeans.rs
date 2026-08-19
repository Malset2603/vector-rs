//! GPU-accelerated k-Means clustering engine.
//!
//! Executes parallel centroid assignment and block-level coordinate accumulation
//! across CUDA thread grids for rapid ANN index training.

use rand::Rng;
use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_simd::DistanceEngine;

use crate::device::{CudaDeviceContext, DeviceBuffer};

/// Result from GPU-accelerated k-Means clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaKMeansResult {
    /// Device-fitted centroid matrix of length $k \times D$.
    pub centroids: Vec<f32>,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Number of clusters.
    pub k: usize,
    /// Completed iterations.
    pub iterations: usize,
    /// Final sum of squared errors (inertia).
    pub inertia: f32,
}

impl CudaKMeansResult {
    /// Returns a slice view of centroid $i$.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        let start = i * self.dimension;
        &self.centroids[start..start + self.dimension]
    }
}

/// GPU-accelerated k-Means clustering engine.
#[derive(Debug)]
pub struct CudaKMeansEngine {
    context: CudaDeviceContext,
}

impl Default for CudaKMeansEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaKMeansEngine {
    /// Creates a new `CudaKMeansEngine`.
    pub fn new() -> Self {
        Self {
            context: CudaDeviceContext::new(),
        }
    }

    /// Fits $k$ centroids on the provided dataset using GPU-accelerated parallel execution.
    pub fn fit(
        &self,
        data: &[f32],
        dimension: usize,
        k: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> CudaKMeansResult {
        let n = data.len() / dimension;
        if n == 0 || k == 0 {
            return CudaKMeansResult {
                centroids: Vec::new(),
                dimension,
                k: 0,
                iterations: 0,
                inertia: 0.0,
            };
        }

        let effective_k = k.min(n);

        // Hardware GPU accelerated execution if device is active
        if let Some(dev) = self.context.cuda_device()
            && let Ok(res) = Self::fit_gpu(
                dev,
                data,
                dimension,
                effective_k,
                max_iters,
                tolerance,
                metric,
            )
        {
            return res;
        }

        // Software CPU SIMD/Rayon Fallback
        self.fit_cpu(data, dimension, effective_k, max_iters, tolerance, metric)
    }

    fn fit_gpu(
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        data: &[f32],
        dimension: usize,
        k: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> Result<CudaKMeansResult, Box<dyn std::error::Error + Send + Sync>> {
        use cudarc::driver::{LaunchAsync, LaunchConfig};

        let n = data.len() / dimension;
        let centroids = Self::init_random(data, dimension, n, k);

        let d_data = dev.htod_copy(data.to_vec())?;
        let mut d_centroids = dev.htod_copy(centroids)?;
        let mut d_assignments = dev.alloc_zeros::<i32>(n)?;
        let mut d_cluster_sums = dev.alloc_zeros::<f32>(k * dimension)?;
        let mut d_cluster_counts = dev.alloc_zeros::<i32>(k)?;
        let mut d_inertias = dev.alloc_zeros::<f32>(n)?;
        let mut d_shifts = dev.alloc_zeros::<f32>(k)?;

        let assign_func = dev
            .get_func("kmeans_module", "kmeans_assign_and_accumulate")
            .ok_or("kmeans_assign_and_accumulate not found")?;
        let zero_func = dev
            .get_func("kmeans_module", "kmeans_zero_accumulators")
            .ok_or("kmeans_zero_accumulators not found")?;
        let update_func = dev
            .get_func("kmeans_module", "kmeans_update_centroids")
            .ok_or("kmeans_update_centroids not found")?;

        let block_dim = 256;
        let grid_dim_assign = n.div_ceil(block_dim) as u32;
        let cfg_assign = LaunchConfig {
            grid_dim: (grid_dim_assign.max(1), 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: (dimension * 4) as u32,
        };

        let grid_dim_zero = (k * dimension).div_ceil(block_dim) as u32;
        let cfg_zero = LaunchConfig {
            grid_dim: (grid_dim_zero.max(1), 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let cfg_update = LaunchConfig {
            grid_dim: (k as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let metric_code: i32 = match metric {
            DistanceMetric::L2Squared => 0,
            DistanceMetric::DotProduct => 1,
            DistanceMetric::CosineSimilarity => 2,
            DistanceMetric::Manhattan => 3,
            DistanceMetric::Minkowski => 4,
            DistanceMetric::Chebyshev => 5,
            DistanceMetric::Hamming => 6,
            DistanceMetric::Mahalanobis => 7,
            DistanceMetric::Jaccard => 8,
            DistanceMetric::Hellinger => 9,
        };

        let mut iterations = 0;

        for iter in 0..max_iters {
            iterations = iter + 1;

            // 1. Fast zero-out accumulator buffers on GPU (zero PCIe overhead)
            unsafe {
                zero_func.clone().launch(
                    cfg_zero,
                    (
                        &mut d_cluster_sums,
                        &mut d_cluster_counts,
                        k as i32,
                        dimension as i32,
                    ),
                )?;
            }

            // 2. Parallel assignment, vector inertia accumulation, and cluster sums
            unsafe {
                assign_func.clone().launch(
                    cfg_assign,
                    (
                        &d_data,
                        &d_centroids,
                        &mut d_assignments,
                        &mut d_cluster_sums,
                        &mut d_cluster_counts,
                        &mut d_inertias,
                        n as i32,
                        k as i32,
                        dimension as i32,
                        metric_code,
                    ),
                )?;
            }

            // 3. Fast centroid coordinate update and block-reduction shift directly on GPU
            unsafe {
                update_func.clone().launch(
                    cfg_update,
                    (
                        &d_cluster_sums,
                        &d_cluster_counts,
                        &mut d_centroids,
                        &mut d_shifts,
                        k as i32,
                        dimension as i32,
                    ),
                )?;
            }

            // 4. Convergence check (only K floats transferred over PCIe instead of full dataset)
            let shifts = dev.dtoh_sync_copy(&d_shifts)?;
            let max_shift = shifts.into_iter().fold(0.0f32, f32::max);

            if max_shift <= tolerance {
                break;
            }
        }

        let final_centroids = dev.dtoh_sync_copy(&d_centroids)?;
        let inertias = dev.dtoh_sync_copy(&d_inertias)?;
        let final_inertia = inertias.into_iter().sum::<f32>();

        Ok(CudaKMeansResult {
            centroids: final_centroids,
            dimension,
            k,
            iterations,
            inertia: final_inertia,
        })
    }

    /// Fits $k$ centroids using multi-threaded CPU SIMD/Rayon execution.
    pub fn fit_cpu(
        &self,
        data: &[f32],
        dimension: usize,
        effective_k: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> CudaKMeansResult {
        let n = data.len() / dimension;
        let engine = DistanceEngine::auto();

        // 1. Initial random centroid sampling
        let mut centroids = Self::init_random(data, dimension, n, effective_k);
        let mut iterations = 0;
        let mut final_inertia = f32::MAX;

        // Device buffer allocations
        let d_data = DeviceBuffer::from_host(data);
        let mut d_centroids = DeviceBuffer::from_host(&centroids);

        for iter in 0..max_iters {
            iterations = iter + 1;
            let cent_slice = d_centroids.as_slice();

            // 2. Parallel Assignment & Accumulation across thread blocks
            let (cluster_counts, cluster_sums, total_inertia) = (0..n)
                .into_par_iter()
                .fold(
                    || {
                        (
                            vec![0usize; effective_k],
                            vec![0.0f32; effective_k * dimension],
                            0.0f32,
                        )
                    },
                    |(mut counts, mut sums, mut inertia), i| {
                        let vec_slice = &d_data.as_slice()[i * dimension..(i + 1) * dimension];
                        let (best_c, dist) =
                            Self::find_nearest(vec_slice, cent_slice, dimension, metric, &engine);

                        counts[best_c] += 1;
                        let s_start = best_c * dimension;
                        for d in 0..dimension {
                            sums[s_start + d] += vec_slice[d];
                        }
                        inertia += dist;

                        (counts, sums, inertia)
                    },
                )
                .reduce(
                    || {
                        (
                            vec![0usize; effective_k],
                            vec![0.0f32; effective_k * dimension],
                            0.0f32,
                        )
                    },
                    |(mut c1, mut s1, i1), (c2, s2, i2)| {
                        for k_idx in 0..effective_k {
                            c1[k_idx] += c2[k_idx];
                        }
                        for idx in 0..(effective_k * dimension) {
                            s1[idx] += s2[idx];
                        }
                        (c1, s1, i1 + i2)
                    },
                );

            final_inertia = total_inertia;

            // 3. Update Centroids & Convergence Check
            let mut max_shift = 0.0f32;
            let mut rng = rand::thread_rng();
            let mut new_centroids = vec![0.0f32; effective_k * dimension];

            for (k_idx, &count) in cluster_counts.iter().enumerate().take(effective_k) {
                let start = k_idx * dimension;

                if count > 0 {
                    let inv_count = 1.0 / (count as f32);
                    let mut shift = 0.0f32;

                    for d in 0..dimension {
                        let val = cluster_sums[start + d] * inv_count;
                        let diff = val - cent_slice[start + d];
                        shift += diff * diff;
                        new_centroids[start + d] = val;
                    }

                    if shift > max_shift {
                        max_shift = shift;
                    }
                } else {
                    let random_idx = rng.gen_range(0..n);
                    let sample_start = random_idx * dimension;
                    new_centroids[start..start + dimension]
                        .copy_from_slice(&data[sample_start..sample_start + dimension]);
                }
            }

            d_centroids.copy_from_host(&new_centroids);
            centroids = new_centroids;

            if max_shift <= tolerance {
                break;
            }
        }

        CudaKMeansResult {
            centroids,
            dimension,
            k: effective_k,
            iterations,
            inertia: final_inertia,
        }
    }

    /// Finds the nearest centroid index for a single vector.
    #[inline]
    fn find_nearest(
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
            let start = c_idx * dimension;
            let c_slice = &centroids[start..start + dimension];

            let dist = match metric {
                DistanceMetric::L2Squared => engine.l2_squared(vector, c_slice),
                DistanceMetric::DotProduct => engine.dot_product(vector, c_slice),
                DistanceMetric::CosineSimilarity => engine.cosine_similarity(vector, c_slice),
                DistanceMetric::Manhattan => engine.manhattan(vector, c_slice),
                DistanceMetric::Minkowski => engine.minkowski(vector, c_slice, 3.0),
                DistanceMetric::Chebyshev => engine.chebyshev(vector, c_slice),
                DistanceMetric::Hamming => engine.hamming(vector, c_slice),
                DistanceMetric::Mahalanobis => engine.mahalanobis(vector, c_slice),
                DistanceMetric::Jaccard => engine.jaccard(vector, c_slice),
                DistanceMetric::Hellinger => engine.hellinger(vector, c_slice),
            };

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

    /// Initializes centroids via k-Means++ sampling for faster, optimal convergence.
    fn init_random(data: &[f32], dimension: usize, n: usize, k: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let engine = DistanceEngine::auto();

        let mut centroids = Vec::with_capacity(k * dimension);
        if n == 0 || k == 0 {
            return centroids;
        }

        // 1. Pick first centroid uniformly at random
        let first_idx = rng.gen_range(0..n);
        let start = first_idx * dimension;
        centroids.extend_from_slice(&data[start..start + dimension]);

        // 2. Pick remaining centroids with probability proportional to D(x)^2
        let mut min_distances = vec![f32::MAX; n];
        for _ in 1..k {
            let last_c = &centroids[centroids.len() - dimension..];

            let mut total_weight = 0.0f32;
            for i in 0..n {
                let vec_slice = &data[i * dimension..(i + 1) * dimension];
                let d = engine.l2_squared(vec_slice, last_c);
                if d < min_distances[i] {
                    min_distances[i] = d;
                }
                total_weight += min_distances[i];
            }

            if total_weight <= 1e-6 {
                let rand_idx = rng.gen_range(0..n);
                let start = rand_idx * dimension;
                centroids.extend_from_slice(&data[start..start + dimension]);
                continue;
            }

            let mut threshold = rng.gen_range(0.0..total_weight);
            let mut selected_idx = 0;
            for (i, &d) in min_distances.iter().enumerate().take(n) {
                threshold -= d;
                if threshold <= 0.0 {
                    selected_idx = i;
                    break;
                }
            }

            let start = selected_idx * dimension;
            centroids.extend_from_slice(&data[start..start + dimension]);
        }

        centroids
    }

    /// Returns the device context.
    #[inline]
    pub fn context(&self) -> &CudaDeviceContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kmeans_clustering() {
        let dimension = 4;
        let n_per_cluster = 50;

        let mut data = Vec::new();
        // Cluster 0: around 0.0
        for _ in 0..n_per_cluster {
            data.extend_from_slice(&[0.1, -0.1, 0.05, -0.05]);
        }
        // Cluster 1: around 10.0
        for _ in 0..n_per_cluster {
            data.extend_from_slice(&[10.1, 9.9, 10.05, 9.95]);
        }
        // Cluster 2: around 20.0
        for _ in 0..n_per_cluster {
            data.extend_from_slice(&[20.1, 19.9, 20.05, 19.95]);
        }

        let engine = CudaKMeansEngine::new();
        let result = engine.fit(&data, dimension, 3, 20, 1e-4, DistanceMetric::L2Squared);

        assert_eq!(result.k, 3);
        assert_eq!(result.dimension, dimension);
        assert!(result.iterations > 0);
        assert!(result.inertia < 1.0);

        for c in 0..3 {
            let centroid = result.centroid(c);
            assert_eq!(centroid.len(), dimension);
        }
    }

    #[test]
    fn test_cuda_kmeans_single_cluster() {
        let dimension = 2;
        let data = vec![1.0, 1.0, 1.1, 0.9, 0.9, 1.1];
        let engine = CudaKMeansEngine::new();
        let result = engine.fit(&data, dimension, 1, 10, 1e-4, DistanceMetric::L2Squared);

        assert_eq!(result.k, 1);
        let c = result.centroid(0);
        assert!((c[0] - 1.0).abs() < 0.1);
        assert!((c[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cuda_kmeans_empty_and_zero_k() {
        let engine = CudaKMeansEngine::new();
        let res_empty = engine.fit(&[], 4, 3, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res_empty.k, 0);

        let res_zero_k = engine.fit(&[1.0, 2.0], 2, 0, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res_zero_k.k, 0);
    }

    #[test]
    fn test_cuda_kmeans_k_greater_than_n() {
        let dimension = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 vectors
        let engine = CudaKMeansEngine::new();
        let result = engine.fit(&data, dimension, 5, 10, 1e-4, DistanceMetric::L2Squared);

        assert_eq!(result.k, 2);
    }

    #[test]
    fn test_cuda_kmeans_cosine_and_dot_product_metrics() {
        let dimension = 3;
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.1, 0.0];
        let engine = CudaKMeansEngine::new();

        let res_cos = engine.fit(
            &data,
            dimension,
            3,
            10,
            1e-4,
            DistanceMetric::CosineSimilarity,
        );
        assert_eq!(res_cos.k, 3);

        let res_dot = engine.fit(&data, dimension, 3, 10, 1e-4, DistanceMetric::DotProduct);
        assert_eq!(res_dot.k, 3);
    }

    #[test]
    fn test_cuda_kmeans_high_dimension() {
        let dimension = 128;
        let n = 20;
        let data: Vec<f32> = (0..(n * dimension)).map(|i| (i as f32) * 0.01).collect();

        let engine = CudaKMeansEngine::new();
        let result = engine.fit(&data, dimension, 4, 5, 1e-3, DistanceMetric::L2Squared);

        assert_eq!(result.k, 4);
        assert_eq!(result.dimension, 128);
    }

    #[test]
    fn test_cuda_kmeans_engine_context() {
        let engine = CudaKMeansEngine::new();
        assert_eq!(engine.context().device_id(), 0);
        assert!(engine.context().total_memory_bytes() > 0);
    }
}
