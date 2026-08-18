//! Distributed Data Parallel (DDP) Multi-GPU k-Means Clustering Engine.
//!
//! Partitions massive vector datasets across $G$ GPU ranks and performs parallel
//! centroid evaluation synchronized via NCCL-style `AllReduce` and `Broadcast`.

use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_simd::DistanceEngine;

use super::collective::CollectiveOps;
use crate::device::DeviceBuffer;
use crate::kmeans::CudaKMeansResult;

use crate::device::CudaDeviceContext;
use crate::error::CudaError;

/// DDP Multi-GPU k-Means clustering engine.
#[derive(Debug)]
pub struct DistributedKMeansEngine {
    num_gpus: usize,
}

impl DistributedKMeansEngine {
    /// Creates a new `DistributedKMeansEngine` configured for `num_gpus` device ranks,
    /// strictly validating physical GPU hardware availability.
    pub fn try_new(num_gpus: usize) -> Result<Self, CudaError> {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        let available = CudaDeviceContext::device_count();
        if num_gpus > available {
            return Err(CudaError::InsufficientDevices {
                requested: num_gpus,
                available,
            });
        }
        Ok(Self { num_gpus })
    }

    /// Creates a `DistributedKMeansEngine` using software emulation (for CPU testing and simulation).
    pub fn emulator(num_gpus: usize) -> Self {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        Self { num_gpus }
    }

    /// Creates a new `DistributedKMeansEngine` configured for `num_gpus` device ranks.
    ///
    /// # Panics
    /// Panics if the requested `num_gpus` exceeds available physical hardware GPUs.
    pub fn new(num_gpus: usize) -> Self {
        Self::try_new(num_gpus).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Fits $k$ centroids over a dataset partitioned across $G$ GPU ranks.
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
        let g = self.num_gpus.min(n);

        // 1. Shard dataset across G GPU device buffers
        let chunk_size = n.div_ceil(g);
        let mut rank_data_buffers = Vec::with_capacity(g);
        let mut rank_vector_counts = Vec::with_capacity(g);

        for r in 0..g {
            let start_v = r * chunk_size;
            let end_v = (start_v + chunk_size).min(n);
            let count = if start_v < n { end_v - start_v } else { 0 };

            let start_idx = start_v * dimension;
            let end_idx = end_v * dimension;
            let slice = if count > 0 {
                &data[start_idx..end_idx]
            } else {
                &[]
            };

            rank_data_buffers.push(DeviceBuffer::from_host(slice));
            rank_vector_counts.push(count);
        }

        // 2. Initial random centroid selection on Rank 0
        let mut centroids = Self::init_random(data, dimension, n, effective_k);
        let mut rank_centroid_buffers: Vec<DeviceBuffer<f32>> = (0..g)
            .map(|_| DeviceBuffer::from_host(&centroids))
            .collect();

        let engine = DistanceEngine::auto();
        let mut iterations = 0;
        let mut final_inertia = f32::MAX;

        for iter in 0..max_iters {
            iterations = iter + 1;

            // 3. Broadcast updated centroids from Rank 0 to all G GPU ranks
            CollectiveOps::broadcast(0, &mut rank_centroid_buffers);

            // 4. Parallel local assignment & accumulation across all G GPUs
            let rank_results: Vec<(Vec<usize>, Vec<f32>, f32)> = (0..g)
                .into_par_iter()
                .map(|r| {
                    let d_slice = rank_data_buffers[r].as_slice();
                    let cent_slice = rank_centroid_buffers[r].as_slice();
                    let v_count = rank_vector_counts[r];
                    if v_count == 0 {
                        return (
                            vec![0usize; effective_k],
                            vec![0.0f32; effective_k * dimension],
                            0.0f32,
                        );
                    }

                    let mut local_counts = vec![0usize; effective_k];
                    let mut local_sums = vec![0.0f32; effective_k * dimension];
                    let mut local_inertia = 0.0f32;

                    for i in 0..v_count {
                        let vec_slice = &d_slice[i * dimension..(i + 1) * dimension];
                        let (best_c, dist) =
                            Self::find_nearest(vec_slice, cent_slice, dimension, metric, &engine);

                        local_counts[best_c] += 1;
                        let s_start = best_c * dimension;
                        for d in 0..dimension {
                            local_sums[s_start + d] += vec_slice[d];
                        }
                        local_inertia += dist;
                    }

                    (local_counts, local_sums, local_inertia)
                })
                .collect();

            let mut rank_sum_buffers: Vec<DeviceBuffer<f32>> = Vec::with_capacity(g);
            let mut rank_count_buffers: Vec<DeviceBuffer<usize>> = Vec::with_capacity(g);
            let mut total_iter_inertia = 0.0f32;

            for (counts, sums, inertia) in rank_results {
                rank_count_buffers.push(DeviceBuffer::from_host(&counts));
                rank_sum_buffers.push(DeviceBuffer::from_host(&sums));
                total_iter_inertia += inertia;
            }

            final_inertia = total_iter_inertia;

            // 5. NCCL-style AllReduce across all GPU rank sum and count buffers
            CollectiveOps::all_reduce_sum_f32(&mut rank_sum_buffers);
            CollectiveOps::all_reduce_sum_usize(&mut rank_count_buffers);

            // 6. Update Centroids on Rank 0
            let global_sums = rank_sum_buffers[0].as_slice();
            let global_counts = rank_count_buffers[0].as_slice();
            let mut max_shift = 0.0f32;
            let mut rng = rand::thread_rng();
            let mut new_centroids = vec![0.0f32; effective_k * dimension];

            for (k_idx, &count) in global_counts.iter().enumerate().take(effective_k) {
                let start = k_idx * dimension;

                if count > 0 {
                    let inv_count = 1.0 / (count as f32);
                    let mut shift = 0.0f32;

                    for d in 0..dimension {
                        let val = global_sums[start + d] * inv_count;
                        let diff = val - centroids[start + d];
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

            centroids = new_centroids;
            rank_centroid_buffers[0].copy_from_host(&centroids);

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

    fn init_random(data: &[f32], dimension: usize, n: usize, k: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Vec::with_capacity(k * dimension);
        for &idx in indices.iter().take(k) {
            let start = idx * dimension;
            centroids.extend_from_slice(&data[start..start + dimension]);
        }
        centroids
    }

    /// Returns the number of configured GPU devices.
    #[inline]
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddp_kmeans_multi_gpu() {
        let engine = DistributedKMeansEngine::emulator(4); // 4 GPUs
        let data = vec![
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 9.9,
        ];

        let result = engine.fit(&data, 2, 2, 25, 1e-5, DistanceMetric::L2Squared);
        assert_eq!(result.k, 2);
        assert_eq!(result.dimension, 2);
        assert!(result.iterations > 0);
        assert!(result.inertia >= 0.0);

        let c0 = result.centroid(0);
        let c1 = result.centroid(1);

        let (near_0, near_10) = if c0[0] < 5.0 { (c0, c1) } else { (c1, c0) };
        assert!(near_0[0].abs() < 0.5);
        assert!(near_0[1].abs() < 0.5);
        assert!((near_10[0] - 10.0).abs() < 0.5);
        assert!((near_10[1] - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_ddp_kmeans_empty_and_zero_k() {
        let engine = DistributedKMeansEngine::emulator(4);
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let res_empty = engine.fit(&[], 2, 2, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res_empty.k, 0);
        assert!(res_empty.centroids.is_empty());
        assert_eq!(res_empty.iterations, 0);

        let res_zero_k = engine.fit(&data, 2, 0, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res_zero_k.k, 0);
        assert!(res_zero_k.centroids.is_empty());
    }

    #[test]
    fn test_ddp_kmeans_more_gpus_than_vectors() {
        let engine = DistributedKMeansEngine::emulator(8); // 8 GPUs for 3 vectors
        let data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

        let result = engine.fit(&data, 2, 2, 15, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(result.k, 2);
        assert_eq!(result.centroids.len(), 4);
    }

    #[test]
    fn test_ddp_kmeans_single_cluster() {
        let engine = DistributedKMeansEngine::emulator(2);
        let data = vec![1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0];

        let result = engine.fit(&data, 2, 1, 10, 1e-5, DistanceMetric::L2Squared);
        assert_eq!(result.k, 1);
        let c = result.centroid(0);
        assert!((c[0] - 4.0).abs() < 1e-3);
        assert!((c[1] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_ddp_kmeans_cosine_metric() {
        let engine = DistributedKMeansEngine::emulator(2);
        let data = vec![1.0, 0.0, 0.98, 0.2, 0.0, 1.0, 0.2, 0.98];

        let result = engine.fit(&data, 2, 2, 20, 1e-4, DistanceMetric::CosineSimilarity);
        assert_eq!(result.k, 2);
        assert_eq!(result.dimension, 2);
    }

    #[test]
    fn test_ddp_kmeans_getters() {
        let engine = DistributedKMeansEngine::emulator(6);
        assert_eq!(engine.num_gpus(), 6);
    }

    #[test]
    fn test_ddp_kmeans_insufficient_hardware_error() {
        let available = CudaDeviceContext::device_count();
        let requested = available + 100;

        let res = DistributedKMeansEngine::try_new(requested);
        assert!(res.is_err());
        match res.unwrap_err() {
            CudaError::InsufficientDevices {
                requested: r,
                available: a,
            } => {
                assert_eq!(r, requested);
                assert_eq!(a, available);
            }
            other => panic!("Expected InsufficientDevices error, got: {:?}", other),
        }
    }
}
