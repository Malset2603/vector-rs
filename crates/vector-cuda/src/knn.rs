//! GPU-accelerated Exact K-NN search engine using matrix multiplication (GEMM).
//!
//! Evaluates pairwise distances between batch query vectors and dataset partitions
//! via $\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2 \langle q, x \rangle$.

use std::collections::BinaryHeap;

use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_index::types::{SearchResult, VectorId};
use vector_simd::DistanceEngine;

use crate::device::{CudaDeviceContext, DeviceBuffer};

/// Internal candidate item for bounded heap top-K ranking.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: VectorId,
    distance: f32,
    heap_score: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.heap_score == other.heap_score
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.heap_score
            .partial_cmp(&other.heap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// CUDA-accelerated exact Nearest Neighbor search engine.
pub struct CudaKnnEngine {
    context: CudaDeviceContext,
    dataset: DeviceBuffer<f32>,
    data_norms: DeviceBuffer<f32>,
    d_dataset: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    d_data_norms: Option<std::sync::Arc<cudarc::driver::CudaSlice<f32>>>,
    dimension: usize,
    num_vectors: usize,
    metric: DistanceMetric,
}

impl std::fmt::Debug for CudaKnnEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKnnEngine")
            .field("context", &self.context)
            .field("dimension", &self.dimension)
            .field("num_vectors", &self.num_vectors)
            .field("metric", &self.metric)
            .finish()
    }
}

impl Clone for CudaKnnEngine {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            dataset: self.dataset.clone(),
            data_norms: self.data_norms.clone(),
            d_dataset: self.d_dataset.clone(),
            d_data_norms: self.d_data_norms.clone(),
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            metric: self.metric,
        }
    }
}

impl CudaKnnEngine {
    /// Uploads a dataset to GPU device memory and initializes the KNN engine on GPU ordinal 0.
    pub fn new(dataset: &[f32], dimension: usize, metric: DistanceMetric) -> Self {
        Self::with_ordinal(dataset, dimension, metric, 0)
    }

    /// Uploads a dataset to GPU device memory and initializes the KNN engine on a specific GPU ordinal.
    pub fn with_ordinal(
        dataset: &[f32],
        dimension: usize,
        metric: DistanceMetric,
        ordinal: usize,
    ) -> Self {
        assert_eq!(dataset.len() % dimension, 0);
        let num_vectors = dataset.len() / dimension;
        let context = CudaDeviceContext::with_ordinal(ordinal);

        // Precompute squared L2 norms for fast GEMM distance: ||x||^2 = sum(x_i^2)
        let engine = DistanceEngine::auto();
        let mut data_norms = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let start = i * dimension;
            let slice = &dataset[start..start + dimension];
            let norm_sq = engine.dot_product(slice, slice);
            data_norms.push(norm_sq);
        }

        let d_dataset = DeviceBuffer::from_host(dataset);
        let d_norms = DeviceBuffer::from_host(&data_norms);

        // Pre-allocate and upload dataset into GPU VRAM once if device is available
        let (gpu_dataset, gpu_norms) = if let Some(dev) = context.cuda_device() {
            // Transpose dataset to SoA (Structure of Arrays) for perfect memory coalescing on GPU
            let mut soa_dataset = vec![0.0f32; dataset.len()];
            for i in 0..num_vectors {
                for d in 0..dimension {
                    soa_dataset[d * num_vectors + i] = dataset[i * dimension + d];
                }
            }
            let d_d = dev.htod_copy(soa_dataset).ok().map(std::sync::Arc::new);
            let d_n = dev.htod_copy(data_norms.clone()).ok().map(std::sync::Arc::new);
            (d_d, d_n)
        } else {
            (None, None)
        };

        Self {
            context,
            dataset: d_dataset,
            data_norms: d_norms,
            d_dataset: gpu_dataset,
            d_data_norms: gpu_norms,
            dimension,
            num_vectors,
            metric,
        }
    }

    /// Performs exact K-NN search for a single query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let batch_res = self.search_batch(query, k);
        batch_res.into_iter().next().unwrap_or_default()
    }

    /// Performs exact K-NN search for a batch of $Q$ query vectors in parallel.
    ///
    /// `queries` contains $Q \times D$ flat floating-point elements.
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Vec<Vec<SearchResult>> {
        assert_eq!(queries.len() % self.dimension, 0);
        let q_count = queries.len() / self.dimension;
        if q_count == 0 || self.num_vectors == 0 || k == 0 {
            return vec![Vec::new(); q_count];
        }

        let effective_k = k.min(self.num_vectors);

        // Hardware GPU accelerated GEMM search if device is active
        if let Some(dev) = self.context.cuda_device()
            && let Ok(res) = self.search_batch_gpu(dev, queries, effective_k)
        {
            return res;
        }

        // Software CPU SIMD/Rayon Fallback
        self.search_batch_cpu(queries, effective_k)
    }

    fn search_batch_gpu(
        &self,
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        queries: &[f32],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>, Box<dyn std::error::Error + Send + Sync>> {
        use cudarc::driver::{LaunchAsync, LaunchConfig};

        let q_count = queries.len() / self.dimension;
        let effective_k = k.min(self.num_vectors);

        // Precompute query norms
        let engine = DistanceEngine::auto();
        let query_norms: Vec<f32> = (0..q_count)
            .into_par_iter()
            .map(|q| {
                let start = q * self.dimension;
                let slice = &queries[start..start + self.dimension];
                engine.dot_product(slice, slice)
            })
            .collect();

        let d_queries = dev.htod_copy(queries.to_vec())?;
        let d_query_norms = dev.htod_copy(query_norms)?;
        let mut d_dist_matrix = dev.alloc_zeros::<f32>(q_count * self.num_vectors)?;

        // Reuse persistent VRAM allocations or fallback to on-demand upload
        let (d_dataset_temp, d_data_norms_temp);
        let (d_dataset_ref, d_data_norms_ref) = match (&self.d_dataset, &self.d_data_norms) {
            (Some(d), Some(n)) => (d.as_ref(), n.as_ref()),
            _ => {
                let mut soa_dataset = vec![0.0f32; self.dataset.len()];
                for i in 0..self.num_vectors {
                    for d in 0..self.dimension {
                        soa_dataset[d * self.num_vectors + i] = self.dataset.as_slice()[i * self.dimension + d];
                    }
                }
                d_dataset_temp = dev.htod_copy(soa_dataset)?;
                d_data_norms_temp = dev.htod_copy(self.data_norms.as_slice().to_vec())?;
                (&d_dataset_temp, &d_data_norms_temp)
            }
        };

        let func = dev
            .get_func("knn_module", "knn_compute_distance_matrix")
            .ok_or("knn_compute_distance_matrix not found")?;

        let topk_func = dev
            .get_func("knn_module", "knn_topk_select")
            .ok_or("knn_topk_select not found")?;

        let block_dim = (32u32, 8u32, 1u32); // 32x8 threads, computing 4 rows per thread
        let grid_dim = (
            (self.num_vectors as u32).div_ceil(32).max(1),
            (q_count as u32).div_ceil(32).max(1),
            1,
        );

        let cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        let metric_code: i32 = match self.metric {
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

        unsafe {
            func.launch(
                cfg,
                (
                    &d_queries,
                    d_dataset_ref,
                    &d_query_norms,
                    d_data_norms_ref,
                    &mut d_dist_matrix,
                    q_count as i32,
                    self.num_vectors as i32,
                    self.dimension as i32,
                    metric_code,
                ),
            )?;
        }

        let mut d_topk_distances = dev.alloc_zeros::<f32>(q_count * effective_k)?;
        let mut d_topk_indices = dev.alloc_zeros::<i32>(q_count * effective_k)?;

        let topk_cfg = LaunchConfig {
            grid_dim: (q_count as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            topk_func.launch(
                topk_cfg,
                (
                    &d_dist_matrix,
                    &mut d_topk_distances,
                    &mut d_topk_indices,
                    q_count as i32,
                    self.num_vectors as i32,
                    effective_k as i32,
                    metric_code,
                ),
            )?;
        }

        let h_topk_distances = dev.dtoh_sync_copy(&d_topk_distances)?;
        let h_topk_indices = dev.dtoh_sync_copy(&d_topk_indices)?;

        let results: Vec<Vec<SearchResult>> = (0..q_count)
            .map(|q_idx| {
                let mut top_k = Vec::with_capacity(effective_k);
                for k in 0..effective_k {
                    let idx = q_idx * effective_k + k;
                    let vec_id = h_topk_indices[idx];
                    if vec_id >= 0 {
                        top_k.push(SearchResult {
                            id: vec_id as VectorId,
                            distance: h_topk_distances[idx],
                        });
                    }
                }
                top_k
            })
            .collect();

        Ok(results)
    }

    fn search_batch_cpu(&self, queries: &[f32], effective_k: usize) -> Vec<Vec<SearchResult>> {
        let q_count = queries.len() / self.dimension;
        let engine = DistanceEngine::auto();
        let d_slice = self.dataset.as_slice();
        let dim = self.dimension;
        let num_vectors = self.num_vectors;
        let metric = self.metric;

        (0..q_count)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * dim;
                let q_vec = &queries[q_start..q_start + dim];

                let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(effective_k + 1);

                for n_idx in 0..num_vectors {
                    let d_start = n_idx * dim;
                    let d_vec = &d_slice[d_start..d_start + dim];

                    let dist = match metric {
                        DistanceMetric::L2Squared => engine.l2_squared(q_vec, d_vec),
                        DistanceMetric::DotProduct => engine.dot_product(q_vec, d_vec),
                        DistanceMetric::CosineSimilarity => engine.cosine_similarity(q_vec, d_vec),
                        DistanceMetric::Manhattan => engine.manhattan(q_vec, d_vec),
                        DistanceMetric::Minkowski => engine.minkowski(q_vec, d_vec, 3.0),
                        DistanceMetric::Chebyshev => engine.chebyshev(q_vec, d_vec),
                        DistanceMetric::Hamming => engine.hamming(q_vec, d_vec),
                        DistanceMetric::Mahalanobis => engine.mahalanobis(q_vec, d_vec),
                        DistanceMetric::Jaccard => engine.jaccard(q_vec, d_vec),
                        DistanceMetric::Hellinger => engine.hellinger(q_vec, d_vec),
                    };

                    let heap_score = if metric.higher_is_better() {
                        -dist
                    } else {
                        dist
                    };

                    if heap.len() < effective_k {
                        heap.push(Candidate {
                            id: n_idx as VectorId,
                            distance: dist,
                            heap_score,
                        });
                    } else if let Some(worst) = heap.peek()
                        && heap_score < worst.heap_score
                    {
                        heap.pop();
                        heap.push(Candidate {
                            id: n_idx as VectorId,
                            distance: dist,
                            heap_score,
                        });
                    }
                }

                let mut results = Vec::with_capacity(heap.len());
                while let Some(cand) = heap.pop() {
                    results.push(SearchResult::new(cand.id, cand.distance));
                }
                results.reverse();
                results
            })
            .collect()
    }

    /// Returns the device context.
    #[inline]
    pub fn context(&self) -> &CudaDeviceContext {
        &self.context
    }

    /// Returns the vector dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of vectors stored on the device.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns the distance metric.
    #[inline]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vector_index::flat::FlatIndex;
    use vector_index::storage::{HeapStorage, VectorStorage};

    #[test]
    fn test_cuda_knn_exact_match_vs_cpu_flat() {
        let dimension = 4;
        let mut data = Vec::new();
        let mut heap_storage = HeapStorage::new(dimension);

        for i in 0..50 {
            let vec = vec![
                (i * i) as f32 * 0.1,
                i as f32 * 2.0,
                (i + 5) as f32,
                (i * 3) as f32,
            ];
            data.extend_from_slice(&vec);
            heap_storage.push(&vec).unwrap();
        }

        let flat_cpu = FlatIndex::new(heap_storage.clone(), DistanceMetric::L2Squared);
        let cuda_knn = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);

        let query = heap_storage.get(10);
        let cpu_results = flat_cpu.search(query, 5).unwrap();
        let gpu_results = cuda_knn.search(query, 5);

        assert_eq!(cpu_results.len(), 5);
        assert_eq!(gpu_results.len(), 5);

        for i in 0..5 {
            assert_eq!(gpu_results[i].id, cpu_results[i].id);
            assert!((gpu_results[i].distance - cpu_results[i].distance).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cuda_knn_batch_queries() {
        let dimension = 2;
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);
        let batch_queries = vec![0.1, 0.1, 1.9, 1.9];

        let results = engine.search_batch(&batch_queries, 1);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].id, 0); // Closest to (0.1, 0.1) is (0, 0)
        assert_eq!(results[1][0].id, 2); // Closest to (1.9, 1.9) is (2, 2)
    }

    #[test]
    fn test_cuda_knn_empty_and_zero_k() {
        let dimension = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);

        // Empty query
        assert!(engine.search(&[], 5).is_empty());
        assert!(engine.search_batch(&[], 5).is_empty());

        // Zero k
        assert!(engine.search(&[1.0, 2.0], 0).is_empty());
        let batch_zero_k = engine.search_batch(&[1.0, 2.0], 0);
        assert_eq!(batch_zero_k.len(), 1);
        assert!(batch_zero_k[0].is_empty());
    }

    #[test]
    fn test_cuda_knn_k_greater_than_num_vectors() {
        let dimension = 2;
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);

        let results = engine.search(&[0.5, 0.5], 100);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_cuda_knn_cosine_similarity() {
        let dimension = 2;
        let data = vec![
            1.0, 0.0, // id 0 (along X)
            0.0, 1.0, // id 1 (along Y)
            -1.0, 0.0, // id 2 (along -X)
        ];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::CosineSimilarity);

        let query = [0.95, 0.05];
        let results = engine.search(&query, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 0); // highest cosine similarity
        assert_eq!(results[2].id, 2); // lowest cosine similarity
    }

    #[test]
    fn test_cuda_knn_dot_product() {
        let dimension = 2;
        let data = vec![
            1.0, 1.0, // dot with (2,3) = 2 + 3 = 5
            2.0, 2.0, // dot with (2,3) = 4 + 6 = 10
            3.0, 3.0, // dot with (2,3) = 6 + 9 = 15
        ];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::DotProduct);

        let query = [2.0, 3.0];
        let results = engine.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 2); // score 15.0
        assert_eq!(results[1].id, 1); // score 10.0
    }

    #[test]
    fn test_cuda_knn_engine_getters() {
        let dimension = 3;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);

        assert_eq!(engine.dimension(), 3);
        assert_eq!(engine.num_vectors(), 2);
        assert_eq!(engine.metric(), DistanceMetric::L2Squared);
        assert_eq!(engine.context().device_id(), 0);
    }

    #[test]
    fn test_cuda_knn_exact_self_query() {
        let dimension = 2;
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let engine = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);

        for (i, chunk) in data.chunks(dimension).enumerate() {
            let res = engine.search(chunk, 1);
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].id, i as VectorId);
            assert!(res[0].distance.abs() < 1e-4);
        }
    }
}
