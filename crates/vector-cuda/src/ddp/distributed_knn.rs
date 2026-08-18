//! Distributed Data Parallel (DDP) Multi-GPU Exact K-NN Search Engine.
//!
//! Supports both VRAM Sharding (Memory Scaling across GPUs) and Dataset Replication (QPS Scaling).

use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_index::types::{SearchResult, VectorId};

use super::collective::CollectiveOps;
use crate::knn::CudaKnnEngine;

/// Multi-GPU execution and sharding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuShardMode {
    /// Dataset is partitioned across $G$ GPUs ($N/G$ per GPU) to maximize VRAM capacity.
    Sharded,
    /// Dataset is fully replicated on each GPU to maximize batch query QPS throughput.
    Replicated,
}

use crate::device::CudaDeviceContext;
use crate::error::CudaError;

/// DDP Multi-GPU Exact Nearest Neighbor Search Engine.
#[derive(Debug)]
pub struct DistributedKnnEngine {
    num_gpus: usize,
    mode: GpuShardMode,
    dimension: usize,
    num_vectors: usize,
    metric: DistanceMetric,
    // Per-GPU engines
    rank_engines: Vec<CudaKnnEngine>,
    rank_offsets: Vec<usize>,
}

impl DistributedKnnEngine {
    /// Initializes a `DistributedKnnEngine` with a dataset distributed across `num_gpus` device ranks,
    /// strictly validating physical GPU hardware availability.
    pub fn try_new(
        dataset: &[f32],
        dimension: usize,
        num_gpus: usize,
        mode: GpuShardMode,
        metric: DistanceMetric,
    ) -> Result<Self, CudaError> {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        assert_eq!(dataset.len() % dimension, 0);

        let available = CudaDeviceContext::device_count();
        if num_gpus > available {
            return Err(CudaError::InsufficientDevices {
                requested: num_gpus,
                available,
            });
        }

        Ok(Self::init_engine(
            dataset, dimension, num_gpus, mode, metric,
        ))
    }

    /// Initializes a `DistributedKnnEngine` using software emulation (for CPU testing and simulation).
    pub fn emulator(
        dataset: &[f32],
        dimension: usize,
        num_gpus: usize,
        mode: GpuShardMode,
        metric: DistanceMetric,
    ) -> Self {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        assert_eq!(dataset.len() % dimension, 0);
        Self::init_engine(dataset, dimension, num_gpus, mode, metric)
    }

    /// Initializes a `DistributedKnnEngine` with a dataset distributed across `num_gpus` device ranks.
    ///
    /// # Panics
    /// Panics if the requested `num_gpus` exceeds available physical hardware GPUs.
    pub fn new(
        dataset: &[f32],
        dimension: usize,
        num_gpus: usize,
        mode: GpuShardMode,
        metric: DistanceMetric,
    ) -> Self {
        Self::try_new(dataset, dimension, num_gpus, mode, metric)
            .unwrap_or_else(|e| panic!("{}", e))
    }

    fn init_engine(
        dataset: &[f32],
        dimension: usize,
        num_gpus: usize,
        mode: GpuShardMode,
        metric: DistanceMetric,
    ) -> Self {
        let num_vectors = dataset.len() / dimension;
        let g = num_gpus.min(num_vectors.max(1));

        let mut rank_engines = Vec::with_capacity(g);
        let mut rank_offsets = Vec::with_capacity(g);

        match mode {
            GpuShardMode::Sharded => {
                let chunk_size = num_vectors.div_ceil(g);

                for r in 0..g {
                    let start_v = r * chunk_size;
                    let end_v = (start_v + chunk_size).min(num_vectors);
                    let count = if start_v < num_vectors {
                        end_v - start_v
                    } else {
                        0
                    };

                    let start_idx = start_v * dimension;
                    let end_idx = end_v * dimension;
                    let slice = if count > 0 {
                        &dataset[start_idx..end_idx]
                    } else {
                        &[]
                    };

                    rank_engines.push(CudaKnnEngine::new(slice, dimension, metric));
                    rank_offsets.push(start_v);
                }
            }
            GpuShardMode::Replicated => {
                for _ in 0..g {
                    rank_engines.push(CudaKnnEngine::new(dataset, dimension, metric));
                    rank_offsets.push(0);
                }
            }
        }

        Self {
            num_gpus: g,
            mode,
            dimension,
            num_vectors,
            metric,
            rank_engines,
            rank_offsets,
        }
    }

    /// Performs exact K-NN search for a single query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let batch_res = self.search_batch(query, k);
        batch_res.into_iter().next().unwrap_or_default()
    }

    /// Performs exact K-NN search for a batch of $Q$ queries across all $G$ GPU ranks.
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Vec<Vec<SearchResult>> {
        assert_eq!(queries.len() % self.dimension, 0);
        let q_count = queries.len() / self.dimension;
        if q_count == 0 || self.num_vectors == 0 || k == 0 {
            return vec![Vec::new(); q_count];
        }

        match self.mode {
            GpuShardMode::Sharded => {
                // Each GPU evaluates all Q queries against its local shard in parallel
                let per_rank_batch_results: Vec<Vec<Vec<SearchResult>>> = (0..self.num_gpus)
                    .into_par_iter()
                    .map(|r| {
                        let offset = self.rank_offsets[r];
                        let local_batch = self.rank_engines[r].search_batch(queries, k);

                        // Translate local shard vector ID to global vector ID
                        local_batch
                            .into_iter()
                            .map(|res_list| {
                                res_list
                                    .into_iter()
                                    .map(|item| {
                                        SearchResult::new(
                                            (offset as VectorId) + item.id,
                                            item.distance,
                                        )
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();

                // Merge partial results per query across GPU ranks via TopKReduce
                (0..q_count)
                    .into_par_iter()
                    .map(|q_idx| {
                        let partial_for_query: Vec<Vec<SearchResult>> = (0..self.num_gpus)
                            .map(|r| per_rank_batch_results[r][q_idx].clone())
                            .collect();

                        CollectiveOps::top_k_reduce(partial_for_query, k, self.metric)
                    })
                    .collect()
            }
            GpuShardMode::Replicated => {
                // Partition batch queries across GPUs (Q/G per GPU) with zero inter-GPU sync
                let q_chunk_size = q_count.div_ceil(self.num_gpus);

                let chunk_results: Vec<Vec<Vec<SearchResult>>> = (0..self.num_gpus)
                    .into_par_iter()
                    .map(|r| {
                        let q_start = r * q_chunk_size;
                        let q_end = (q_start + q_chunk_size).min(q_count);

                        if q_start >= q_count {
                            return Vec::new();
                        }

                        let start_idx = q_start * self.dimension;
                        let end_idx = q_end * self.dimension;
                        let q_slice = &queries[start_idx..end_idx];

                        self.rank_engines[r].search_batch(q_slice, k)
                    })
                    .collect();

                // Flatten chunk results in original query order
                let mut all_results = Vec::with_capacity(q_count);
                for chunk in chunk_results {
                    all_results.extend(chunk);
                }
                all_results
            }
        }
    }

    /// Returns the number of GPU ranks.
    #[inline]
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }

    /// Returns the sharding mode.
    #[inline]
    pub fn mode(&self) -> GpuShardMode {
        self.mode
    }

    /// Returns the vector dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the total number of vectors across all GPU ranks.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns the distance metric configured for this engine.
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
    fn test_ddp_knn_sharded_mode_exact_match() {
        let dimension = 4;
        let mut data = Vec::new();
        let mut heap_storage = HeapStorage::new(dimension);

        for i in 0..60 {
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
        let ddp_knn = DistributedKnnEngine::emulator(
            &data,
            dimension,
            3,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );

        assert_eq!(ddp_knn.num_gpus(), 3);
        assert_eq!(ddp_knn.num_vectors(), 60);

        let query = heap_storage.get(25);
        let cpu_results = flat_cpu.search(query, 5).unwrap();
        let ddp_results = ddp_knn.search(query, 5);

        assert_eq!(cpu_results.len(), 5);
        assert_eq!(ddp_results.len(), 5);

        for i in 0..5 {
            assert_eq!(ddp_results[i].id, cpu_results[i].id);
            assert!((ddp_results[i].distance - cpu_results[i].distance).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ddp_knn_replicated_mode() {
        let dimension = 2;
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

        let ddp_knn = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Replicated,
            DistanceMetric::L2Squared,
        );

        let batch_queries = vec![0.1, 0.1, 2.9, 2.9];

        let results = ddp_knn.search_batch(&batch_queries, 1);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].id, 0); // (0,0)
        assert_eq!(results[1][0].id, 3); // (3,3)
    }

    #[test]
    fn test_ddp_knn_empty_and_zero_k() {
        let dimension = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sharded = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );
        let replicated = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Replicated,
            DistanceMetric::L2Squared,
        );

        // Empty query
        assert!(sharded.search(&[], 5).is_empty());
        assert!(sharded.search_batch(&[], 5).is_empty());
        assert!(replicated.search(&[], 5).is_empty());
        assert!(replicated.search_batch(&[], 5).is_empty());

        // Zero k
        assert!(sharded.search(&[1.0, 2.0], 0).is_empty());
        assert!(replicated.search(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn test_ddp_knn_sharded_global_id_mapping() {
        let dimension = 2;
        // 4 vectors partitioned across 2 GPUs (2 vectors each)
        let data = vec![
            10.0, 10.0, // Global ID 0 (Rank 0)
            20.0, 20.0, // Global ID 1 (Rank 0)
            30.0, 30.0, // Global ID 2 (Rank 1)
            40.0, 40.0, // Global ID 3 (Rank 1)
        ];

        let sharded = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );

        // Query close to vector at index 3 (40, 40)
        let res3 = sharded.search(&[39.9, 40.1], 1);
        assert_eq!(res3.len(), 1);
        assert_eq!(res3[0].id, 3);

        // Query close to vector at index 1 (20, 20)
        let res1 = sharded.search(&[20.1, 19.9], 1);
        assert_eq!(res1.len(), 1);
        assert_eq!(res1[0].id, 1);
    }

    #[test]
    fn test_ddp_knn_cosine_metric_sharded_and_replicated() {
        let dimension = 2;
        let data = vec![
            1.0, 0.0, // id 0
            0.0, 1.0, // id 1
            -1.0, 0.0, // id 2
            0.0, -1.0, // id 3
        ];

        let sharded = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Sharded,
            DistanceMetric::CosineSimilarity,
        );
        let replicated = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Replicated,
            DistanceMetric::CosineSimilarity,
        );

        let query = [0.99, 0.01];
        let res_sharded = sharded.search(&query, 2);
        let res_replicated = replicated.search(&query, 2);

        assert_eq!(res_sharded.len(), 2);
        assert_eq!(res_sharded[0].id, 0);

        assert_eq!(res_replicated.len(), 2);
        assert_eq!(res_replicated[0].id, 0);
    }

    #[test]
    fn test_ddp_knn_more_gpus_than_vectors() {
        let dimension = 2;
        let data = vec![1.0, 1.0, 2.0, 2.0];

        // 8 GPUs requested for 2 vectors -> clamps g to 2
        let engine = DistributedKnnEngine::emulator(
            &data,
            dimension,
            8,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );
        assert_eq!(engine.num_gpus(), 2);
        assert_eq!(engine.num_vectors(), 2);

        let res = engine.search(&[1.0, 1.0], 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn test_ddp_knn_getters() {
        let dimension = 3;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let engine = DistributedKnnEngine::emulator(
            &data,
            dimension,
            2,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );

        assert_eq!(engine.num_gpus(), 2);
        assert_eq!(engine.num_vectors(), 2);
        assert_eq!(engine.dimension(), 3);
        assert_eq!(engine.mode(), GpuShardMode::Sharded);
        assert_eq!(engine.metric(), DistanceMetric::L2Squared);
    }

    #[test]
    fn test_ddp_knn_insufficient_hardware_error() {
        let dimension = 2;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let available = CudaDeviceContext::device_count();
        let requested = available + 100;

        let res = DistributedKnnEngine::try_new(
            &data,
            dimension,
            requested,
            GpuShardMode::Sharded,
            DistanceMetric::L2Squared,
        );
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
