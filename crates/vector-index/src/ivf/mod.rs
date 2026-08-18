//! Inverted File with Product Quantization (IVF-PQ) index.
//!
//! Delivers high-throughput, low-memory approximate nearest neighbor (ANN) search
//! via Voronoi coarse partitioning and Asymmetric Distance Computation (ADC).

pub mod config;
pub mod inverted_list;
pub mod kmeans;
pub mod pq;
pub mod serializer;

pub use config::IvfPqConfig;
pub use inverted_list::{InvertedIndex, InvertedList};
pub use kmeans::{KMeans, KMeansResult};
pub use pq::ProductQuantizer;
pub use serializer::{CentroidFileData, IvfPqSerializer};

use std::collections::{BinaryHeap, HashSet};
use std::path::Path;
use vector_simd::DistanceEngine;

use crate::DistanceMetric;
use crate::storage::VectorStorage;
use crate::types::{Result, SearchResult, VectorId, VectorIndexError};

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

/// Inverted File with Product Quantization (IVF-PQ) index over a [`VectorStorage`] backend.
///
/// Compresses dense vector datasets by up to 96% and enables rapid similarity retrieval
/// by evaluating distances exclusively against precomputed lookup tables (ADC).
///
/// # Type Parameter
///
/// `S` — the storage backend implementing [`VectorStorage`].
pub struct IvfPqIndex<S: VectorStorage> {
    storage: S,
    config: IvfPqConfig,
    metric: DistanceMetric,
    engine: DistanceEngine,
    coarse_centroids: Vec<f32>,
    pq: ProductQuantizer,
    inverted_index: InvertedIndex,
}

impl<S: VectorStorage> IvfPqIndex<S> {
    /// Builds an `IvfPqIndex` over the provided storage with default configuration.
    pub fn build(storage: S, metric: DistanceMetric) -> Result<Self> {
        Self::build_with_config(storage, IvfPqConfig::default(), metric)
    }

    /// Builds an `IvfPqIndex` over the provided storage with custom configuration in parallel (Rayon).
    pub fn build_with_config(
        storage: S,
        config: IvfPqConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        Self::build_parallel_with_config(storage, config, metric)
    }

    /// Builds an `IvfPqIndex` using multi-threaded parallel training with `rayon`.
    pub fn build_parallel_with_config(
        storage: S,
        config: IvfPqConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let dim = storage.dimension();
        config.validate(dim)?;

        if storage.is_empty() {
            return Err(VectorIndexError::EmptyIndex);
        }

        let engine = DistanceEngine::auto();
        let raw_data = storage.as_raw_slice();
        let n = storage.len();

        // 1. Train coarse centroids using parallel k-Means
        let coarse_km = KMeans::fit(
            raw_data,
            dim,
            config.nlist,
            config.max_kmeans_iters,
            config.kmeans_tolerance,
            metric,
        );
        let coarse_centroids = coarse_km.centroids;
        let actual_nlist = coarse_km.k;

        // 2. Compute residuals for PQ training and encoding: r_i = x_i - c_j
        let mut residuals = vec![0.0f32; n * dim];
        let mut assigned_clusters = vec![0usize; n];

        for i in 0..n {
            let vec_slice = &raw_data[i * dim..(i + 1) * dim];
            let (best_cluster, _) =
                KMeans::find_nearest_centroid(vec_slice, &coarse_centroids, dim, metric, &engine);
            assigned_clusters[i] = best_cluster;

            let c_slice = &coarse_centroids[best_cluster * dim..(best_cluster + 1) * dim];
            let r_start = i * dim;
            for d in 0..dim {
                residuals[r_start + d] = vec_slice[d] - c_slice[d];
            }
        }

        // 3. Train Product Quantizer on residuals in parallel
        let pq = ProductQuantizer::train(
            &residuals,
            dim,
            config.num_subvectors,
            config.sub_clusters,
            config.max_kmeans_iters,
            config.kmeans_tolerance,
            metric,
        );

        // 4. Encode residuals and populate inverted lists
        let mut inverted_index = InvertedIndex::new(actual_nlist, config.num_subvectors);
        let mut code_buf = vec![0u8; config.num_subvectors];

        for i in 0..n {
            let r_slice = &residuals[i * dim..(i + 1) * dim];
            pq.encode(r_slice, &mut code_buf, metric);
            inverted_index.add(assigned_clusters[i], i as VectorId, &code_buf);
        }

        Ok(Self {
            storage,
            config,
            metric,
            engine,
            coarse_centroids,
            pq,
            inverted_index,
        })
    }

    /// Builds an `IvfPqIndex` sequentially (single-threaded) with default configuration.
    pub fn build_sequential(storage: S, metric: DistanceMetric) -> Result<Self> {
        Self::build_sequential_with_config(storage, IvfPqConfig::default(), metric)
    }

    /// Builds an `IvfPqIndex` sequentially (single-threaded) with custom configuration.
    pub fn build_sequential_with_config(
        storage: S,
        config: IvfPqConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let dim = storage.dimension();
        config.validate(dim)?;

        if storage.is_empty() {
            return Err(VectorIndexError::EmptyIndex);
        }

        let engine = DistanceEngine::auto();
        let raw_data = storage.as_raw_slice();
        let n = storage.len();

        // 1. Train coarse centroids using sequential k-Means
        let coarse_km = KMeans::fit_sequential(
            raw_data,
            dim,
            config.nlist,
            config.max_kmeans_iters,
            config.kmeans_tolerance,
            metric,
        );
        let coarse_centroids = coarse_km.centroids;
        let actual_nlist = coarse_km.k;

        // 2. Compute residuals for PQ training and encoding: r_i = x_i - c_j
        let mut residuals = vec![0.0f32; n * dim];
        let mut assigned_clusters = vec![0usize; n];

        for i in 0..n {
            let vec_slice = &raw_data[i * dim..(i + 1) * dim];
            let (best_cluster, _) =
                KMeans::find_nearest_centroid(vec_slice, &coarse_centroids, dim, metric, &engine);
            assigned_clusters[i] = best_cluster;

            let c_slice = &coarse_centroids[best_cluster * dim..(best_cluster + 1) * dim];
            let r_start = i * dim;
            for d in 0..dim {
                residuals[r_start + d] = vec_slice[d] - c_slice[d];
            }
        }

        // 3. Train Product Quantizer on residuals sequentially
        let pq = ProductQuantizer::train_sequential(
            &residuals,
            dim,
            config.num_subvectors,
            config.sub_clusters,
            config.max_kmeans_iters,
            config.kmeans_tolerance,
            metric,
        );

        // 4. Encode residuals and populate inverted lists
        let mut inverted_index = InvertedIndex::new(actual_nlist, config.num_subvectors);
        let mut code_buf = vec![0u8; config.num_subvectors];

        for i in 0..n {
            let r_slice = &residuals[i * dim..(i + 1) * dim];
            pq.encode(r_slice, &mut code_buf, metric);
            inverted_index.add(assigned_clusters[i], i as VectorId, &code_buf);
        }

        Ok(Self {
            storage,
            config,
            metric,
            engine,
            coarse_centroids,
            pq,
            inverted_index,
        })
    }

    /// Performs approximate nearest neighbor search using the configured `nprobe`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Performs approximate nearest neighbor search with a custom `nprobe` parameter.
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let dim = self.storage.dimension();
        if query.len() != dim {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        if k == 0 || self.storage.is_empty() {
            return Ok(Vec::new());
        }

        let effective_nprobe = nprobe.min(self.inverted_index.nlist()).max(1);

        // 1. Coarse Search: Find the top-nprobe nearest coarse centroids
        let top_centroids = KMeans::find_top_centroids(
            query,
            &self.coarse_centroids,
            dim,
            effective_nprobe,
            self.metric,
            &self.engine,
        );

        // 2. Bounded Max-Heap for Top-K candidates
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        let mut query_residual = vec![0.0f32; dim];
        let mut adc_lut = vec![0.0f32; self.config.num_subvectors * self.config.sub_clusters];

        // 3. Scan inverted lists of probed clusters
        for (cluster_id, _) in top_centroids {
            let inv_list = self.inverted_index.get_list(cluster_id);
            if inv_list.is_empty() {
                continue;
            }

            // Compute query residual: q' = query - coarse_centroid
            let c_slice = &self.coarse_centroids[cluster_id * dim..(cluster_id + 1) * dim];
            for d in 0..dim {
                query_residual[d] = query[d] - c_slice[d];
            }

            // Compute ADC Lookup Table for this coarse centroid
            self.pq
                .compute_adc_lut(&query_residual, &mut adc_lut, self.metric, &self.engine);

            // Scan all quantized vectors in this inverted list
            let list_len = inv_list.len();
            for item_idx in 0..list_len {
                let id = inv_list.ids[item_idx];
                let code = inv_list.get_code(item_idx, self.config.num_subvectors);

                let approx_dist = self.pq.compute_distance_with_lut(code, &adc_lut);
                let heap_score = self.to_heap_score(approx_dist);

                if heap.len() < k {
                    heap.push(Candidate {
                        id,
                        distance: approx_dist,
                        heap_score,
                    });
                } else if let Some(worst) = heap.peek()
                    && heap_score < worst.heap_score
                {
                    heap.pop();
                    heap.push(Candidate {
                        id,
                        distance: approx_dist,
                        heap_score,
                    });
                }
            }
        }

        // 4. Extract and sort final results
        let mut results = Vec::with_capacity(heap.len());
        while let Some(cand) = heap.pop() {
            results.push(SearchResult::new(cand.id, cand.distance));
        }

        // Results come out worst-to-best from max-heap, reverse to best-to-worst
        results.reverse();
        Ok(results)
    }

    /// Converts a raw distance/similarity score into a heap comparison value where **larger = worse**.
    #[inline]
    fn to_heap_score(&self, raw: f32) -> f32 {
        if self.metric.higher_is_better() {
            -raw
        } else {
            raw
        }
    }

    /// Evaluates the recall accuracy of IVF-PQ results against ground truth results (e.g. from `FlatIndex`).
    pub fn evaluate_recall(ground_truth: &[SearchResult], results: &[SearchResult]) -> f32 {
        if ground_truth.is_empty() || results.is_empty() {
            return 0.0;
        }

        let k = ground_truth.len().min(results.len());
        let gt_set: HashSet<VectorId> = ground_truth.iter().take(k).map(|r| r.id).collect();
        let match_count = results
            .iter()
            .take(k)
            .filter(|r| gt_set.contains(&r.id))
            .count();

        match_count as f32 / k as f32
    }

    /// Persists the trained IVF-PQ index state to a binary file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        IvfPqSerializer::save_to_file(
            path,
            &self.config,
            &self.coarse_centroids,
            &self.pq,
            &self.inverted_index,
            self.storage.dimension(),
            self.metric,
        )
    }

    /// Loads a trained IVF-PQ index state from a binary file over existing storage.
    pub fn load_from_file<P: AsRef<Path>>(storage: S, path: P) -> Result<Self> {
        let (config, coarse_centroids, pq, inverted_index, dimension, metric) =
            IvfPqSerializer::load_from_file(path)?;

        if storage.dimension() != dimension {
            return Err(VectorIndexError::DimensionMismatch {
                expected: storage.dimension(),
                got: dimension,
            });
        }

        Ok(Self {
            storage,
            config,
            metric,
            engine: DistanceEngine::auto(),
            coarse_centroids,
            pq,
            inverted_index,
        })
    }

    /// Returns a reference to the underlying vector storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Returns the configuration parameters.
    pub fn config(&self) -> &IvfPqConfig {
        &self.config
    }

    /// Returns the distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Returns the total number of indexed vectors.
    pub fn len(&self) -> usize {
        self.inverted_index.total_vectors()
    }

    /// Returns `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the coarse centroids buffer.
    pub fn coarse_centroids(&self) -> &[f32] {
        &self.coarse_centroids
    }

    /// Returns a reference to the product quantizer.
    pub fn pq(&self) -> &ProductQuantizer {
        &self.pq
    }

    /// Returns a reference to the inverted index.
    pub fn inverted_index(&self) -> &InvertedIndex {
        &self.inverted_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use crate::storage::HeapStorage;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ivf_pq_end_to_end_search() {
        let dimension = 8;
        let mut storage = HeapStorage::new(dimension);

        // Add 50 synthetic vectors
        for i in 0..50 {
            let v: Vec<f32> = (0..dimension).map(|d| (i * 10 + d) as f32 * 0.1).collect();
            storage.push(&v).unwrap();
        }

        let config = IvfPqConfig::new(4, 2, 2)
            .with_sub_clusters(8)
            .with_max_kmeans_iters(15);

        let index =
            IvfPqIndex::build_with_config(storage.clone(), config, DistanceMetric::L2Squared)
                .unwrap();
        assert_eq!(index.len(), 50);

        let query = storage.get(10);
        let results = index.search(query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // The exact vector (id=10) should be among top results
        let ids: Vec<VectorId> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&10));
    }

    #[test]
    fn test_ivf_pq_recall_vs_flat() {
        let dimension = 16;
        let mut storage = HeapStorage::new(dimension);

        for i in 0..100 {
            let v: Vec<f32> = (0..dimension).map(|d| ((i * 7 + d) % 30) as f32).collect();
            storage.push(&v).unwrap();
        }

        let flat = FlatIndex::new(storage.clone(), DistanceMetric::L2Squared);
        let config = IvfPqConfig::new(8, 4, 4)
            .with_sub_clusters(16)
            .with_max_kmeans_iters(20);

        let ivf = IvfPqIndex::build_with_config(storage.clone(), config, DistanceMetric::L2Squared)
            .unwrap();

        let query = storage.get(42);
        let gt = flat.search(query, 10).unwrap();
        let res = ivf.search(query, 10).unwrap();

        let recall = IvfPqIndex::<HeapStorage>::evaluate_recall(&gt, &res);
        assert!(
            recall >= 0.5,
            "Expected recall >= 50%, got {:.2}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_ivf_pq_save_and_load() {
        let dimension = 4;
        let mut storage = HeapStorage::new(dimension);
        storage.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        storage.push(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        storage.push(&[9.0, 10.0, 11.0, 12.0]).unwrap();

        let config = IvfPqConfig::new(2, 2, 2).with_sub_clusters(4);
        let index =
            IvfPqIndex::build_with_config(storage.clone(), config, DistanceMetric::L2Squared)
                .unwrap();

        let tmp_file = NamedTempFile::new().unwrap();
        index.save_to_file(tmp_file.path()).unwrap();

        let loaded = IvfPqIndex::load_from_file(storage, tmp_file.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.inverted_index().nlist(), 2);
    }

    #[test]
    fn test_ivf_pq_cosine_and_dot_product_search() {
        let dimension = 4;
        let mut storage_cos = HeapStorage::new(dimension);
        let mut storage_dot = HeapStorage::new(dimension);

        for i in 1..=20 {
            let f = i as f32;
            storage_cos.push(&[f, f + 1.0, f + 2.0, f + 3.0]).unwrap();
            storage_dot.push(&[f, f * 2.0, f * 0.5, f + 1.0]).unwrap();
        }

        let config = IvfPqConfig::new(2, 2, 2).with_sub_clusters(4);
        let ivf_cos = IvfPqIndex::build_with_config(
            storage_cos,
            config.clone(),
            DistanceMetric::CosineSimilarity,
        )
        .unwrap();
        let res_cos = ivf_cos.search(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_eq!(res_cos.len(), 2);

        let ivf_dot =
            IvfPqIndex::build_with_config(storage_dot, config, DistanceMetric::DotProduct).unwrap();
        let res_dot = ivf_dot.search(&[1.0, 2.0, 0.5, 2.0], 2).unwrap();
        assert_eq!(res_dot.len(), 2);
    }

    #[test]
    fn test_ivf_pq_search_k_zero_and_empty_storage() {
        let dimension = 2;
        let storage_empty = HeapStorage::new(dimension);
        let config = IvfPqConfig::new(1, 1, 1).with_sub_clusters(1);
        let ivf_empty =
            IvfPqIndex::build_with_config(storage_empty, config.clone(), DistanceMetric::L2Squared);
        assert!(ivf_empty.is_err());

        let mut storage = HeapStorage::new(dimension);
        storage.push(&[1.0, 2.0]).unwrap();
        let ivf =
            IvfPqIndex::build_with_config(storage, config, DistanceMetric::L2Squared).unwrap();
        let res_k0 = ivf.search(&[1.0, 2.0], 0).unwrap();
        assert!(res_k0.is_empty());
    }

    #[test]
    fn test_ivf_pq_search_with_custom_nprobe() {
        let dimension = 4;
        let mut storage = HeapStorage::new(dimension);
        for i in 0..30 {
            storage
                .push(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32])
                .unwrap();
        }

        let config = IvfPqConfig::new(4, 1, 2).with_sub_clusters(4);
        let ivf =
            IvfPqIndex::build_with_config(storage, config, DistanceMetric::L2Squared).unwrap();

        let res_probe1 = ivf.search_with_nprobe(&[0.0, 1.0, 2.0, 3.0], 3, 1).unwrap();
        let res_probe4 = ivf.search_with_nprobe(&[0.0, 1.0, 2.0, 3.0], 3, 4).unwrap();

        assert!(!res_probe1.is_empty());
        assert!(!res_probe4.is_empty());
    }

    #[test]
    fn test_ivf_pq_dimension_mismatch() {
        let mut storage = HeapStorage::new(4);
        storage.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let config = IvfPqConfig::new(1, 1, 2).with_sub_clusters(2);
        let ivf =
            IvfPqIndex::build_with_config(storage, config, DistanceMetric::L2Squared).unwrap();

        // 2D query for 4D index
        let res = ivf.search(&[1.0, 2.0], 1);
        assert!(res.is_err());
    }

    #[test]
    fn test_ivf_pq_evaluate_recall_edge_cases() {
        assert_eq!(IvfPqIndex::<HeapStorage>::evaluate_recall(&[], &[]), 0.0);
        let dummy = vec![SearchResult::new(1, 0.5)];
        assert_eq!(IvfPqIndex::<HeapStorage>::evaluate_recall(&dummy, &[]), 0.0);
        assert_eq!(IvfPqIndex::<HeapStorage>::evaluate_recall(&[], &dummy), 0.0);

        let disjoint = vec![SearchResult::new(99, 0.5)];
        assert_eq!(
            IvfPqIndex::<HeapStorage>::evaluate_recall(&dummy, &disjoint),
            0.0
        );
    }

    #[test]
    fn test_ivf_pq_getters() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 2.0]).unwrap();
        let config = IvfPqConfig::new(1, 1, 1).with_sub_clusters(2);
        let ivf = IvfPqIndex::build_with_config(storage, config.clone(), DistanceMetric::L2Squared)
            .unwrap();

        assert_eq!(ivf.storage().dimension(), 2);
        assert_eq!(ivf.config().nlist, 1);
        assert_eq!(ivf.metric(), DistanceMetric::L2Squared);
        assert_eq!(ivf.coarse_centroids().len(), 2);
        assert_eq!(ivf.pq().num_subvectors, 1);
        assert_eq!(ivf.inverted_index().nlist(), 1);
        assert!(!ivf.is_empty());
    }
}
