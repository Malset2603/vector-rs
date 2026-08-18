//! Hierarchical Navigable Small World (HNSW) graph index.
//!
//! Provides high-throughput, low-latency approximate nearest neighbor (ANN) search
//! with sub-millisecond query latency and near-perfect recall.

pub mod builder;
pub mod config;
pub mod graph;
pub mod serializer;
pub mod visited;

pub use builder::HnswBuilder;
pub use config::HnswConfig;
pub use graph::HnswGraph;
pub use serializer::HnswSerializer;
pub use visited::VisitedTracker;

use std::path::Path;
use vector_simd::DistanceEngine;

use crate::DistanceMetric;
use crate::storage::VectorStorage;
use crate::types::{Result, SearchResult, VectorId, VectorIndexError};

/// Hierarchical Navigable Small World (HNSW) index over a [`VectorStorage`] backend.
///
/// Implements state-of-the-art approximate nearest neighbor search via multi-layer
/// proximity graphs. Upper layers provide long-range skips for logarithmic-time routing,
/// while layer 0 contains dense connections for granular nearest-neighbor exploration.
///
/// # Type Parameter
///
/// `S` — the underlying storage backend implementing [`VectorStorage`].
///
/// # Example
///
/// ```rust
/// use vector_index::{DistanceMetric, hnsw::HnswIndex, storage::{HeapStorage, VectorStorage}};
///
/// let mut storage = HeapStorage::new(3);
/// storage.push(&[1.0, 0.0, 0.0]).unwrap();
/// storage.push(&[0.0, 1.0, 0.0]).unwrap();
/// storage.push(&[0.0, 0.0, 1.0]).unwrap();
///
/// // Build HNSW index sequentially or in parallel
/// let index = HnswIndex::build(storage, DistanceMetric::L2Squared);
/// let results = index.search_default(&[1.0, 0.1, 0.0], 2).unwrap();
///
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].id, 0); // Closest is (1, 0, 0)
/// ```
pub struct HnswIndex<S: VectorStorage> {
    storage: S,
    graph: HnswGraph,
    metric: DistanceMetric,
    engine: DistanceEngine,
}

impl<S: VectorStorage> HnswIndex<S> {
    /// Builds an `HnswIndex` over the provided storage with default configuration
    /// using sequential construction.
    pub fn build(storage: S, metric: DistanceMetric) -> Self {
        Self::build_with_config(storage, HnswConfig::default(), metric)
    }

    /// Builds an `HnswIndex` with custom configuration using sequential construction.
    pub fn build_with_config(storage: S, config: HnswConfig, metric: DistanceMetric) -> Self {
        let builder = HnswBuilder::new(&storage, config, metric);
        let graph = builder.build();
        Self {
            storage,
            graph,
            metric,
            engine: DistanceEngine::auto(),
        }
    }

    /// Builds an `HnswIndex` in parallel across all available CPU cores via `rayon`.
    pub fn build_parallel(storage: S, metric: DistanceMetric) -> Self {
        Self::build_parallel_with_config(storage, HnswConfig::default(), metric)
    }

    /// Builds an `HnswIndex` in parallel with custom configuration via `rayon`.
    pub fn build_parallel_with_config(
        storage: S,
        config: HnswConfig,
        metric: DistanceMetric,
    ) -> Self {
        let builder = HnswBuilder::new(&storage, config, metric);
        let graph = builder.build_parallel();
        Self {
            storage,
            graph,
            metric,
            engine: DistanceEngine::auto(),
        }
    }

    /// Constructs an `HnswIndex` from pre-existing storage and graph components.
    pub fn from_parts(storage: S, graph: HnswGraph, metric: DistanceMetric) -> Self {
        Self {
            storage,
            graph,
            metric,
            engine: DistanceEngine::auto(),
        }
    }

    /// Returns a reference to the storage backend.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Returns a reference to the graph structure.
    pub fn graph(&self) -> &HnswGraph {
        &self.graph
    }

    /// Returns the distance metric used.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Returns the configuration parameters.
    pub fn config(&self) -> &HnswConfig {
        &self.graph.config
    }

    /// Saves the HNSW graph topology to a binary file.
    pub fn save_graph<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        HnswSerializer::save_to_file(&self.graph, path)
    }

    /// Loads an `HnswIndex` from a pre-existing storage and a saved graph file.
    pub fn load_graph<P: AsRef<Path>>(storage: S, path: P, metric: DistanceMetric) -> Result<Self> {
        let graph = HnswSerializer::load_from_file(path)?;
        Ok(Self::from_parts(storage, graph, metric))
    }

    /// Computes internal distance between query vector and a stored vector ID.
    #[inline]
    fn internal_distance(&self, query: &[f32], target_id: VectorId) -> f32 {
        let vt = self.storage.get(target_id);
        match self.metric {
            DistanceMetric::L2Squared => self.engine.l2_squared(query, vt),
            DistanceMetric::DotProduct => -self.engine.dot_product(query, vt),
            DistanceMetric::CosineSimilarity => 1.0 - self.engine.cosine_similarity(query, vt),
            DistanceMetric::Manhattan => self.engine.manhattan(query, vt),
            DistanceMetric::Minkowski => self.engine.minkowski(query, vt, 3.0),
            DistanceMetric::Chebyshev => self.engine.chebyshev(query, vt),
            DistanceMetric::Hamming => self.engine.hamming(query, vt),
            DistanceMetric::Mahalanobis => self.engine.mahalanobis(query, vt),
            DistanceMetric::Jaccard => self.engine.jaccard(query, vt),
            DistanceMetric::Hellinger => self.engine.hellinger(query, vt),
        }
    }

    /// Converts internal distance back to external user score.
    #[inline]
    fn to_user_score(&self, internal_dist: f32) -> f32 {
        match self.metric {
            DistanceMetric::L2Squared => internal_dist,
            DistanceMetric::DotProduct => -internal_dist,
            DistanceMetric::CosineSimilarity => 1.0 - internal_dist,
            DistanceMetric::Manhattan
            | DistanceMetric::Minkowski
            | DistanceMetric::Chebyshev
            | DistanceMetric::Hamming
            | DistanceMetric::Mahalanobis
            | DistanceMetric::Jaccard
            | DistanceMetric::Hellinger => internal_dist,
        }
    }

    /// Searches for the `k` approximate nearest neighbors to `query` using default `ef_search`.
    pub fn search_default(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search(query, k, self.graph.config.ef_search)
    }

    /// Searches for the `k` approximate nearest neighbors to `query` using a custom `ef_search`.
    ///
    /// # Arguments
    ///
    /// * `query` — Query vector slice (must have length == `storage.dimension()`).
    /// * `k` — Number of top nearest neighbors to return.
    /// * `ef_search` — Size of dynamic candidate list (higher = better recall, slightly higher latency).
    ///
    /// # Errors
    ///
    /// - `EmptyIndex` if the index contains no vectors.
    /// - `DimensionMismatch` if `query.len() != storage.dimension()`.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<SearchResult>> {
        if self.storage.is_empty() {
            return Err(VectorIndexError::EmptyIndex);
        }
        if query.len() != self.storage.dimension() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.storage.dimension(),
                got: query.len(),
            });
        }

        let k = k.min(self.storage.len());
        if k == 0 {
            return Ok(Vec::new());
        }

        let ef = ef_search.max(k);

        let ep = {
            let ep_lock = self.graph.entry_point.read();
            match *ep_lock {
                Some(ep) => ep,
                None => return Err(VectorIndexError::EmptyIndex),
            }
        };

        let max_l = *self.graph.max_level.read();
        let mut curr_obj = ep;
        let mut curr_dist = self.internal_distance(query, curr_obj);

        // Phase 1: Greedy routing from max_level down to 1
        for l in (1..=max_l).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.graph.get_neighbors(curr_obj, l);
                for n_id in neighbors {
                    let d = self.internal_distance(query, n_id);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_obj = n_id;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: Beam search on layer 0 with beam width `ef`
        let builder = HnswBuilder::new(&self.storage, self.graph.config.clone(), self.metric);
        let mut tracker = VisitedTracker::new(self.storage.len());
        let mut candidates =
            builder.search_layer_internal(&self.graph, query, &[curr_obj], ef, 0, &mut tracker);

        // Sort ascending by internal distance (closest first)
        candidates.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);

        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|e| SearchResult::new(e.id, self.to_user_score(e.dist)))
            .collect();

        // Sort results by external user score
        if self.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.distance
                    .partial_cmp(&a.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results)
    }

    /// Evaluates the recall@K of retrieved results against a ground truth result set.
    ///
    /// Returns the fraction of ground truth items present in the retrieved set ($[0.0, 1.0]$).
    pub fn evaluate_recall(ground_truth: &[SearchResult], retrieved: &[SearchResult]) -> f32 {
        if ground_truth.is_empty() {
            return 1.0;
        }

        let mut hits = 0;
        for gt in ground_truth {
            if retrieved.iter().any(|r| r.id == gt.id) {
                hits += 1;
            }
        }

        hits as f32 / ground_truth.len() as f32
    }
}

impl<S: VectorStorage> std::fmt::Debug for HnswIndex<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("num_vectors", &self.storage.len())
            .field("dimension", &self.storage.dimension())
            .field("metric", &self.metric)
            .field("max_level", &*self.graph.max_level.read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use crate::storage::HeapStorage;

    const EPSILON: f32 = 1e-4;

    fn make_test_data_3d() -> HeapStorage {
        let mut s = HeapStorage::new(3);
        s.push(&[1.0, 0.0, 0.0]).unwrap();
        s.push(&[0.0, 1.0, 0.0]).unwrap();
        s.push(&[0.0, 0.0, 1.0]).unwrap();
        s.push(&[1.0, 1.0, 0.0]).unwrap();
        s.push(&[1.0, 1.0, 1.0]).unwrap();
        s
    }

    #[test]
    fn test_hnsw_search_l2_basic() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage, DistanceMetric::L2Squared);

        let query = [1.0, 0.0, 0.0];
        let results = index.search_default(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance.abs() < EPSILON);
    }

    #[test]
    fn test_hnsw_search_dot_product() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage, DistanceMetric::DotProduct);

        let query = [1.0, 1.0, 1.0];
        let results = index.search_default(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 4); // Vector (1,1,1) has maximum dot product = 3
        assert!((results[0].distance - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_hnsw_search_cosine() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage, DistanceMetric::CosineSimilarity);

        let query = [1.0, 0.0, 0.0];
        let results = index.search_default(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!((results[0].distance - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_hnsw_parallel_build_consistency() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build_parallel(storage, DistanceMetric::L2Squared);

        let query = [0.0, 1.0, 0.0];
        let results = index.search_default(&query, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_hnsw_recall_evaluation_against_flat() {
        let dim = 128;
        let n = 200;
        let mut storage = HeapStorage::new(dim);

        // Generate synthetic vectors with unique values (avoid duplicate vectors from small modulus)
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 7 + d * 13 + 3) % 10007) as f32 * 0.001)
                .collect();
            storage.push(&v).unwrap();
        }

        let flat = FlatIndex::new(storage.clone(), DistanceMetric::L2Squared);
        let config = HnswConfig::new(16, 100, 50);
        let hnsw = HnswIndex::build_with_config(storage, config, DistanceMetric::L2Squared);

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 17 + 5) % 10007) as f32 * 0.001)
            .collect();

        let k = 10;
        let ground_truth = flat.search(&query, k).unwrap();
        let retrieved = hnsw.search(&query, k, 60).unwrap();

        let recall = HnswIndex::<HeapStorage>::evaluate_recall(&ground_truth, &retrieved);
        println!("HNSW 128-D Recall@{k}: {:.2}%", recall * 100.0);
        assert!(
            recall >= 0.90,
            "Recall@10 should be at least 90%, got {:.2}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_hnsw_high_dimensional_768d_recall() {
        let dim = 768;
        let n = 100;
        let mut storage = HeapStorage::new(dim);

        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 11 + d * 19 + 7) % 101) as f32 * 0.01)
                .collect();
            storage.push(&v).unwrap();
        }

        let flat = FlatIndex::new(storage.clone(), DistanceMetric::CosineSimilarity);
        let config = HnswConfig::new(16, 100, 50);
        let hnsw = HnswIndex::build_parallel_with_config(
            storage,
            config,
            DistanceMetric::CosineSimilarity,
        );

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 23 + 11) % 89) as f32 * 0.01)
            .collect();

        let k = 5;
        let ground_truth = flat.search(&query, k).unwrap();
        let retrieved = hnsw.search(&query, k, 80).unwrap();

        let recall = HnswIndex::<HeapStorage>::evaluate_recall(&ground_truth, &retrieved);
        println!("HNSW 768-D Recall@{k}: {:.2}%", recall * 100.0);
        assert!(
            recall >= 0.80,
            "Recall@5 should be at least 80%, got {:.2}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_hnsw_save_and_load_graph() {
        let dir = tempfile::tempdir().unwrap();
        let graph_path = dir.path().join("hnsw_test.bin");

        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage.clone(), DistanceMetric::L2Squared);
        index.save_graph(&graph_path).unwrap();

        let loaded =
            HnswIndex::load_graph(storage, &graph_path, DistanceMetric::L2Squared).unwrap();
        let query = [1.0, 0.0, 0.0];
        let original_res = index.search_default(&query, 2).unwrap();
        let loaded_res = loaded.search_default(&query, 2).unwrap();

        assert_eq!(original_res.len(), loaded_res.len());
        for (a, b) in original_res.iter().zip(loaded_res.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.distance - b.distance).abs() < EPSILON);
        }
    }

    #[test]
    fn test_hnsw_empty_index_error() {
        let storage = HeapStorage::new(3);
        let config = HnswConfig::default();
        let index = HnswIndex::build_with_config(storage, config, DistanceMetric::L2Squared);

        let res = index.search_default(&[1.0, 2.0, 3.0], 5);
        assert!(matches!(res, Err(VectorIndexError::EmptyIndex)));
    }

    #[test]
    fn test_hnsw_dimension_mismatch() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage, DistanceMetric::L2Squared);

        let res = index.search_default(&[1.0, 2.0], 5);
        assert!(matches!(
            res,
            Err(VectorIndexError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_hnsw_k_zero_and_k_larger_than_n() {
        let storage = make_test_data_3d();
        let index = HnswIndex::build(storage, DistanceMetric::L2Squared);

        let res_zero = index.search_default(&[1.0, 0.0, 0.0], 0).unwrap();
        assert_eq!(res_zero.len(), 0);

        let res_large = index.search_default(&[1.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(res_large.len(), 5); // clamped to storage len
    }

    #[test]
    fn test_hnsw_with_mmap_storage() {
        let dir = tempfile::tempdir().unwrap();
        let mmap_path = dir.path().join("mmap_test.bin");

        let flat_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        crate::storage::MmapStorage::create_from_flat(&mmap_path, 3, &flat_data).unwrap();
        let mmap_storage = crate::storage::MmapStorage::open(&mmap_path).unwrap();

        let index = HnswIndex::build(mmap_storage, DistanceMetric::L2Squared);
        let res = index.search_default(&[1.0, 0.1, 0.0], 2).unwrap();

        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn test_hnsw_1536d_recall() {
        let dim = 1536;
        let n = 50;
        let mut storage = HeapStorage::new(dim);

        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 13 + d * 7 + 11) % 10007) as f32 * 0.001)
                .collect();
            storage.push(&v).unwrap();
        }

        let flat = FlatIndex::new(storage.clone(), DistanceMetric::CosineSimilarity);
        let config = HnswConfig::new(16, 100, 50);
        let hnsw = HnswIndex::build_parallel_with_config(
            storage,
            config,
            DistanceMetric::CosineSimilarity,
        );

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 31 + 17) % 10007) as f32 * 0.001)
            .collect();

        let k = 5;
        let ground_truth = flat.search(&query, k).unwrap();
        let retrieved = hnsw.search(&query, k, 60).unwrap();

        let recall = HnswIndex::<HeapStorage>::evaluate_recall(&ground_truth, &retrieved);
        assert!(
            recall >= 0.90,
            "1536-D Recall@5 should be >= 90%, got {:.2}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_hnsw_dot_product_recall() {
        let dim = 64;
        let n = 150;
        let mut storage = HeapStorage::new(dim);

        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 17 + d * 5 + 3) % 10007) as f32 * 0.001)
                .collect();
            storage.push(&v).unwrap();
        }

        let flat = FlatIndex::new(storage.clone(), DistanceMetric::DotProduct);
        let config = HnswConfig::new(16, 100, 50);
        let hnsw = HnswIndex::build_with_config(storage, config, DistanceMetric::DotProduct);

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 11 + 7) % 10007) as f32 * 0.001)
            .collect();

        let k = 10;
        let ground_truth = flat.search(&query, k).unwrap();
        let retrieved = hnsw.search(&query, k, 60).unwrap();

        let recall = HnswIndex::<HeapStorage>::evaluate_recall(&ground_truth, &retrieved);
        assert!(
            recall >= 0.90,
            "DotProduct Recall@10 should be >= 90%, got {:.2}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_hnsw_evaluate_recall_helper_cases() {
        // Empty ground truth
        assert_eq!(HnswIndex::<HeapStorage>::evaluate_recall(&[], &[]), 1.0);

        let gt = vec![
            SearchResult::new(0, 1.0),
            SearchResult::new(1, 2.0),
            SearchResult::new(2, 3.0),
            SearchResult::new(3, 4.0),
        ];

        // Perfect match
        let ret_perfect = vec![
            SearchResult::new(0, 1.0),
            SearchResult::new(1, 2.0),
            SearchResult::new(2, 3.0),
            SearchResult::new(3, 4.0),
        ];
        assert_eq!(
            HnswIndex::<HeapStorage>::evaluate_recall(&gt, &ret_perfect),
            1.0
        );

        // Half match (2 out of 4)
        let ret_half = vec![
            SearchResult::new(0, 1.0),
            SearchResult::new(2, 3.0),
            SearchResult::new(99, 10.0),
        ];
        assert_eq!(
            HnswIndex::<HeapStorage>::evaluate_recall(&gt, &ret_half),
            0.5
        );

        // Zero match
        let ret_none = vec![SearchResult::new(99, 10.0)];
        assert_eq!(
            HnswIndex::<HeapStorage>::evaluate_recall(&gt, &ret_none),
            0.0
        );
    }

    #[test]
    fn test_hnsw_accessors_and_debug() {
        let storage = make_test_data_3d();
        let config = HnswConfig::new(8, 40, 20);
        let index =
            HnswIndex::build_with_config(storage, config.clone(), DistanceMetric::L2Squared);

        assert_eq!(index.storage().len(), 5);
        assert_eq!(index.storage().dimension(), 3);
        assert_eq!(index.metric(), DistanceMetric::L2Squared);
        assert_eq!(index.config().m, 8);

        let debug_str = format!("{index:?}");
        assert!(debug_str.contains("HnswIndex"));
        assert!(debug_str.contains("num_vectors: 5"));
    }
}
