//! Flat brute-force vector search index.
//!
//! Provides exact nearest-neighbor search by computing the distance from the
//! query vector to every stored vector and maintaining a bounded top-K result set
//! via a binary heap. This serves as the correctness baseline for approximate
//! indexes (HNSW, IVF) in later phases.
//!
//! # Performance Characteristics
//!
//! - **Search complexity:** O(N × D) where N = number of vectors, D = dimension
//! - **Memory overhead:** O(K) for the result heap — zero additional allocation
//!   beyond what the storage backend already provides
//! - **Parallelism:** `search_parallel` splits the dataset across `rayon` threads,
//!   each maintaining a local top-K heap, then merges results

use std::collections::BinaryHeap;

use rayon::prelude::*;
use vector_simd::DistanceEngine;

use crate::DistanceMetric;
use crate::storage::VectorStorage;
use crate::types::{Result, SearchResult, VectorIndexError};

/// A flat brute-force search index over a [`VectorStorage`] backend.
///
/// Iterates over all vectors in storage to find the exact top-K nearest
/// neighbors. All distance computations are delegated to
/// [`vector_simd::DistanceEngine`] for SIMD-accelerated throughput.
///
/// # Type Parameter
///
/// `S` — the storage backend. Must implement [`VectorStorage`].
///
/// # Example
///
/// ```rust
/// use vector_index::{DistanceMetric, flat::FlatIndex, storage::HeapStorage};
/// use vector_index::storage::VectorStorage;
///
/// let mut storage = HeapStorage::new(3);
/// storage.push(&[1.0, 0.0, 0.0]).unwrap();
/// storage.push(&[0.0, 1.0, 0.0]).unwrap();
/// storage.push(&[1.0, 1.0, 0.0]).unwrap();
///
/// let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
/// let results = index.search(&[1.0, 0.5, 0.0], 2).unwrap();
///
/// assert_eq!(results.len(), 2);
/// ```
pub struct FlatIndex<S: VectorStorage> {
    storage: S,
    engine: DistanceEngine,
    metric: DistanceMetric,
}

impl<S: VectorStorage> FlatIndex<S> {
    /// Creates a new `FlatIndex` over the given storage with the specified distance metric.
    ///
    /// Automatically selects the best SIMD backend via [`DistanceEngine::auto()`].
    pub fn new(storage: S, metric: DistanceMetric) -> Self {
        Self {
            storage,
            engine: DistanceEngine::auto(),
            metric,
        }
    }

    /// Returns a reference to the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Returns the distance metric used by this index.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Computes the distance/similarity between two vectors using the configured metric.
    #[inline]
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2Squared => self.engine.l2_squared(a, b),
            DistanceMetric::DotProduct => self.engine.dot_product(a, b),
            DistanceMetric::CosineSimilarity => self.engine.cosine_similarity(a, b),
            DistanceMetric::Manhattan => self.engine.manhattan(a, b),
            DistanceMetric::Minkowski => self.engine.minkowski(a, b, 3.0),
            DistanceMetric::Chebyshev => self.engine.chebyshev(a, b),
            DistanceMetric::Hamming => self.engine.hamming(a, b),
            DistanceMetric::Mahalanobis => self.engine.mahalanobis(a, b),
            DistanceMetric::Jaccard => self.engine.jaccard(a, b),
            DistanceMetric::Hellinger => self.engine.hellinger(a, b),
        }
    }

    /// Converts a raw distance to a "heap score" where **larger = worse**.
    ///
    /// For distance metrics (L2): score = distance (larger = farther = worse) ✓
    /// For similarity metrics (Dot/Cosine): score = -similarity (larger = less similar = worse) ✓
    ///
    /// This allows using a standard max-heap to maintain top-K by evicting the
    /// worst (largest score) element when the heap exceeds capacity K.
    #[inline]
    fn to_heap_score(&self, raw: f32) -> f32 {
        if self.metric.higher_is_better() {
            -raw // negate so max-heap evicts least similar
        } else {
            raw // distance metric: max-heap evicts farthest
        }
    }

    /// Converts a heap score back to the raw distance/similarity value.
    #[inline]
    fn to_raw_score(&self, score: f32) -> f32 {
        if self.metric.higher_is_better() {
            -score
        } else {
            score
        }
    }

    /// Searches for the `k` nearest neighbors to `query` using brute-force
    /// sequential scan.
    ///
    /// Returns results sorted by relevance (closest/most similar first).
    ///
    /// # Errors
    ///
    /// - `EmptyIndex` if the storage contains no vectors.
    /// - `DimensionMismatch` if `query.len() != storage.dimension()`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
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

        // Max-heap of (heap_score, id) — we keep at most K entries.
        // The root is the *worst* result, which we evict when a better one arrives.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

        for id in 0..self.storage.len() as u32 {
            let vec = self.storage.get(id);
            let raw_dist = self.compute_distance(query, vec);
            let score = self.to_heap_score(raw_dist);

            if heap.len() < k {
                heap.push(HeapEntry { score, id });
            } else if let Some(worst) = heap.peek()
                && score < worst.score
            {
                // This result is better than the worst in the heap — swap them
                heap.pop();
                heap.push(HeapEntry { score, id });
            }
        }

        // Extract results and sort by relevance (best first)
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|e| SearchResult::new(e.id, self.to_raw_score(e.score)))
            .collect();

        // Sort: for distance metrics, ascending; for similarity metrics, descending
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

    /// Searches for the `k` nearest neighbors using parallel brute-force scan.
    ///
    /// Splits the storage into chunks processed by `rayon` worker threads.
    /// Each thread maintains a local top-K heap, and results are merged
    /// using a final bounded heap.
    ///
    /// This is beneficial for large datasets where the linear scan dominates
    /// wall-clock time. For small datasets (< ~10,000 vectors), the overhead
    /// of thread synchronization may outweigh the parallelism benefit.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.storage.is_empty() {
            return Err(VectorIndexError::EmptyIndex);
        }
        if query.len() != self.storage.dimension() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.storage.dimension(),
                got: query.len(),
            });
        }

        let n = self.storage.len();
        let k = k.min(n);
        if k == 0 {
            return Ok(Vec::new());
        }

        let dim = self.storage.dimension();
        let raw = self.storage.as_raw_slice();

        // Determine chunk size for parallelism.
        // Each chunk should be large enough to amortize thread overhead.
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = n.div_ceil(num_threads); // ceiling division

        // Each parallel chunk computes a local top-K
        let local_results: Vec<Vec<HeapEntry>> = (0..n)
            .into_par_iter()
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(n);
                let engine = DistanceEngine::auto();
                let mut local_heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

                for id in start..end {
                    let offset = id * dim;
                    let vec = &raw[offset..offset + dim];
                    let raw_dist = match self.metric {
                        DistanceMetric::L2Squared => engine.l2_squared(query, vec),
                        DistanceMetric::DotProduct => engine.dot_product(query, vec),
                        DistanceMetric::CosineSimilarity => engine.cosine_similarity(query, vec),
                        DistanceMetric::Manhattan => engine.manhattan(query, vec),
                        DistanceMetric::Minkowski => engine.minkowski(query, vec, 3.0),
                        DistanceMetric::Chebyshev => engine.chebyshev(query, vec),
                        DistanceMetric::Hamming => engine.hamming(query, vec),
                        DistanceMetric::Mahalanobis => engine.mahalanobis(query, vec),
                        DistanceMetric::Jaccard => engine.jaccard(query, vec),
                        DistanceMetric::Hellinger => engine.hellinger(query, vec),
                    };
                    let score = self.to_heap_score(raw_dist);

                    if local_heap.len() < k {
                        local_heap.push(HeapEntry {
                            score,
                            id: id as u32,
                        });
                    } else if let Some(worst) = local_heap.peek()
                        && score < worst.score
                    {
                        local_heap.pop();
                        local_heap.push(HeapEntry {
                            score,
                            id: id as u32,
                        });
                    }
                }

                local_heap.into_vec()
            })
            .collect();

        // Merge local results into a global top-K heap
        let mut global_heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        for local in local_results {
            for entry in local {
                if global_heap.len() < k {
                    global_heap.push(entry);
                } else if let Some(worst) = global_heap.peek()
                    && entry.score < worst.score
                {
                    global_heap.pop();
                    global_heap.push(entry);
                }
            }
        }

        let mut results: Vec<SearchResult> = global_heap
            .into_iter()
            .map(|e| SearchResult::new(e.id, self.to_raw_score(e.score)))
            .collect();

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
}

/// Internal heap entry for bounded top-K selection.
///
/// The `score` field is a "heap score" where larger = worse.
/// `BinaryHeap` is a max-heap, so the root is always the worst entry,
/// which is the first candidate for eviction.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    score: f32,
    id: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<S: VectorStorage> std::fmt::Debug for FlatIndex<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatIndex")
            .field("num_vectors", &self.storage.len())
            .field("dimension", &self.storage.dimension())
            .field("metric", &self.metric)
            .field("backend", &self.engine.backend())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::HeapStorage;

    const EPSILON: f32 = 1e-5;

    /// Helper: create a HeapStorage with known vectors.
    fn make_storage_3d() -> HeapStorage {
        let mut s = HeapStorage::new(3);
        s.push(&[1.0, 0.0, 0.0]).unwrap(); // id=0: unit x
        s.push(&[0.0, 1.0, 0.0]).unwrap(); // id=1: unit y
        s.push(&[0.0, 0.0, 1.0]).unwrap(); // id=2: unit z
        s.push(&[1.0, 1.0, 0.0]).unwrap(); // id=3: (1,1,0)
        s.push(&[1.0, 1.0, 1.0]).unwrap(); // id=4: (1,1,1)
        s
    }

    // ─────────────────────────────────────────────
    // L2 Squared tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_l2_search_basic() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let query = [1.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // Closest to (1,0,0) should be id=0 (distance=0), then id=3 (dist=1), then id=4 (dist=2)
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance.abs() < EPSILON);
        assert_eq!(results[1].id, 3);
        assert!((results[1].distance - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_search_exact_match() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let results = index.search(&[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
        assert!(results[0].distance.abs() < EPSILON);
    }

    #[test]
    fn test_l2_search_k_larger_than_n() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let results = index.search(&[0.0, 0.0, 0.0], 100).unwrap();
        assert_eq!(results.len(), 5); // clamped to storage size
    }

    #[test]
    fn test_l2_search_k_zero() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let results = index.search(&[0.0, 0.0, 0.0], 0).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_l2_results_sorted_ascending() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let results = index.search(&[0.5, 0.5, 0.5], 5).unwrap();
        for w in results.windows(2) {
            assert!(
                w[0].distance <= w[1].distance + EPSILON,
                "L2 results not sorted ascending: {} > {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    // ─────────────────────────────────────────────
    // Dot Product tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_dot_product_search() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::DotProduct);
        // Query aligned with (1,1,1): id=4 should have highest dot product
        let results = index.search(&[1.0, 1.0, 1.0], 2).unwrap();
        assert_eq!(results[0].id, 4); // dot = 3.0
        assert!((results[0].distance - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_results_sorted_descending() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::DotProduct);
        let results = index.search(&[1.0, 1.0, 1.0], 5).unwrap();
        for w in results.windows(2) {
            assert!(
                w[0].distance >= w[1].distance - EPSILON,
                "DotProduct results not sorted descending: {} < {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    // ─────────────────────────────────────────────
    // Cosine Similarity tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_cosine_search() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::CosineSimilarity);
        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 0);
        assert!((results[0].distance - 1.0).abs() < EPSILON); // identical direction
    }

    #[test]
    fn test_cosine_results_sorted_descending() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::CosineSimilarity);
        let results = index.search(&[1.0, 0.5, 0.0], 5).unwrap();
        for w in results.windows(2) {
            assert!(
                w[0].distance >= w[1].distance - EPSILON,
                "Cosine results not sorted descending: {} < {}",
                w[0].distance,
                w[1].distance
            );
        }
    }

    // ─────────────────────────────────────────────
    // Error handling tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_search_empty_index() {
        let storage = HeapStorage::new(3);
        let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
        let result = index.search(&[1.0, 2.0, 3.0], 5);
        assert!(matches!(result, Err(VectorIndexError::EmptyIndex)));
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::L2Squared);
        let result = index.search(&[1.0, 2.0], 5); // 2D query, 3D storage
        assert!(matches!(
            result,
            Err(VectorIndexError::DimensionMismatch { .. })
        ));
    }

    // ─────────────────────────────────────────────
    // Parallel search tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_parallel_search_matches_sequential() {
        let storage = make_storage_3d();
        let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
        let query = [0.5, 0.5, 0.5];

        let seq = index.search(&query, 3).unwrap();
        let par = index.search_parallel(&query, 3).unwrap();

        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.id, p.id, "parallel result id mismatch");
            assert!(
                (s.distance - p.distance).abs() < EPSILON,
                "parallel result distance mismatch: {} vs {}",
                s.distance,
                p.distance
            );
        }
    }

    #[test]
    fn test_parallel_search_dot_product() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::DotProduct);
        let query = [1.0, 1.0, 1.0];

        let seq = index.search(&query, 5).unwrap();
        let par = index.search_parallel(&query, 5).unwrap();

        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.id, p.id);
            assert!((s.distance - p.distance).abs() < EPSILON);
        }
    }

    #[test]
    fn test_parallel_search_cosine() {
        let index = FlatIndex::new(make_storage_3d(), DistanceMetric::CosineSimilarity);
        let query = [1.0, 0.5, 0.0];

        let seq = index.search(&query, 5).unwrap();
        let par = index.search_parallel(&query, 5).unwrap();

        assert_eq!(seq.len(), par.len());
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.id, p.id);
            assert!((s.distance - p.distance).abs() < EPSILON);
        }
    }

    #[test]
    fn test_parallel_search_empty_index() {
        let storage = HeapStorage::new(3);
        let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
        let result = index.search_parallel(&[1.0, 2.0, 3.0], 5);
        assert!(matches!(result, Err(VectorIndexError::EmptyIndex)));
    }

    // ─────────────────────────────────────────────
    // High-dimensional accuracy tests
    // ─────────────────────────────────────────────

    #[test]
    fn test_high_dimensional_accuracy() {
        let dim = 768;
        let n = 100;
        let mut storage = HeapStorage::new(dim);

        // Generate deterministic pseudo-random vectors
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 7 + d * 13 + 3) % 97) as f32 * 0.01)
                .collect();
            storage.push(&v).unwrap();
        }

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 11 + 5) % 89) as f32 * 0.01)
            .collect();

        let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
        let results = index.search(&query, 10).unwrap();

        // Verify correctness by computing distances manually with scalar engine
        let scalar = DistanceEngine::scalar();
        let storage_ref = index.storage();
        for r in &results {
            let vec = storage_ref.get(r.id);
            let expected = scalar.l2_squared(&query, vec);
            let rel = if expected.abs() > 1e-6 {
                (r.distance - expected).abs() / expected.abs()
            } else {
                (r.distance - expected).abs()
            };
            assert!(
                rel < 1e-3,
                "accuracy mismatch for id={}: got {}, expected {}, rel={}",
                r.id,
                r.distance,
                expected,
                rel
            );
        }
    }

    #[test]
    fn test_1536d_accuracy() {
        // OpenAI embedding dimension
        let dim = 1536;
        let n = 50;
        let mut storage = HeapStorage::new(dim);

        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 3 + d * 17 + 7) % 101) as f32 * 0.01)
                .collect();
            storage.push(&v).unwrap();
        }

        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 19 + 11) % 83) as f32 * 0.01)
            .collect();

        let index = FlatIndex::new(storage, DistanceMetric::CosineSimilarity);
        let results = index.search(&query, 5).unwrap();

        // Verify results are sorted correctly (descending for cosine)
        for w in results.windows(2) {
            assert!(w[0].distance >= w[1].distance - EPSILON);
        }

        // Verify against scalar engine
        let scalar = DistanceEngine::scalar();
        let storage_ref = index.storage();
        for r in &results {
            let vec = storage_ref.get(r.id);
            let expected = scalar.cosine_similarity(&query, vec);
            assert!(
                (r.distance - expected).abs() < 1e-3,
                "cosine mismatch for id={}: got {}, expected {}",
                r.id,
                r.distance,
                expected
            );
        }
    }

    #[test]
    fn test_single_vector() {
        let mut storage = HeapStorage::new(4);
        storage.push(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
        let results = index.search(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance.abs() < EPSILON);
    }
}
