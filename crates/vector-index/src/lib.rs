//! # vector-index
//!
//! Memory layout, storage backends, and vector search engines for the VectorRS engine.
//!
//! This crate provides:
//! - **Storage backends** — flat contiguous memory layouts for vector data,
//!   including heap-allocated (`HeapStorage`) and memory-mapped (`MmapStorage`) variants.
//! - **Brute-force flat index** — exact nearest-neighbor search with bounded top-K
//!   heap and parallel search via `rayon`.
//!
//! All distance computations are delegated to the [`vector_simd::DistanceEngine`],
//! ensuring optimal SIMD utilization on the host CPU.
//!
//! ## Usage
//!
//! ```rust
//! use vector_index::{DistanceMetric, flat::FlatIndex, storage::HeapStorage};
//!
//! // Build storage
//! let mut storage = HeapStorage::new(3); // 3-dimensional vectors
//! storage.push(&[1.0, 2.0, 3.0]).unwrap();
//! storage.push(&[4.0, 5.0, 6.0]).unwrap();
//! storage.push(&[7.0, 8.0, 9.0]).unwrap();
//!
//! // Create flat brute-force index
//! let index = FlatIndex::new(storage, DistanceMetric::L2Squared);
//!
//! // Search for the nearest vector to the query
//! let results = index.search(&[2.0, 3.0, 4.0], 2).unwrap();
//! assert_eq!(results.len(), 2);
//! ```

pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod storage;
pub mod types;

pub use hnsw::{HnswBuilder, HnswConfig, HnswGraph, HnswIndex};
pub use ivf::{IvfPqConfig, IvfPqIndex, KMeans, ProductQuantizer};
pub use types::{SearchResult, VectorId, VectorIndexError};

/// Specifies the distance/similarity metric used for vector comparisons.
///
/// The choice of metric affects how search results are ranked:
/// - **Distance metrics** (e.g., `L2Squared`): lower values indicate more similar vectors.
/// - **Similarity metrics** (e.g., `DotProduct`, `CosineSimilarity`): higher values indicate
///   more similar vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Squared Euclidean distance: $L_2^2(\mathbf{u}, \mathbf{v}) = \sum (u_i - v_i)^2$.
    /// Lower is more similar.
    L2Squared,
    /// Inner product: $\text{dot}(\mathbf{u}, \mathbf{v}) = \sum u_i \cdot v_i$.
    /// Higher is more similar.
    DotProduct,
    /// Cosine similarity: normalized dot product.
    /// Higher is more similar (range \[-1, 1\]).
    CosineSimilarity,
    /// Manhattan ($L_1$) distance: $\sum |u_i - v_i|$.
    /// Lower is more similar.
    Manhattan,
    /// Minkowski ($L_p$) distance with $p=3$: $(\sum |u_i - v_i|^3)^{1/3}$.
    /// Lower is more similar.
    Minkowski,
    /// Chebyshev ($L_\infty$) distance: $\max |u_i - v_i|$.
    /// Lower is more similar.
    Chebyshev,
    /// Hamming distance: coordinate mismatch count.
    /// Lower is more similar.
    Hamming,
    /// Mahalanobis distance: $D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$.
    /// Lower is more similar.
    Mahalanobis,
    /// Generalized continuous Jaccard distance.
    /// Lower is more similar (range \[0, 1\]).
    Jaccard,
    /// Hellinger distribution distance.
    /// Lower is more similar (range \[0, 1\]).
    Hellinger,
}

impl DistanceMetric {
    /// Returns `true` if higher scores indicate greater similarity.
    ///
    /// For similarity metrics, the search engine negates scores internally
    /// so that a max-heap can be used to maintain the top-K closest results.
    #[inline]
    pub fn higher_is_better(&self) -> bool {
        matches!(
            self,
            DistanceMetric::DotProduct | DistanceMetric::CosineSimilarity
        )
    }
}
