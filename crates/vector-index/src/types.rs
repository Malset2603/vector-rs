use std::fmt;
use std::path::PathBuf;

/// Unique identifier for a vector within a storage backend.
pub type VectorId = u32;

/// A single search result returned by an index query.
///
/// Contains the vector's ID and its computed distance/similarity score
/// relative to the query vector.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    /// Identifier of the matched vector in storage.
    pub id: VectorId,
    /// Distance or similarity score.
    ///
    /// For `L2Squared`: lower is closer (distance).
    /// For `DotProduct` / `CosineSimilarity`: higher is more similar.
    pub distance: f32,
}

impl SearchResult {
    /// Creates a new `SearchResult`.
    #[inline]
    pub fn new(id: VectorId, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total ordering for f32: treat NaN as greater than everything
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Errors that can occur during vector index operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorIndexError {
    /// I/O error during storage operations (file read/write, mmap).
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// The provided vector dimension does not match the index/storage dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The index or storage is empty and cannot perform the requested operation.
    #[error("index is empty")]
    EmptyIndex,

    /// The mmap file has an invalid or unrecognized header.
    #[error("invalid storage file header in {path:?}: {reason}")]
    InvalidHeader { path: PathBuf, reason: String },

    /// The mmap file size does not match the expected size from the header.
    #[error("storage file size mismatch in {path:?}: expected {expected} bytes, got {got} bytes")]
    FileSizeMismatch {
        path: PathBuf,
        expected: u64,
        got: u64,
    },

    /// Invalid configuration provided for index initialization.
    #[error("invalid index configuration: {reason}")]
    InvalidConfig { reason: String },
}

/// Result type alias for vector index operations.
pub type Result<T> = std::result::Result<T, VectorIndexError>;

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchResult(id={}, distance={:.6})",
            self.id, self.distance
        )
    }
}
