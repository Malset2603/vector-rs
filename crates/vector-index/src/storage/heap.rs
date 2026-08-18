//! In-memory heap-allocated vector storage.
//!
//! Vectors are stored in a single contiguous `Vec<f32>` buffer using a
//! row-major flat layout. This is the simplest storage backend, suitable
//! for building indexes in memory before optionally persisting to disk.

use super::traits::VectorStorage;
use crate::types::{Result, VectorIndexError};

/// Heap-allocated vector storage backed by a flat `Vec<f32>`.
///
/// # Memory Layout
///
/// ```text
/// [v0_d0, v0_d1, ..., v0_dD-1, v1_d0, v1_d1, ..., v1_dD-1, ...]
/// ```
///
/// Access to vector `id` returns `&data[id * dim .. (id+1) * dim]`.
///
/// # Example
///
/// ```rust
/// use vector_index::storage::{HeapStorage, VectorStorage};
///
/// let mut storage = HeapStorage::new(3);
/// storage.push(&[1.0, 2.0, 3.0]).unwrap();
/// storage.push(&[4.0, 5.0, 6.0]).unwrap();
///
/// assert_eq!(storage.len(), 2);
/// assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone)]
pub struct HeapStorage {
    /// Flat contiguous buffer of all vector data.
    data: Vec<f32>,
    /// Dimensionality of each vector.
    dimension: usize,
    /// Number of vectors stored (cached for O(1) access).
    num_vectors: usize,
}

impl HeapStorage {
    /// Creates a new empty `HeapStorage` for vectors of the given dimension.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` is 0.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "vector dimension must be > 0");
        Self {
            data: Vec::new(),
            dimension,
            num_vectors: 0,
        }
    }

    /// Creates a `HeapStorage` with pre-allocated capacity for `capacity` vectors.
    ///
    /// This avoids re-allocations during bulk inserts.
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        assert!(dimension > 0, "vector dimension must be > 0");
        Self {
            data: Vec::with_capacity(dimension * capacity),
            dimension,
            num_vectors: 0,
        }
    }

    /// Constructs a `HeapStorage` from raw data.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `data.len()` is not a multiple of `dimension`.
    pub fn from_raw(data: Vec<f32>, dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !data.len().is_multiple_of(dimension) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dimension,
                got: data.len() % dimension,
            });
        }
        let num_vectors = data.len() / dimension;
        Ok(Self {
            data,
            dimension,
            num_vectors,
        })
    }

    /// Appends a vector to the storage.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `vector.len() != self.dimension()`.
    pub fn push(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        self.data.extend_from_slice(vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Appends multiple vectors from a flat buffer.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `flat_data.len()` is not a multiple of `dimension`.
    pub fn extend_from_flat(&mut self, flat_data: &[f32]) -> Result<()> {
        if !flat_data.len().is_multiple_of(self.dimension) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dimension,
                got: flat_data.len() % self.dimension,
            });
        }
        let count = flat_data.len() / self.dimension;
        self.data.extend_from_slice(flat_data);
        self.num_vectors += count;
        Ok(())
    }
}

impl VectorStorage for HeapStorage {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    fn len(&self) -> usize {
        self.num_vectors
    }

    #[inline]
    fn get(&self, id: u32) -> &[f32] {
        let id = id as usize;
        assert!(
            id < self.num_vectors,
            "vector id {id} out of bounds (len={})",
            self.num_vectors
        );
        let start = id * self.dimension;
        &self.data[start..start + self.dimension]
    }

    #[inline]
    fn as_raw_slice(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_push() {
        let mut storage = HeapStorage::new(3);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());

        storage.push(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(storage.len(), 1);
        assert!(!storage.is_empty());
        assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);

        storage.push(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.get(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_push_wrong_dimension() {
        let mut storage = HeapStorage::new(3);
        let result = storage.push(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_raw() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = HeapStorage::from_raw(data, 3).unwrap();
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.dimension(), 3);
        assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.get(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_from_raw_invalid() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = HeapStorage::from_raw(data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_extend_from_flat() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 2.0]).unwrap();
        storage.extend_from_flat(&[3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(2), &[5.0, 6.0]);
    }

    #[test]
    fn test_with_capacity() {
        let storage = HeapStorage::with_capacity(128, 1000);
        assert_eq!(storage.dimension(), 128);
        assert_eq!(storage.len(), 0);
    }

    #[test]
    fn test_as_raw_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = HeapStorage::from_raw(data.clone(), 2).unwrap();
        assert_eq!(storage.as_raw_slice(), &data[..]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let storage = HeapStorage::new(3);
        storage.get(0);
    }

    #[test]
    #[should_panic(expected = "vector dimension must be > 0")]
    fn test_zero_dimension() {
        HeapStorage::new(0);
    }

    #[test]
    fn test_high_dimensional() {
        let dim = 768;
        let mut storage = HeapStorage::new(dim);
        let vec1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let vec2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.02).collect();
        storage.push(&vec1).unwrap();
        storage.push(&vec2).unwrap();
        assert_eq!(storage.get(0), &vec1[..]);
        assert_eq!(storage.get(1), &vec2[..]);
    }
}
