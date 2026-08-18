//! Memory-mapped vector storage using `memmap2`.
//!
//! Provides zero-copy access to vector data stored on disk. Vectors are
//! memory-mapped into the process's virtual address space, allowing the OS
//! to manage page faults and caching transparently.
//!
//! # File Format
//!
//! ```text
//! Offset  Size   Field
//! ──────  ─────  ───────────────────────
//! 0       8      Magic number (0x56454352_53544F52 = "VECRSTOR")
//! 8       8      Number of vectors (u64, little-endian)
//! 16      8      Dimension (u64, little-endian)
//! 24      N×D×4  Vector data (f32, little-endian, contiguous)
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::traits::VectorStorage;
use crate::types::{Result, VectorIndexError};

/// Magic number identifying a valid VectorRS storage file.
/// ASCII representation: "VECRSTOR" (8 bytes).
const MAGIC: u64 = 0x5645_4352_5354_4F52;

/// Size of the file header in bytes (magic + num_vectors + dimension).
const HEADER_SIZE: usize = 24;

/// Memory-mapped vector storage for zero-copy access to on-disk data.
///
/// The underlying file is memory-mapped as read-only. Vector data is accessed
/// directly from the mapped region via `bytemuck::cast_slice`, avoiding any
/// deserialization or heap allocation.
///
/// # Example
///
/// ```no_run
/// use vector_index::storage::{MmapStorage, VectorStorage};
///
/// // Create a storage file from vectors
/// let vectors: Vec<&[f32]> = vec![
///     &[1.0, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
/// ];
/// MmapStorage::create("vectors.bin", 3, &vectors).unwrap();
///
/// // Open and query the storage
/// let storage = MmapStorage::open("vectors.bin").unwrap();
/// assert_eq!(storage.len(), 2);
/// assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);
/// ```
pub struct MmapStorage {
    /// Memory-mapped file region.
    _mmap: Mmap,
    /// Pointer to the start of the f32 data (after the header).
    data: *const f32,
    /// Number of vectors.
    num_vectors: usize,
    /// Dimensionality of each vector.
    dimension: usize,
    /// Path to the underlying file (retained for error messages).
    #[allow(dead_code)]
    path: PathBuf,
}

// SAFETY: The underlying mmap is read-only and the data pointer is derived
// from it. The mmap lifetime is tied to the struct, so the pointer is always
// valid. Read-only access to a fixed memory region is inherently thread-safe.
unsafe impl Send for MmapStorage {}
unsafe impl Sync for MmapStorage {}

impl MmapStorage {
    /// Creates a new storage file at `path` and writes all vectors into it.
    ///
    /// # Arguments
    ///
    /// * `path` — File path to create (overwrites if exists).
    /// * `dimension` — Dimensionality of each vector.
    /// * `vectors` — Slice of vector slices to write. Each must have length == `dimension`.
    ///
    /// # Errors
    ///
    /// Returns `Io` on file creation failure, or `DimensionMismatch` if any
    /// vector's length differs from `dimension`.
    pub fn create<P: AsRef<Path>>(path: P, dimension: usize, vectors: &[&[f32]]) -> Result<()> {
        let path = path.as_ref();

        // Validate dimensions
        for v in vectors {
            if v.len() != dimension {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dimension,
                    got: v.len(),
                });
            }
        }

        let num_vectors = vectors.len() as u64;
        let dim = dimension as u64;

        let mut file = File::create(path)?;

        // Write header
        file.write_all(&MAGIC.to_le_bytes())?;
        file.write_all(&num_vectors.to_le_bytes())?;
        file.write_all(&dim.to_le_bytes())?;

        // Write vector data
        for v in vectors {
            let bytes: &[u8] = bytemuck::cast_slice(v);
            file.write_all(bytes)?;
        }

        file.flush()?;
        Ok(())
    }

    /// Creates a new storage file from a flat contiguous f32 buffer.
    ///
    /// This is more efficient than `create()` for bulk writes since it
    /// avoids per-vector iteration.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `flat_data.len()` is not a multiple of `dimension`.
    pub fn create_from_flat<P: AsRef<Path>>(
        path: P,
        dimension: usize,
        flat_data: &[f32],
    ) -> Result<()> {
        if !flat_data.len().is_multiple_of(dimension) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dimension,
                got: flat_data.len() % dimension,
            });
        }

        let path = path.as_ref();
        let num_vectors = (flat_data.len() / dimension) as u64;
        let dim = dimension as u64;

        let mut file = File::create(path)?;

        // Write header
        file.write_all(&MAGIC.to_le_bytes())?;
        file.write_all(&num_vectors.to_le_bytes())?;
        file.write_all(&dim.to_le_bytes())?;

        // Write all vector data in one call
        let bytes: &[u8] = bytemuck::cast_slice(flat_data);
        file.write_all(bytes)?;

        file.flush()?;
        Ok(())
    }

    /// Opens an existing storage file and memory-maps it for read-only access.
    ///
    /// Validates the file header (magic number, dimensions) and ensures the
    /// file size matches the declared data size.
    ///
    /// # Errors
    ///
    /// Returns `InvalidHeader` if the magic number is wrong or the header is malformed.
    /// Returns `FileSizeMismatch` if the file is truncated.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let metadata = fs::metadata(&path)?;
        let file_size = metadata.len();

        if file_size < HEADER_SIZE as u64 {
            return Err(VectorIndexError::InvalidHeader {
                path: path.clone(),
                reason: format!("file too small: {file_size} bytes (minimum {HEADER_SIZE})"),
            });
        }

        // SAFETY: The file is opened read-only and we validate the contents
        // before deriving any pointers.
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header
        let header_bytes = &mmap[..HEADER_SIZE];

        let magic = u64::from_le_bytes(header_bytes[0..8].try_into().unwrap());
        if magic != MAGIC {
            return Err(VectorIndexError::InvalidHeader {
                path: path.clone(),
                reason: format!("invalid magic number: 0x{magic:016X} (expected 0x{MAGIC:016X})"),
            });
        }

        let num_vectors = u64::from_le_bytes(header_bytes[8..16].try_into().unwrap()) as usize;
        let dimension = u64::from_le_bytes(header_bytes[16..24].try_into().unwrap()) as usize;

        // Validate file size
        let expected_data_size = num_vectors * dimension * std::mem::size_of::<f32>();
        let expected_file_size = HEADER_SIZE as u64 + expected_data_size as u64;
        if file_size < expected_file_size {
            return Err(VectorIndexError::FileSizeMismatch {
                path: path.clone(),
                expected: expected_file_size,
                got: file_size,
            });
        }

        // Derive data pointer
        let data = &mmap[HEADER_SIZE..] as *const [u8] as *const u8 as *const f32;

        Ok(Self {
            _mmap: mmap,
            data,
            num_vectors,
            dimension,
            path,
        })
    }
}

impl VectorStorage for MmapStorage {
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
        // SAFETY: The mmap is valid for the lifetime of self, and we validated
        // that the file contains enough data for all declared vectors.
        unsafe { std::slice::from_raw_parts(self.data.add(start), self.dimension) }
    }

    #[inline]
    fn as_raw_slice(&self) -> &[f32] {
        let total = self.num_vectors * self.dimension;
        if total == 0 {
            return &[];
        }
        // SAFETY: Same as get() — mmap is alive and data was validated.
        unsafe { std::slice::from_raw_parts(self.data, total) }
    }
}

impl std::fmt::Debug for MmapStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapStorage")
            .field("path", &self.path)
            .field("num_vectors", &self.num_vectors)
            .field("dimension", &self.dimension)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_open_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_vectors.bin");

        let v0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let v2: Vec<f32> = vec![9.0, 10.0, 11.0, 12.0];

        MmapStorage::create(&path, 4, &[&v0, &v1, &v2]).unwrap();

        let storage = MmapStorage::open(&path).unwrap();
        assert_eq!(storage.dimension(), 4);
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());

        assert_eq!(storage.get(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(storage.get(1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(storage.get(2), &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_create_from_flat_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat_vectors.bin");

        let flat_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        MmapStorage::create_from_flat(&path, 3, &flat_data).unwrap();

        let storage = MmapStorage::open(&path).unwrap();
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.dimension(), 3);
        assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.get(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_as_raw_slice() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw_slice.bin");

        let flat = vec![10.0, 20.0, 30.0, 40.0];
        MmapStorage::create_from_flat(&path, 2, &flat).unwrap();

        let storage = MmapStorage::open(&path).unwrap();
        assert_eq!(storage.as_raw_slice(), &flat[..]);
    }

    #[test]
    fn test_empty_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");

        let vectors: Vec<&[f32]> = vec![];
        MmapStorage::create(&path, 128, &vectors).unwrap();

        let storage = MmapStorage::open(&path).unwrap();
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.dimension(), 128);
        assert_eq!(storage.as_raw_slice(), &[] as &[f32]);
    }

    #[test]
    fn test_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_magic.bin");

        // Write garbage header
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0u8; 24]).unwrap();

        let result = MmapStorage::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("invalid magic number"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.bin");

        // Write valid header but not enough data
        let mut f = File::create(&path).unwrap();
        f.write_all(&MAGIC.to_le_bytes()).unwrap();
        f.write_all(&10u64.to_le_bytes()).unwrap(); // 10 vectors
        f.write_all(&4u64.to_le_bytes()).unwrap(); // 4 dimensions
        // Missing 10 * 4 * 4 = 160 bytes of data
        f.flush().unwrap();

        let result = MmapStorage::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_on_create() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mismatch.bin");

        let v0 = vec![1.0, 2.0, 3.0];
        let v1 = vec![4.0, 5.0]; // wrong dimension
        let result = MmapStorage::create(&path, 3, &[&v0, &v1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_high_dimensional_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("high_dim.bin");

        let dim = 1536;
        let vec1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
        let vec2: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.001).collect();

        MmapStorage::create(&path, dim, &[&vec1, &vec2]).unwrap();
        let storage = MmapStorage::open(&path).unwrap();

        assert_eq!(storage.dimension(), dim);
        assert_eq!(storage.len(), 2);

        // Verify exact data integrity
        for (a, b) in storage.get(0).iter().zip(vec1.iter()) {
            assert_eq!(a, b);
        }
        for (a, b) in storage.get(1).iter().zip(vec2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_get_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oob.bin");

        let v0 = vec![1.0, 2.0];
        MmapStorage::create(&path, 2, &[&v0]).unwrap();
        let storage = MmapStorage::open(&path).unwrap();
        storage.get(1); // only id=0 exists
    }

    #[test]
    fn test_file_too_small() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.bin");

        // Write only 10 bytes — less than the minimum header size
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();

        let result = MmapStorage::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("file too small"),
            "unexpected error: {err}"
        );
    }
}
