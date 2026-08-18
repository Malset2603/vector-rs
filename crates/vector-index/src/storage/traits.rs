/// Abstraction over vector storage backends.
///
/// Implementations provide zero-copy access to vectors stored in a flat
/// contiguous memory layout. The trait is designed to be object-safe and
/// thread-safe (`Send + Sync`).
///
/// # Memory Layout
///
/// All implementations use a flat contiguous array where vector `id`'s data
/// occupies indices `[id * dimension .. (id + 1) * dimension]` in the
/// underlying `f32` buffer.
pub trait VectorStorage: Send + Sync {
    /// Returns the dimensionality of vectors in this storage.
    fn dimension(&self) -> usize;

    /// Returns the number of vectors stored.
    fn len(&self) -> usize;

    /// Returns `true` if no vectors are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice view of the vector at the given `id`.
    ///
    /// This is a zero-copy operation — the returned slice points directly
    /// into the underlying contiguous buffer.
    ///
    /// # Panics
    ///
    /// Panics if `id >= self.len()`.
    fn get(&self, id: u32) -> &[f32];

    /// Returns a reference to the entire flat f32 data buffer.
    ///
    /// The buffer contains `len() * dimension()` elements in row-major order.
    fn as_raw_slice(&self) -> &[f32];
}
