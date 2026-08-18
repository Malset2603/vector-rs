//! Vector storage backends.
//!
//! This module provides the [`VectorStorage`] trait and two implementations:
//! - [`HeapStorage`] — in-memory heap-allocated flat buffer
//! - [`MmapStorage`] — memory-mapped file with zero-copy access

mod heap;
mod mmap;
mod traits;

pub use heap::HeapStorage;
pub use mmap::MmapStorage;
pub use traits::VectorStorage;
