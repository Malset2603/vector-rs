//! Zero-allocation visited node tracking for HNSW graph traversal.
//!
//! In high-throughput nearest-neighbor search, allocating a new `HashSet` or clearing
//! a large bitset on every query introduces substantial CPU and memory overhead.
//!
//! `VisitedTracker` uses an epoch/version-tagged array pattern:
//! - Each vector ID has an associated `u32` tag indicating the epoch when it was visited.
//! - Advancing to the next search query simply increments `current_epoch` in $O(1)$ time.
//! - No memory is cleared or reallocated between queries until the 32-bit epoch counter rolls over.

/// Epoch-based visited tracker with $O(1)$ reset time and zero per-query allocations.
#[derive(Debug, Clone)]
pub struct VisitedTracker {
    /// Array where `tags[id]` stores the epoch when vector `id` was visited.
    tags: Vec<u32>,
    /// Current search epoch.
    current_epoch: u32,
}

impl Default for VisitedTracker {
    fn default() -> Self {
        Self::new(0)
    }
}

impl VisitedTracker {
    /// Creates a new `VisitedTracker` with pre-allocated capacity for `capacity` vectors.
    pub fn new(capacity: usize) -> Self {
        Self {
            tags: vec![0; capacity],
            current_epoch: 1,
        }
    }

    /// Advances to the next search epoch.
    ///
    /// This is an $O(1)$ operation that invalidates all previous visit marks without
    /// clearing the underlying buffer.
    #[inline]
    pub fn advance_epoch(&mut self, capacity: usize) {
        if self.tags.len() < capacity {
            self.tags.resize(capacity, 0);
        }

        self.current_epoch = self.current_epoch.wrapping_add(1);
        // If wrapped to 0 (reserved for unvisited default), clear tags and reset to 1
        if self.current_epoch == 0 {
            self.tags.fill(0);
            self.current_epoch = 1;
        }
    }

    /// Returns `true` if node `id` has been visited in the current epoch.
    #[inline]
    pub fn is_visited(&self, id: u32) -> bool {
        let idx = id as usize;
        if idx < self.tags.len() {
            // SAFETY / in-bounds check
            self.tags[idx] == self.current_epoch
        } else {
            false
        }
    }

    /// Marks node `id` as visited in the current epoch.
    #[inline]
    pub fn mark_visited(&mut self, id: u32) {
        let idx = id as usize;
        if idx >= self.tags.len() {
            self.tags.resize(idx + 1 + 64, 0);
        }
        self.tags[idx] = self.current_epoch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visited_basic() {
        let mut tracker = VisitedTracker::new(10);
        assert!(!tracker.is_visited(0));
        assert!(!tracker.is_visited(5));

        tracker.mark_visited(5);
        assert!(tracker.is_visited(5));
        assert!(!tracker.is_visited(0));
        assert!(!tracker.is_visited(4));
    }

    #[test]
    fn test_visited_advance_epoch() {
        let mut tracker = VisitedTracker::new(10);
        tracker.mark_visited(3);
        tracker.mark_visited(7);
        assert!(tracker.is_visited(3));
        assert!(tracker.is_visited(7));

        // Advance epoch: should clear in O(1)
        tracker.advance_epoch(10);
        assert!(!tracker.is_visited(3));
        assert!(!tracker.is_visited(7));

        tracker.mark_visited(3);
        assert!(tracker.is_visited(3));
        assert!(!tracker.is_visited(7));
    }

    #[test]
    fn test_visited_auto_grow() {
        let mut tracker = VisitedTracker::new(5);
        assert!(!tracker.is_visited(100));

        tracker.mark_visited(100);
        assert!(tracker.is_visited(100));
        assert!(!tracker.is_visited(99));
    }

    #[test]
    fn test_visited_default() {
        let mut tracker = VisitedTracker::default();
        assert!(!tracker.is_visited(0));
        tracker.mark_visited(0);
        assert!(tracker.is_visited(0));
    }

    #[test]
    fn test_visited_epoch_wrapping() {
        let mut tracker = VisitedTracker::new(10);
        // Force epoch to u32::MAX
        tracker.current_epoch = u32::MAX;
        tracker.mark_visited(2);
        assert!(tracker.is_visited(2));

        // Advance: wrapping_add(1) gives 0 -> reset to 1 and clear buffer
        tracker.advance_epoch(10);
        assert_eq!(tracker.current_epoch, 1);
        assert!(!tracker.is_visited(2));

        tracker.mark_visited(2);
        assert!(tracker.is_visited(2));
    }

    #[test]
    fn test_visited_multiple_nodes() {
        let mut tracker = VisitedTracker::new(100);
        for id in (0..100).step_by(2) {
            tracker.mark_visited(id);
        }

        for id in 0..100 {
            if id % 2 == 0 {
                assert!(tracker.is_visited(id));
            } else {
                assert!(!tracker.is_visited(id));
            }
        }
    }

    #[test]
    fn test_visited_out_of_bounds_query_returns_false() {
        let tracker = VisitedTracker::new(10);
        assert!(!tracker.is_visited(10_000));
    }
}
