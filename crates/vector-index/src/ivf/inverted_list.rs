//! Inverted List storage structure for IVF partitions.
//!
//! Stores vector IDs and compact PQ byte codes in cache-friendly contiguous buffers per Voronoi cluster.

use crate::types::VectorId;

/// Single inverted list corresponding to one coarse centroid partition.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InvertedList {
    /// Vector identifiers assigned to this cluster.
    pub ids: Vec<VectorId>,
    /// Flat buffer containing contiguous $M$-byte PQ codes for all vectors in this list.
    /// Total bytes = `ids.len() * num_subvectors`.
    pub codes: Vec<u8>,
}

impl InvertedList {
    /// Creates an empty inverted list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty inverted list with reserved capacity.
    pub fn with_capacity(capacity: usize, num_subvectors: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            codes: Vec::with_capacity(capacity * num_subvectors),
        }
    }

    /// Appends a vector and its quantized code to this cluster list.
    #[inline]
    pub fn push(&mut self, id: VectorId, code: &[u8]) {
        self.ids.push(id);
        self.codes.extend_from_slice(code);
    }

    /// Returns the number of vectors in this list.
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Returns `true` if the list contains no vectors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Returns the $M$-byte code for the $i$-th vector in this list.
    #[inline]
    pub fn get_code(&self, index: usize, num_subvectors: usize) -> &[u8] {
        let start = index * num_subvectors;
        &self.codes[start..start + num_subvectors]
    }
}

/// Collection of $C$ inverted lists representing the entire partitioned dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct InvertedIndex {
    /// Inverted lists per coarse cluster centroid.
    pub lists: Vec<InvertedList>,
    /// Number of sub-vectors per quantized entry ($M$).
    pub num_subvectors: usize,
    /// Total number of indexed vectors across all lists.
    pub total_vectors: usize,
}

impl InvertedIndex {
    /// Creates a new `InvertedIndex` for `nlist` clusters with $M$ sub-vectors.
    pub fn new(nlist: usize, num_subvectors: usize) -> Self {
        let mut lists = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            lists.push(InvertedList::new());
        }

        Self {
            lists,
            num_subvectors,
            total_vectors: 0,
        }
    }

    /// Appends a vector to the specified cluster's inverted list.
    #[inline]
    pub fn add(&mut self, cluster_id: usize, id: VectorId, code: &[u8]) {
        assert_eq!(code.len(), self.num_subvectors);
        self.lists[cluster_id].push(id, code);
        self.total_vectors += 1;
    }

    /// Returns the total number of indexed vectors across all clusters.
    #[inline]
    pub fn total_vectors(&self) -> usize {
        self.total_vectors
    }

    /// Returns the number of coarse clusters ($nlist$).
    #[inline]
    pub fn nlist(&self) -> usize {
        self.lists.len()
    }

    /// Returns a reference to the inverted list for cluster `cluster_id`.
    #[inline]
    pub fn get_list(&self, cluster_id: usize) -> &InvertedList {
        &self.lists[cluster_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_list_push_and_get() {
        let num_subvectors = 4;
        let mut list = InvertedList::with_capacity(10, num_subvectors);

        list.push(101, &[1, 2, 3, 4]);
        list.push(102, &[5, 6, 7, 8]);

        assert_eq!(list.len(), 2);
        assert_eq!(list.ids, vec![101, 102]);
        assert_eq!(list.get_code(0, num_subvectors), &[1, 2, 3, 4]);
        assert_eq!(list.get_code(1, num_subvectors), &[5, 6, 7, 8]);
    }

    #[test]
    fn test_inverted_index() {
        let mut inv_index = InvertedIndex::new(4, 2);

        inv_index.add(0, 0, &[10, 20]);
        inv_index.add(0, 1, &[11, 21]);
        inv_index.add(2, 2, &[30, 40]);

        assert_eq!(inv_index.total_vectors(), 3);
        assert_eq!(inv_index.nlist(), 4);
        assert_eq!(inv_index.get_list(0).len(), 2);
        assert_eq!(inv_index.get_list(1).len(), 0);
        assert_eq!(inv_index.get_list(2).len(), 1);
        assert_eq!(inv_index.get_list(3).len(), 0);
    }

    #[test]
    fn test_inverted_list_empty_and_default() {
        let list1 = InvertedList::new();
        let list2 = InvertedList::default();
        assert_eq!(list1, list2);
        assert!(list1.is_empty());
        assert_eq!(list1.len(), 0);

        let debug_str = format!("{:?}", list1);
        assert!(debug_str.contains("InvertedList"));

        let cloned = list1.clone();
        assert_eq!(cloned, list1);
    }

    #[test]
    fn test_inverted_index_accessors_and_clone() {
        let mut inv_index = InvertedIndex::new(3, 2);
        assert_eq!(inv_index.nlist(), 3);
        assert_eq!(inv_index.total_vectors(), 0);

        inv_index.add(1, 42, &[7, 8]);
        assert_eq!(inv_index.total_vectors(), 1);
        assert_eq!(inv_index.get_list(1).ids, vec![42]);
        assert_eq!(inv_index.get_list(1).get_code(0, 2), &[7, 8]);

        let cloned = inv_index.clone();
        assert_eq!(cloned, inv_index);
    }
}
