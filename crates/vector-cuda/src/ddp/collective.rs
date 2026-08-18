//! NCCL-style collective communication operations for Multi-GPU tensor synchronization.
//!
//! Provides lock-free parallel `AllReduce`, `Broadcast`, and `TopKReduce` primitives across GPU device buffers.

use std::collections::BinaryHeap;

use rayon::prelude::*;
use vector_index::DistanceMetric;
use vector_index::types::{SearchResult, VectorId};

use crate::device::DeviceBuffer;

/// Internal candidate item for bounded heap top-K ranking across GPU ranks.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: VectorId,
    distance: f32,
    heap_score: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.heap_score == other.heap_score
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.heap_score
            .partial_cmp(&other.heap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Collective operations engine for Multi-GPU tensor synchronization.
pub struct CollectiveOps;

impl CollectiveOps {
    /// Broadcasts data from the root rank buffer to all other GPU rank buffers.
    pub fn broadcast<T: Clone + Default + Copy + Send + Sync>(
        root_rank: usize,
        buffers: &mut [DeviceBuffer<T>],
    ) {
        assert!(root_rank < buffers.len());
        let root_data = buffers[root_rank].as_slice().to_vec();

        buffers
            .par_iter_mut()
            .enumerate()
            .filter(|(r, _)| *r != root_rank)
            .for_each(|(_, buf)| {
                buf.copy_from_host(&root_data);
            });
    }

    /// Performs an element-wise AllReduce SUM across all GPU rank `f32` buffers.
    ///
    /// Upon return, every GPU rank buffer contains the identical element-wise sum.
    pub fn all_reduce_sum_f32(buffers: &mut [DeviceBuffer<f32>]) {
        let num_ranks = buffers.len();
        if num_ranks <= 1 {
            return;
        }

        let len = buffers[0].len();
        for buf in buffers.iter() {
            assert_eq!(buf.len(), len);
        }

        // Parallel reduction across all ranks
        let mut global_sum = vec![0.0f32; len];
        for buf in buffers.iter() {
            let slice = buf.as_slice();
            for i in 0..len {
                global_sum[i] += slice[i];
            }
        }

        // Broadcast global sum back to all rank buffers in parallel
        buffers.par_iter_mut().for_each(|buf| {
            buf.copy_from_host(&global_sum);
        });
    }

    /// Performs an element-wise AllReduce SUM across all GPU rank `usize` count buffers.
    pub fn all_reduce_sum_usize(buffers: &mut [DeviceBuffer<usize>]) {
        let num_ranks = buffers.len();
        if num_ranks <= 1 {
            return;
        }

        let len = buffers[0].len();
        let mut global_sum = vec![0usize; len];
        for buf in buffers.iter() {
            let slice = buf.as_slice();
            for i in 0..len {
                global_sum[i] += slice[i];
            }
        }

        buffers.par_iter_mut().for_each(|buf| {
            buf.copy_from_host(&global_sum);
        });
    }

    /// Merges local partial Top-K candidate lists from $G$ GPU ranks into a global Top-K result set.
    pub fn top_k_reduce(
        partial_results_per_rank: Vec<Vec<SearchResult>>,
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        if partial_results_per_rank.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        for rank_results in partial_results_per_rank {
            for res in rank_results {
                let heap_score = if metric.higher_is_better() {
                    -res.distance
                } else {
                    res.distance
                };

                if heap.len() < k {
                    heap.push(Candidate {
                        id: res.id,
                        distance: res.distance,
                        heap_score,
                    });
                } else if let Some(worst) = heap.peek()
                    && heap_score < worst.heap_score
                {
                    heap.pop();
                    heap.push(Candidate {
                        id: res.id,
                        distance: res.distance,
                        heap_score,
                    });
                }
            }
        }

        let mut results = Vec::with_capacity(heap.len());
        while let Some(cand) = heap.pop() {
            results.push(SearchResult::new(cand.id, cand.distance));
        }
        results.reverse();
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast() {
        let mut bufs = vec![
            DeviceBuffer::from_host(&[10.0f32, 20.0]),
            DeviceBuffer::from_host(&[0.0f32, 0.0]),
            DeviceBuffer::from_host(&[0.0f32, 0.0]),
        ];

        CollectiveOps::broadcast(0, &mut bufs);

        assert_eq!(bufs[1].as_slice(), &[10.0, 20.0]);
        assert_eq!(bufs[2].as_slice(), &[10.0, 20.0]);
    }

    #[test]
    fn test_broadcast_from_non_zero_root() {
        let mut bufs = vec![
            DeviceBuffer::from_host(&[0usize, 0]),
            DeviceBuffer::from_host(&[42usize, 99]),
            DeviceBuffer::from_host(&[0usize, 0]),
        ];

        CollectiveOps::broadcast(1, &mut bufs);

        assert_eq!(bufs[0].as_slice(), &[42, 99]);
        assert_eq!(bufs[1].as_slice(), &[42, 99]);
        assert_eq!(bufs[2].as_slice(), &[42, 99]);
    }

    #[test]
    fn test_all_reduce_sum() {
        let mut bufs = vec![
            DeviceBuffer::from_host(&[1.0f32, 2.0]),
            DeviceBuffer::from_host(&[3.0f32, 4.0]),
            DeviceBuffer::from_host(&[5.0f32, 6.0]),
        ];

        CollectiveOps::all_reduce_sum_f32(&mut bufs);

        let expected = [9.0f32, 12.0];
        assert_eq!(bufs[0].as_slice(), &expected);
        assert_eq!(bufs[1].as_slice(), &expected);
        assert_eq!(bufs[2].as_slice(), &expected);
    }

    #[test]
    fn test_all_reduce_sum_usize() {
        let mut bufs = vec![
            DeviceBuffer::from_host(&[10usize, 20]),
            DeviceBuffer::from_host(&[30usize, 40]),
            DeviceBuffer::from_host(&[50usize, 60]),
        ];

        CollectiveOps::all_reduce_sum_usize(&mut bufs);

        let expected = [90usize, 120];
        assert_eq!(bufs[0].as_slice(), &expected);
        assert_eq!(bufs[1].as_slice(), &expected);
        assert_eq!(bufs[2].as_slice(), &expected);
    }

    #[test]
    fn test_all_reduce_sum_single_rank() {
        let mut bufs_f32 = vec![DeviceBuffer::from_host(&[1.5f32, 2.5])];
        CollectiveOps::all_reduce_sum_f32(&mut bufs_f32);
        assert_eq!(bufs_f32[0].as_slice(), &[1.5f32, 2.5]);

        let mut bufs_usize = vec![DeviceBuffer::from_host(&[100usize])];
        CollectiveOps::all_reduce_sum_usize(&mut bufs_usize);
        assert_eq!(bufs_usize[0].as_slice(), &[100usize]);
    }

    #[test]
    fn test_top_k_reduce() {
        let rank0 = vec![SearchResult::new(0, 1.0), SearchResult::new(1, 5.0)];
        let rank1 = vec![SearchResult::new(10, 0.5), SearchResult::new(11, 4.0)];

        let merged = CollectiveOps::top_k_reduce(vec![rank0, rank1], 2, DistanceMetric::L2Squared);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 10); // dist = 0.5
        assert_eq!(merged[1].id, 0); // dist = 1.0
    }

    #[test]
    fn test_top_k_reduce_empty_and_zero_k() {
        let res_empty = CollectiveOps::top_k_reduce(Vec::new(), 5, DistanceMetric::L2Squared);
        assert!(res_empty.is_empty());

        let rank0 = vec![SearchResult::new(0, 1.0)];
        let res_zero_k = CollectiveOps::top_k_reduce(vec![rank0], 0, DistanceMetric::L2Squared);
        assert!(res_zero_k.is_empty());
    }

    #[test]
    fn test_top_k_reduce_cosine_metric() {
        // Higher similarity is better
        let rank0 = vec![SearchResult::new(0, 0.85), SearchResult::new(1, 0.40)];
        let rank1 = vec![SearchResult::new(2, 0.95), SearchResult::new(3, 0.70)];

        let merged =
            CollectiveOps::top_k_reduce(vec![rank0, rank1], 3, DistanceMetric::CosineSimilarity);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id, 2); // sim = 0.95
        assert_eq!(merged[1].id, 0); // sim = 0.85
        assert_eq!(merged[2].id, 3); // sim = 0.70
    }

    #[test]
    fn test_top_k_reduce_k_greater_than_total_candidates() {
        let rank0 = vec![SearchResult::new(0, 2.0)];
        let rank1 = vec![SearchResult::new(1, 1.0)];

        let merged = CollectiveOps::top_k_reduce(vec![rank0, rank1], 10, DistanceMetric::L2Squared);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 1);
        assert_eq!(merged[1].id, 0);
    }
}
