//! Index construction algorithms for HNSW (Sequential and Parallel).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::Ordering as AtomicOrdering;

use rand::Rng;
use rayon::prelude::*;
use vector_simd::DistanceEngine;

use super::config::HnswConfig;
use super::graph::HnswGraph;
use super::visited::VisitedTracker;
use crate::DistanceMetric;
use crate::storage::VectorStorage;
use crate::types::VectorId;

/// Min-heap candidate entry: sorted by distance ascending (closest element on top).
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub id: VectorId,
    pub dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap acts as a Min-Heap (lowest dist first)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap result entry: sorted by distance descending (farthest element on top for eviction).
#[derive(Debug, Clone, Copy)]
pub struct ResultEntry {
    pub id: VectorId,
    pub dist: f32,
}

impl PartialEq for ResultEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for ResultEntry {}

impl PartialOrd for ResultEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResultEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Builder for constructing an HNSW graph over a [`VectorStorage`] backend.
pub struct HnswBuilder<'a, S: VectorStorage> {
    storage: &'a S,
    config: HnswConfig,
    metric: DistanceMetric,
    engine: DistanceEngine,
}

impl<'a, S: VectorStorage> HnswBuilder<'a, S> {
    /// Creates a new `HnswBuilder`.
    pub fn new(storage: &'a S, config: HnswConfig, metric: DistanceMetric) -> Self {
        Self {
            storage,
            config,
            metric,
            engine: DistanceEngine::auto(),
        }
    }

    /// Computes distance between two vectors by their storage IDs.
    #[inline]
    pub fn distance_by_id(&self, a: VectorId, b: VectorId) -> f32 {
        let va = self.storage.get(a);
        let vb = self.storage.get(b);
        self.compute_distance(va, vb)
    }

    /// Computes distance between vector slice and a stored vector.
    #[inline]
    pub fn distance_to_query(&self, query: &[f32], target_id: VectorId) -> f32 {
        let vt = self.storage.get(target_id);
        self.compute_distance(query, vt)
    }

    /// Computes raw distance metric according to the index configuration.
    #[inline]
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2Squared => self.engine.l2_squared(a, b),
            DistanceMetric::DotProduct => -self.engine.dot_product(a, b),
            DistanceMetric::CosineSimilarity => 1.0 - self.engine.cosine_similarity(a, b),
            DistanceMetric::Manhattan => self.engine.manhattan(a, b),
            DistanceMetric::Minkowski => self.engine.minkowski(a, b, 3.0),
            DistanceMetric::Chebyshev => self.engine.chebyshev(a, b),
            DistanceMetric::Hamming => self.engine.hamming(a, b),
            DistanceMetric::Mahalanobis => self.engine.mahalanobis(a, b),
            DistanceMetric::Jaccard => self.engine.jaccard(a, b),
            DistanceMetric::Hellinger => self.engine.hellinger(a, b),
        }
    }

    /// Samples a random level for a new node using the configured level multiplier $m_l$.
    pub fn random_level(&self, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.gen_range(0.0..1.0);
        // Avoid ln(0)
        let r = r.max(1e-10);
        (-r.ln() * self.config.ml).floor() as usize
    }

    /// Builds the HNSW graph sequentially for all vectors in storage.
    pub fn build(&self) -> HnswGraph {
        let n = self.storage.len();
        let mut graph = HnswGraph::new(self.config.clone(), n);
        let mut rng = rand::thread_rng();

        let mut tracker = VisitedTracker::new(n);

        for id in 0..n as u32 {
            let level = self.random_level(&mut rng);
            graph.init_node(id, level);
            self.insert_node(&graph, id, level, &mut tracker);
        }

        graph
    }

    /// Builds the HNSW graph using multi-threaded parallel insertion with `rayon`.
    pub fn build_parallel(&self) -> HnswGraph {
        let n = self.storage.len();
        let mut graph = HnswGraph::new(self.config.clone(), n);

        // Pre-generate levels deterministically or with thread-safe RNG to initialize node structures
        let levels: Vec<usize> = {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| self.random_level(&mut rng)).collect()
        };

        for (id, &level) in levels.iter().enumerate() {
            graph.init_node(id as u32, level);
        }

        // Insert first node to establish initial entry point
        if n > 0 {
            let mut tracker = VisitedTracker::new(n);
            self.insert_node(&graph, 0, levels[0], &mut tracker);
        }

        // Insert remaining nodes in parallel
        (1..n as u32).into_par_iter().for_each(|id| {
            let level = levels[id as usize];
            let mut local_tracker = VisitedTracker::new(n);
            self.insert_node(&graph, id, level, &mut local_tracker);
        });

        graph
    }

    /// Inserts a single node into the HNSW graph.
    pub fn insert_node(
        &self,
        graph: &HnswGraph,
        id: VectorId,
        node_level: usize,
        tracker: &mut VisitedTracker,
    ) {
        let query = self.storage.get(id);
        let dist_fn = |a: VectorId, b: VectorId| self.distance_by_id(a, b);

        let mut curr_ep = {
            let ep_lock = graph.entry_point.read();
            *ep_lock
        };

        if curr_ep.is_none() {
            let mut ep_lock = graph.entry_point.write();
            if ep_lock.is_none() {
                *ep_lock = Some(id);
                *graph.max_level.write() = node_level;
                graph.num_nodes.fetch_add(1, AtomicOrdering::Relaxed);
                return;
            }
            curr_ep = *ep_lock;
        }

        let mut curr_obj = curr_ep.unwrap();
        let max_l = *graph.max_level.read();

        let mut curr_dist = self.distance_to_query(query, curr_obj);

        // Phase 1: Greedy search on top layers down to node_level + 1
        for l in (node_level + 1..=max_l).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = graph.get_neighbors(curr_obj, l);
                for n_id in neighbors {
                    let d = self.distance_to_query(query, n_id);
                    if d < curr_dist {
                        curr_dist = d;
                        curr_obj = n_id;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: From min(node_level, max_l) down to layer 0
        let bottom_l = node_level.min(max_l);
        let mut enter_points = vec![curr_obj];

        for l in (0..=bottom_l).rev() {
            let candidates = self.search_layer_internal(
                graph,
                query,
                &enter_points,
                self.config.ef_construction,
                l,
                tracker,
            );

            let mut sorted_candidates = candidates;
            sorted_candidates
                .sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

            if let Some(closest) = sorted_candidates.first() {
                enter_points = vec![closest.id];
            }

            // Select neighbors to connect
            let max_m = graph.max_edges_for_layer(l);
            let selected_neighbors = self.select_neighbors(sorted_candidates, max_m, &dist_fn);

            // Add bidirectional edges
            for &neighbor_id in &selected_neighbors {
                graph.add_edge(id, neighbor_id, l, dist_fn);
            }
        }

        // Update global entry point if new node has a higher level
        if node_level > max_l {
            let mut ep_lock = graph.entry_point.write();
            let mut ml_lock = graph.max_level.write();
            if node_level > *ml_lock {
                *ep_lock = Some(id);
                *ml_lock = node_level;
            }
        }

        graph.num_nodes.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Performs search on a single layer with a given beam width `ef`.
    pub fn search_layer_internal(
        &self,
        graph: &HnswGraph,
        query: &[f32],
        enter_points: &[VectorId],
        ef: usize,
        layer: usize,
        tracker: &mut VisitedTracker,
    ) -> Vec<ResultEntry> {
        tracker.advance_epoch(self.storage.len());

        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<ResultEntry> = BinaryHeap::with_capacity(ef + 1);

        for &ep_id in enter_points {
            let dist = self.distance_to_query(query, ep_id);
            tracker.mark_visited(ep_id);
            candidates.push(Candidate { id: ep_id, dist });
            results.push(ResultEntry { id: ep_id, dist });
        }

        while let Some(Candidate {
            id: c_id,
            dist: c_dist,
        }) = candidates.pop()
        {
            // If closest candidate is farther than the farthest result in top ef, stop
            if let Some(farthest) = results.peek()
                && c_dist > farthest.dist
                && results.len() >= ef
            {
                break;
            }

            let neighbors = graph.get_neighbors(c_id, layer);
            for n_id in neighbors {
                if !tracker.is_visited(n_id) {
                    tracker.mark_visited(n_id);

                    let dist = self.distance_to_query(query, n_id);
                    let should_insert = if results.len() < ef {
                        true
                    } else if let Some(farthest) = results.peek() {
                        dist < farthest.dist
                    } else {
                        false
                    };

                    if should_insert {
                        candidates.push(Candidate { id: n_id, dist });
                        results.push(ResultEntry { id: n_id, dist });
                        if results.len() > ef {
                            results.pop(); // Evicts the farthest
                        }
                    }
                }
            }
        }

        results.into_vec()
    }

    /// Selects neighbors from a list of candidate entries.
    fn select_neighbors(
        &self,
        mut candidates: Vec<ResultEntry>,
        max_m: usize,
        dist_fn: &impl Fn(VectorId, VectorId) -> f32,
    ) -> Vec<VectorId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort ascending by distance
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        if !self.config.use_heuristic {
            candidates.truncate(max_m);
            return candidates.into_iter().map(|e| e.id).collect();
        }

        let mut selected: Vec<VectorId> = Vec::with_capacity(max_m);
        for cand in &candidates {
            if selected.len() >= max_m {
                break;
            }

            let mut is_diverse = true;
            for &sel_id in &selected {
                let dist_to_sel = dist_fn(cand.id, sel_id);
                if dist_to_sel <= cand.dist {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(cand.id);
            }
        }

        // Fallback fill to reach max_m if possible
        if selected.len() < max_m {
            for cand in &candidates {
                if selected.len() >= max_m {
                    break;
                }
                if !selected.contains(&cand.id) {
                    selected.push(cand.id);
                }
            }
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::HeapStorage;

    #[test]
    fn test_builder_sequential() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[0.0, 0.0]).unwrap();
        storage.push(&[1.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0]).unwrap();
        storage.push(&[1.0, 1.0]).unwrap();

        let config = HnswConfig::new(4, 20, 20);
        let builder = HnswBuilder::new(&storage, config, DistanceMetric::L2Squared);
        let graph = builder.build();

        assert!(graph.entry_point.read().is_some());
        assert_eq!(graph.num_nodes.load(AtomicOrdering::Relaxed), 4);
    }

    #[test]
    fn test_builder_parallel() {
        let mut storage = HeapStorage::new(2);
        for i in 0..50 {
            storage.push(&[i as f32, (i * 2) as f32]).unwrap();
        }

        let config = HnswConfig::new(4, 20, 20);
        let builder = HnswBuilder::new(&storage, config, DistanceMetric::L2Squared);
        let graph = builder.build_parallel();

        assert!(graph.entry_point.read().is_some());
        assert_eq!(graph.num_nodes.load(AtomicOrdering::Relaxed), 50);
    }

    #[test]
    fn test_candidate_min_heap_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate { id: 1, dist: 5.0 });
        heap.push(Candidate { id: 2, dist: 1.0 });
        heap.push(Candidate { id: 3, dist: 3.0 });

        // Min-heap: lowest distance popped first
        assert_eq!(heap.pop().unwrap().id, 2); // dist 1.0
        assert_eq!(heap.pop().unwrap().id, 3); // dist 3.0
        assert_eq!(heap.pop().unwrap().id, 1); // dist 5.0
    }

    #[test]
    fn test_result_entry_max_heap_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(ResultEntry { id: 1, dist: 5.0 });
        heap.push(ResultEntry { id: 2, dist: 1.0 });
        heap.push(ResultEntry { id: 3, dist: 3.0 });

        // Max-heap: highest distance popped first
        assert_eq!(heap.pop().unwrap().id, 1); // dist 5.0
        assert_eq!(heap.pop().unwrap().id, 3); // dist 3.0
        assert_eq!(heap.pop().unwrap().id, 2); // dist 1.0
    }

    #[test]
    fn test_distance_metrics_in_builder() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0]).unwrap();

        let builder_l2 =
            HnswBuilder::new(&storage, HnswConfig::default(), DistanceMetric::L2Squared);
        assert!((builder_l2.distance_by_id(0, 1) - 2.0).abs() < 1e-4);

        let builder_dot =
            HnswBuilder::new(&storage, HnswConfig::default(), DistanceMetric::DotProduct);
        assert!((builder_dot.distance_by_id(0, 1) - 0.0).abs() < 1e-4);

        let builder_cos = HnswBuilder::new(
            &storage,
            HnswConfig::default(),
            DistanceMetric::CosineSimilarity,
        );
        assert!((builder_cos.distance_by_id(0, 1) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_random_level_generation_distribution() {
        let storage = HeapStorage::new(2);
        let config = HnswConfig::new(16, 100, 50);
        let builder = HnswBuilder::new(&storage, config, DistanceMetric::L2Squared);

        let mut rng = rand::thread_rng();
        let mut level_counts = [0usize; 10];

        for _ in 0..10_000 {
            let l = builder.random_level(&mut rng);
            if l < 10 {
                level_counts[l] += 1;
            }
        }

        // Level 0 should have the vast majority of nodes
        assert!(level_counts[0] > level_counts[1]);
        assert!(level_counts[1] > level_counts[2]);
        assert!(level_counts[0] > 8000, "Level 0 count: {}", level_counts[0]);
    }

    #[test]
    fn test_builder_single_node() {
        let mut storage = HeapStorage::new(3);
        storage.push(&[1.0, 2.0, 3.0]).unwrap();

        let builder = HnswBuilder::new(&storage, HnswConfig::default(), DistanceMetric::L2Squared);
        let graph = builder.build();

        assert_eq!(*graph.entry_point.read(), Some(0));
        assert_eq!(graph.num_nodes.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn test_select_neighbors_empty_and_small() {
        let storage = HeapStorage::new(2);
        let builder = HnswBuilder::new(&storage, HnswConfig::default(), DistanceMetric::L2Squared);
        let dist_fn = |a: VectorId, b: VectorId| ((a as f32) - (b as f32)).abs();

        // Empty candidates
        let selected = builder.select_neighbors(vec![], 4, &dist_fn);
        assert!(selected.is_empty());

        // Fewer candidates than max_m
        let candidates = vec![
            ResultEntry { id: 1, dist: 2.0 },
            ResultEntry { id: 2, dist: 1.0 },
        ];
        let selected = builder.select_neighbors(candidates, 4, &dist_fn);
        assert_eq!(selected.len(), 2);
    }
}
