//! Graph data structure and neighbor selection algorithms for HNSW.

use parking_lot::RwLock;
use std::sync::atomic::AtomicUsize;

use super::config::HnswConfig;
use crate::types::VectorId;

/// A single node's neighbor lists across all layers it participates in.
///
/// Layer 0 is at index 0, Layer 1 at index 1, up to the node's maximum layer.
#[derive(Default, Debug)]
pub struct NodeLinks {
    /// Neighbor lists per layer, protected by `RwLock` for safe concurrent updates during graph construction.
    pub layers: Vec<RwLock<Vec<VectorId>>>,
}

impl NodeLinks {
    /// Creates a new `NodeLinks` with `level + 1` layers.
    pub fn new(level: usize) -> Self {
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(RwLock::new(Vec::new()));
        }
        Self { layers }
    }

    /// Returns the maximum level this node participates in.
    #[inline]
    pub fn max_level(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }
}

/// The HNSW multi-layer proximity graph.
///
/// Encapsulates the hierarchical graph topology, entry point tracking,
/// and fine-grained locking for multithreaded construction.
pub struct HnswGraph {
    /// Configuration settings (M, M0, ef_construction, etc.).
    pub config: HnswConfig,
    /// Vector of nodes and their neighbor links per layer.
    pub nodes: Vec<NodeLinks>,
    /// Global entry point (node with the highest level in the graph).
    pub entry_point: RwLock<Option<VectorId>>,
    /// Current maximum level present in the graph.
    pub max_level: RwLock<usize>,
    /// Number of nodes currently inserted into the graph.
    pub num_nodes: AtomicUsize,
}

impl std::fmt::Debug for HnswGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswGraph")
            .field("config", &self.config)
            .field("num_nodes", &self.nodes.len())
            .field("entry_point", &*self.entry_point.read())
            .field("max_level", &*self.max_level.read())
            .finish()
    }
}

impl HnswGraph {
    /// Creates a new empty `HnswGraph` with the given configuration and pre-allocated capacity.
    pub fn new(config: HnswConfig, capacity: usize) -> Self {
        Self {
            config,
            nodes: (0..capacity).map(|_| NodeLinks::default()).collect(),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            num_nodes: AtomicUsize::new(0),
        }
    }

    /// Allocates node slot with specified level.
    pub fn init_node(&mut self, id: VectorId, level: usize) {
        let idx = id as usize;
        if idx >= self.nodes.len() {
            self.nodes.resize_with(idx + 1, NodeLinks::default);
        }
        self.nodes[idx] = NodeLinks::new(level);
    }

    /// Returns the maximum allowed neighbors for a given layer.
    #[inline]
    pub fn max_edges_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m0
        } else {
            self.config.m
        }
    }

    /// Returns the neighbor IDs of node `id` at `layer`.
    ///
    /// Copies neighbor IDs to avoid holding a read lock across distance computations.
    #[inline]
    pub fn get_neighbors(&self, id: VectorId, layer: usize) -> Vec<VectorId> {
        let idx = id as usize;
        if let Some(node) = self.nodes.get(idx)
            && let Some(layer_lock) = node.layers.get(layer)
        {
            return layer_lock.read().clone();
        }
        Vec::new()
    }

    /// Adds a bidirectional connection between `u` and `v` at `layer`.
    /// If neighbor capacity is exceeded, trims connections.
    pub fn add_edge(
        &self,
        u: VectorId,
        v: VectorId,
        layer: usize,
        dist_fn: impl Fn(VectorId, VectorId) -> f32,
    ) {
        self.add_directed_edge(u, v, layer, &dist_fn);
        self.add_directed_edge(v, u, layer, &dist_fn);
    }

    /// Adds a directed connection from `src` to `dst` at `layer`.
    pub fn add_directed_edge(
        &self,
        src: VectorId,
        dst: VectorId,
        layer: usize,
        dist_fn: &impl Fn(VectorId, VectorId) -> f32,
    ) {
        if src == dst {
            return;
        }

        let max_m = self.max_edges_for_layer(layer);
        let src_idx = src as usize;
        if let Some(node) = self.nodes.get(src_idx)
            && let Some(layer_lock) = node.layers.get(layer)
        {
            let mut neighbors = layer_lock.write();
            if !neighbors.contains(&dst) {
                neighbors.push(dst);
                if neighbors.len() > max_m {
                    // Shrink neighbor list according to heuristic or simple nearest
                    self.shrink_neighbors(&mut neighbors, src, max_m, dist_fn);
                }
            }
        }
    }

    /// Shrinks the neighbor list to at most `max_m` elements.
    fn shrink_neighbors(
        &self,
        neighbors: &mut Vec<VectorId>,
        base_id: VectorId,
        max_m: usize,
        dist_fn: &impl Fn(VectorId, VectorId) -> f32,
    ) {
        if neighbors.len() <= max_m {
            return;
        }

        if self.config.use_heuristic {
            // Heuristic selection (Algorithm 4 in HNSW paper)
            let mut candidates: Vec<(VectorId, f32)> = neighbors
                .iter()
                .map(|&n| (n, dist_fn(base_id, n)))
                .collect();

            // Sort by distance to base_id ascending
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut selected: Vec<VectorId> = Vec::with_capacity(max_m);
            for &(cand_id, cand_dist) in &candidates {
                if selected.len() >= max_m {
                    break;
                }

                // Check diversity: candidate is accepted if it is closer to base_id
                // than to any already selected neighbor.
                let mut is_diverse = true;
                for &sel_id in &selected {
                    let dist_to_sel = dist_fn(cand_id, sel_id);
                    if dist_to_sel <= cand_dist {
                        is_diverse = false;
                        break;
                    }
                }

                if is_diverse {
                    selected.push(cand_id);
                }
            }

            // If selected is less than max_m, fill remaining with closest candidates
            if selected.len() < max_m {
                for &(cand_id, _) in &candidates {
                    if selected.len() >= max_m {
                        break;
                    }
                    if !selected.contains(&cand_id) {
                        selected.push(cand_id);
                    }
                }
            }

            *neighbors = selected;
        } else {
            // Simple closest neighbor selection
            neighbors.sort_by(|&a, &b| {
                let da = dist_fn(base_id, a);
                let db = dist_fn(base_id, b);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            neighbors.truncate(max_m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_links_creation() {
        let node = NodeLinks::new(3);
        assert_eq!(node.max_level(), 3);
        assert_eq!(node.layers.len(), 4);
    }

    #[test]
    fn test_graph_add_edges_and_shrink() {
        let config = HnswConfig::new(2, 10, 10).with_m0(3);
        let mut graph = HnswGraph::new(config, 10);
        graph.init_node(0, 0);
        graph.init_node(1, 0);
        graph.init_node(2, 0);
        graph.init_node(3, 0);
        graph.init_node(4, 0);

        let dist_fn = |a: VectorId, b: VectorId| -> f32 { ((a as f32) - (b as f32)).abs() };

        // Add 4 edges to node 0 at layer 0 (max m0 = 3)
        graph.add_edge(0, 1, 0, dist_fn);
        graph.add_edge(0, 2, 0, dist_fn);
        graph.add_edge(0, 3, 0, dist_fn);
        graph.add_edge(0, 4, 0, dist_fn);

        let neighbors_0 = graph.get_neighbors(0, 0);
        assert!(
            neighbors_0.len() <= 3,
            "neighbors must be <= m0: {:?}",
            neighbors_0
        );
        // Closest neighbors to 0 are 1, 2, 3
        assert!(neighbors_0.contains(&1));
    }

    #[test]
    fn test_graph_self_edge_prevention() {
        let config = HnswConfig::new(4, 10, 10);
        let mut graph = HnswGraph::new(config, 5);
        graph.init_node(0, 0);

        let dist_fn = |a: VectorId, b: VectorId| ((a as f32) - (b as f32)).abs();
        graph.add_edge(0, 0, 0, dist_fn);

        let neighbors = graph.get_neighbors(0, 0);
        assert!(neighbors.is_empty(), "node must not have self-edge");
    }

    #[test]
    fn test_graph_duplicate_edge_prevention() {
        let config = HnswConfig::new(4, 10, 10);
        let mut graph = HnswGraph::new(config, 5);
        graph.init_node(0, 0);
        graph.init_node(1, 0);

        let dist_fn = |a: VectorId, b: VectorId| ((a as f32) - (b as f32)).abs();
        graph.add_edge(0, 1, 0, dist_fn);
        graph.add_edge(0, 1, 0, dist_fn);

        let neighbors = graph.get_neighbors(0, 0);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], 1);
    }

    #[test]
    fn test_graph_get_neighbors_non_existent() {
        let config = HnswConfig::new(4, 10, 10);
        let mut graph = HnswGraph::new(config, 5);
        graph.init_node(0, 1);

        // Non-existent node
        assert!(graph.get_neighbors(99, 0).is_empty());
        // Non-existent layer for node 0 (which has level 1, so layer 5 does not exist)
        assert!(graph.get_neighbors(0, 5).is_empty());
    }

    #[test]
    fn test_graph_max_edges_for_layer() {
        let config = HnswConfig::new(16, 100, 50).with_m0(32);
        let graph = HnswGraph::new(config, 10);

        assert_eq!(graph.max_edges_for_layer(0), 32);
        assert_eq!(graph.max_edges_for_layer(1), 16);
        assert_eq!(graph.max_edges_for_layer(5), 16);
    }

    #[test]
    fn test_graph_shrink_neighbors_without_heuristic() {
        let config = HnswConfig::new(2, 10, 10).with_m0(2).with_heuristic(false);
        let mut graph = HnswGraph::new(config, 10);
        graph.init_node(0, 0);
        graph.init_node(1, 0);
        graph.init_node(2, 0);
        graph.init_node(3, 0);

        let dist_fn = |a: VectorId, b: VectorId| ((a as f32) - (b as f32)).abs();
        // Add nodes at distances: 1 (dist=1), 2 (dist=2), 3 (dist=3)
        graph.add_edge(0, 3, 0, dist_fn);
        graph.add_edge(0, 1, 0, dist_fn);
        graph.add_edge(0, 2, 0, dist_fn);

        let neighbors = graph.get_neighbors(0, 0);
        assert_eq!(neighbors.len(), 2);
        // Closest 2 are 1 and 2
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&3));
    }
}
