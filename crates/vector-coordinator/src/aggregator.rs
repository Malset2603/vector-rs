//! Async Scatter-Gather query dispatcher and Top-K binary heap aggregator.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use tokio::task::JoinSet;
use vector_proto::{
    ClusterSearchResponse, DistanceMetric as ProtoDistanceMetric, SearchRequest, SearchResultItem,
};

use crate::router::WorkerRouter;

/// Internal entry for Top-K heap aggregation.
#[derive(Debug, Clone)]
struct HeapItem {
    item: SearchResultItem,
    higher_is_better: bool,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.item.distance == other.item.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // In BinaryHeap (which is a max-heap), the root is the "worst" element that should be evicted.
        if self.higher_is_better {
            // For similarity (higher is better): worst is the lowest score -> Min-Heap behavior.
            other
                .item
                .distance
                .partial_cmp(&self.item.distance)
                .unwrap_or(Ordering::Equal)
        } else {
            // For distance (lower is better): worst is the highest score -> Max-Heap behavior.
            self.item
                .distance
                .partial_cmp(&other.item.distance)
                .unwrap_or(Ordering::Equal)
        }
    }
}

/// Scatter-gather query orchestrator and result aggregator.
#[derive(Clone)]
pub struct ScatterGatherAggregator {
    router: Arc<WorkerRouter>,
}

impl ScatterGatherAggregator {
    /// Creates a new `ScatterGatherAggregator` over the given router.
    pub fn new(router: Arc<WorkerRouter>) -> Self {
        Self { router }
    }

    /// Dispatches vector query to all registered workers in parallel and aggregates the global Top-K.
    pub async fn search_cluster(
        &self,
        query: Vec<f32>,
        k: usize,
        ef_search: usize,
        metric: ProtoDistanceMetric,
    ) -> ClusterSearchResponse {
        self.search_cluster_with_error(query, k, ef_search, metric)
            .await
            .0
    }

    /// Dispatches search requests to all registered workers, aggregates Top-K results,
    /// and captures any shard-level gRPC error status.
    pub async fn search_cluster_with_error(
        &self,
        query: Vec<f32>,
        k: usize,
        ef_search: usize,
        metric: ProtoDistanceMetric,
    ) -> (ClusterSearchResponse, Option<tonic::Status>) {
        let start = Instant::now();
        let clients = self.router.get_clients();
        let total_queried = clients.len() as u32;

        if clients.is_empty() || k == 0 {
            return (
                ClusterSearchResponse {
                    results: Vec::new(),
                    total_queried_shards: total_queried,
                    successful_shards: 0,
                    query_latency_micros: start.elapsed().as_micros() as u64,
                },
                None,
            );
        }

        // Scatter: Dispatch search requests concurrently to all workers via JoinSet
        tracing::debug!(
            target_workers = clients.len(),
            k,
            ef_search,
            "Scattering search request to worker nodes"
        );

        let mut join_set = JoinSet::new();

        for mut client in clients {
            let req = SearchRequest {
                query_vector: query.clone(),
                k: k as u32,
                ef_search: ef_search as u32,
                metric: metric as i32,
                shard_id: 0,
            };

            join_set.spawn(async move { client.search(tonic::Request::new(req)).await });
        }

        // Gather: Collect partial results from workers
        let higher_is_better = matches!(
            metric,
            ProtoDistanceMetric::DotProduct | ProtoDistanceMetric::CosineSimilarity
        );

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        let mut successful_shards = 0u32;
        let mut last_error_status: Option<tonic::Status> = None;

        while let Some(join_res) = join_set.join_next().await {
            match join_res {
                Ok(Ok(resp)) => {
                    successful_shards += 1;
                    let search_resp = resp.into_inner();

                    tracing::debug!(
                        shard_id = search_resp.shard_id,
                        results_returned = search_resp.results.len(),
                        shard_latency_micros = search_resp.query_latency_micros,
                        "Received shard search response"
                    );

                    for item in search_resp.results {
                        let heap_item = HeapItem {
                            item,
                            higher_is_better,
                        };

                        if heap.len() < k {
                            heap.push(heap_item);
                        } else if let Some(worst) = heap.peek() {
                            let is_better = if higher_is_better {
                                heap_item.item.distance > worst.item.distance
                            } else {
                                heap_item.item.distance < worst.item.distance
                            };

                            if is_better {
                                heap.pop();
                                heap.push(heap_item);
                            }
                        }
                    }
                }
                Ok(Err(status)) => {
                    tracing::warn!(
                        code = ?status.code(),
                        error = %status.message(),
                        "Worker shard returned gRPC error"
                    );
                    last_error_status = Some(status);
                }
                Err(join_err) => {
                    tracing::error!(
                        error = %join_err,
                        "Worker request dispatch task panicked or was cancelled"
                    );
                }
            }
        }

        // Extract and sort results from best to worst
        let mut final_results: Vec<SearchResultItem> = heap.into_iter().map(|h| h.item).collect();

        if higher_is_better {
            final_results.sort_by(|a, b| {
                b.distance
                    .partial_cmp(&a.distance)
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            final_results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
        }

        let elapsed = start.elapsed().as_micros() as u64;

        tracing::info!(
            queried_shards = total_queried,
            successful_shards,
            merged_results = final_results.len(),
            latency_micros = elapsed,
            "Scatter-gather query aggregation completed"
        );

        (
            ClusterSearchResponse {
                results: final_results,
                total_queried_shards: total_queried,
                successful_shards,
                query_latency_micros: elapsed,
            },
            last_error_status,
        )
    }

    /// Merges multiple local result lists into a single Top-K list (pure function).
    pub fn merge_local_results(
        shard_results: Vec<Vec<SearchResultItem>>,
        k: usize,
        higher_is_better: bool,
    ) -> Vec<SearchResultItem> {
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

        for results in shard_results {
            for item in results {
                let heap_item = HeapItem {
                    item,
                    higher_is_better,
                };

                if heap.len() < k {
                    heap.push(heap_item);
                } else if let Some(worst) = heap.peek() {
                    let is_better = if higher_is_better {
                        heap_item.item.distance > worst.item.distance
                    } else {
                        heap_item.item.distance < worst.item.distance
                    };

                    if is_better {
                        heap.pop();
                        heap.push(heap_item);
                    }
                }
            }
        }

        let mut final_results: Vec<SearchResultItem> = heap.into_iter().map(|h| h.item).collect();
        if higher_is_better {
            final_results.sort_by(|a, b| {
                b.distance
                    .partial_cmp(&a.distance)
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            final_results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
        }

        final_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_local_results_distance_metric() {
        let shard0 = vec![
            SearchResultItem {
                id: 1,
                distance: 0.2,
                shard_id: 0,
            },
            SearchResultItem {
                id: 2,
                distance: 0.8,
                shard_id: 0,
            },
        ];
        let shard1 = vec![
            SearchResultItem {
                id: 3,
                distance: 0.1,
                shard_id: 1,
            },
            SearchResultItem {
                id: 4,
                distance: 0.5,
                shard_id: 1,
            },
        ];

        let merged = ScatterGatherAggregator::merge_local_results(vec![shard0, shard1], 3, false);

        assert_eq!(merged.len(), 3);
        // Best distances: 0.1 (id 3, shard 1), 0.2 (id 1, shard 0), 0.5 (id 4, shard 1)
        assert_eq!(merged[0].id, 3);
        assert_eq!(merged[0].shard_id, 1);
        assert_eq!(merged[1].id, 1);
        assert_eq!(merged[1].shard_id, 0);
        assert_eq!(merged[2].id, 4);
        assert_eq!(merged[2].shard_id, 1);
    }

    #[test]
    fn test_merge_local_results_similarity_metric() {
        let shard0 = vec![
            SearchResultItem {
                id: 1,
                distance: 0.9,
                shard_id: 0,
            },
            SearchResultItem {
                id: 2,
                distance: 0.4,
                shard_id: 0,
            },
        ];
        let shard1 = vec![
            SearchResultItem {
                id: 3,
                distance: 0.95,
                shard_id: 1,
            },
            SearchResultItem {
                id: 4,
                distance: 0.7,
                shard_id: 1,
            },
        ];

        // higher_is_better = true
        let merged = ScatterGatherAggregator::merge_local_results(vec![shard0, shard1], 2, true);

        assert_eq!(merged.len(), 2);
        // Best similarities: 0.95 (id 3, shard 1), 0.9 (id 1, shard 0)
        assert_eq!(merged[0].id, 3);
        assert_eq!(merged[1].id, 1);
    }

    #[test]
    fn test_merge_local_results_empty_and_k_zero() {
        let empty_shards: Vec<Vec<SearchResultItem>> = vec![vec![], vec![]];
        let merged_empty = ScatterGatherAggregator::merge_local_results(empty_shards, 5, false);
        assert!(merged_empty.is_empty());

        let shard0 = vec![SearchResultItem {
            id: 1,
            distance: 0.5,
            shard_id: 0,
        }];
        let merged_k_zero = ScatterGatherAggregator::merge_local_results(vec![shard0], 0, false);
        assert!(merged_k_zero.is_empty());
    }

    #[test]
    fn test_merge_local_results_many_shards_and_k_larger() {
        let mut all_shards = Vec::new();
        for shard_idx in 0..10 {
            let mut shard_items = Vec::new();
            for item_idx in 0..5 {
                shard_items.push(SearchResultItem {
                    id: item_idx as u32,
                    distance: (shard_idx * 10 + item_idx) as f32 * 0.1,
                    shard_id: shard_idx as u32,
                });
            }
            all_shards.push(shard_items);
        }

        // Request k = 100 which is > total elements (50)
        let merged = ScatterGatherAggregator::merge_local_results(all_shards, 100, false);
        assert_eq!(merged.len(), 50);

        // Verify sorted ascending
        for w in merged.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
        assert_eq!(merged[0].shard_id, 0);
        assert_eq!(merged[0].id, 0);
    }

    #[test]
    fn test_merge_local_results_ties_and_identical_distances() {
        let shard0 = vec![SearchResultItem {
            id: 1,
            distance: 0.5,
            shard_id: 0,
        }];
        let shard1 = vec![SearchResultItem {
            id: 2,
            distance: 0.5,
            shard_id: 1,
        }];
        let shard2 = vec![SearchResultItem {
            id: 3,
            distance: 0.5,
            shard_id: 2,
        }];

        let merged =
            ScatterGatherAggregator::merge_local_results(vec![shard0, shard1, shard2], 3, false);
        assert_eq!(merged.len(), 3);
        for item in &merged {
            assert!((item.distance - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_merge_local_results_negative_similarity_values() {
        let shard0 = vec![
            SearchResultItem {
                id: 1,
                distance: -0.8,
                shard_id: 0,
            },
            SearchResultItem {
                id: 2,
                distance: -0.2,
                shard_id: 0,
            },
        ];
        let shard1 = vec![
            SearchResultItem {
                id: 3,
                distance: -0.05,
                shard_id: 1,
            },
            SearchResultItem {
                id: 4,
                distance: -0.99,
                shard_id: 1,
            },
        ];

        // higher_is_better = true -> highest value (-0.05) is best
        let merged = ScatterGatherAggregator::merge_local_results(vec![shard0, shard1], 3, true);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id, 3); // -0.05
        assert_eq!(merged[1].id, 2); // -0.2
        assert_eq!(merged[2].id, 1); // -0.8
    }

    #[test]
    fn test_merge_local_results_mixed_empty_and_populated_shards() {
        let shard0 = vec![];
        let shard1 = vec![SearchResultItem {
            id: 10,
            distance: 0.3,
            shard_id: 1,
        }];
        let shard2 = vec![];
        let shard3 = vec![SearchResultItem {
            id: 20,
            distance: 0.1,
            shard_id: 3,
        }];

        let merged = ScatterGatherAggregator::merge_local_results(
            vec![shard0, shard1, shard2, shard3],
            5,
            false,
        );
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 20); // dist 0.1
        assert_eq!(merged[1].id, 10); // dist 0.3
    }

    #[tokio::test]
    async fn test_search_cluster_empty_router_and_zero_k() {
        let router = Arc::new(WorkerRouter::new());
        let aggregator = ScatterGatherAggregator::new(router);

        // Empty router
        let resp = aggregator
            .search_cluster(vec![1.0, 2.0, 3.0], 5, 50, ProtoDistanceMetric::L2Squared)
            .await;
        assert_eq!(resp.total_queried_shards, 0);
        assert_eq!(resp.successful_shards, 0);
        assert!(resp.results.is_empty());

        // Zero k
        let resp_k0 = aggregator
            .search_cluster(vec![1.0, 2.0, 3.0], 0, 50, ProtoDistanceMetric::L2Squared)
            .await;
        assert_eq!(resp_k0.results.len(), 0);

        let cloned_agg = aggregator.clone();
        let resp_cloned = cloned_agg
            .search_cluster(vec![1.0], 1, 10, ProtoDistanceMetric::DotProduct)
            .await;
        assert!(resp_cloned.results.is_empty());
    }

    #[test]
    fn test_heap_item_ord_and_equality() {
        let item1 = HeapItem {
            item: SearchResultItem {
                id: 1,
                distance: 0.2,
                shard_id: 0,
            },
            higher_is_better: false,
        };
        let item2 = HeapItem {
            item: SearchResultItem {
                id: 2,
                distance: 0.8,
                shard_id: 0,
            },
            higher_is_better: false,
        };
        let item1_dup = HeapItem {
            item: SearchResultItem {
                id: 99,
                distance: 0.2,
                shard_id: 1,
            },
            higher_is_better: false,
        };

        assert_eq!(item1, item1_dup);
        assert!(item1 < item2);
        assert!(item2 > item1);

        // Debug & Clone
        let cloned = item1.clone();
        assert_eq!(cloned, item1);
        let debug_str = format!("{:?}", item1);
        assert!(debug_str.contains("HeapItem"));

        // Similarity (higher is better)
        let sim1 = HeapItem {
            item: SearchResultItem {
                id: 1,
                distance: 0.2,
                shard_id: 0,
            },
            higher_is_better: true,
        };
        let sim2 = HeapItem {
            item: SearchResultItem {
                id: 2,
                distance: 0.8,
                shard_id: 0,
            },
            higher_is_better: true,
        };
        // For similarity, the root of BinaryHeap must be the worst (lowest), so 0.2 > 0.8 in heap order
        assert_eq!(sim1.cmp(&sim2), Ordering::Greater);
    }
}
