//! gRPC service implementation for VectorRS Worker Node.

use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use vector_proto::{
    PingRequest, PingResponse, SearchRequest, SearchResponse, StatsRequest, StatsResponse,
    VectorWorkerService,
};

use crate::engine::WorkerEngine;

/// Implementation of the `VectorWorkerService` gRPC service.
pub struct WorkerServiceImpl {
    worker_id: String,
    engine: Arc<WorkerEngine>,
}

impl WorkerServiceImpl {
    /// Creates a new `WorkerServiceImpl` wrapping a shared `WorkerEngine`.
    pub fn new(worker_id: String, engine: Arc<WorkerEngine>) -> Self {
        Self { worker_id, engine }
    }
}

#[tonic::async_trait]
impl VectorWorkerService for WorkerServiceImpl {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> std::result::Result<Response<SearchResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let query_dim = req.query_vector.len();

        let k = req.k as usize;
        let search_param = req.ef_search as usize;

        tracing::debug!(
            worker_id = %self.worker_id,
            shard_id = self.engine.shard_id(),
            query_dimension = query_dim,
            k,
            ef_search = search_param,
            "Received SearchRequest"
        );

        if req.query_vector.is_empty() {
            tracing::warn!(
                worker_id = %self.worker_id,
                shard_id = self.engine.shard_id(),
                "Rejected search: query_vector must not be empty"
            );
            return Err(Status::invalid_argument("query_vector must not be empty"));
        }

        if query_dim != self.engine.dimension() {
            tracing::warn!(
                worker_id = %self.worker_id,
                shard_id = self.engine.shard_id(),
                expected_dimension = self.engine.dimension(),
                received_dimension = query_dim,
                "Rejected search: vector dimension mismatch"
            );
            return Err(Status::invalid_argument(format!(
                "dimension mismatch: expected {}, got {}",
                self.engine.dimension(),
                req.query_vector.len()
            )));
        }

        let requested_proto_metric = vector_proto::DistanceMetric::try_from(req.metric)
            .unwrap_or(vector_proto::DistanceMetric::Unspecified);

        if requested_proto_metric != vector_proto::DistanceMetric::Unspecified {
            let requested_metric = match requested_proto_metric {
                vector_proto::DistanceMetric::L2Squared => vector_index::DistanceMetric::L2Squared,
                vector_proto::DistanceMetric::DotProduct => {
                    vector_index::DistanceMetric::DotProduct
                }
                vector_proto::DistanceMetric::CosineSimilarity => {
                    vector_index::DistanceMetric::CosineSimilarity
                }
                vector_proto::DistanceMetric::Manhattan => vector_index::DistanceMetric::Manhattan,
                vector_proto::DistanceMetric::Minkowski => vector_index::DistanceMetric::Minkowski,
                vector_proto::DistanceMetric::Chebyshev => vector_index::DistanceMetric::Chebyshev,
                vector_proto::DistanceMetric::Hamming => vector_index::DistanceMetric::Hamming,
                vector_proto::DistanceMetric::Mahalanobis => {
                    vector_index::DistanceMetric::Mahalanobis
                }
                vector_proto::DistanceMetric::Jaccard => vector_index::DistanceMetric::Jaccard,
                vector_proto::DistanceMetric::Hellinger => vector_index::DistanceMetric::Hellinger,
                vector_proto::DistanceMetric::Unspecified => unreachable!(),
            };

            if requested_metric != self.engine.metric() {
                tracing::warn!(
                    worker_id = %self.worker_id,
                    shard_id = self.engine.shard_id(),
                    shard_metric = ?self.engine.metric(),
                    requested_metric = ?requested_metric,
                    "Rejected search: distance metric mismatch"
                );
                return Err(Status::invalid_argument(format!(
                    "metric mismatch: shard index was configured with {:?}, but query requested {:?}",
                    self.engine.metric(),
                    requested_metric
                )));
            }
        }

        let results = self
            .engine
            .search(&req.query_vector, k, search_param)
            .map_err(|e| {
                tracing::error!(
                    worker_id = %self.worker_id,
                    shard_id = self.engine.shard_id(),
                    error = %e,
                    "ANN search execution failed"
                );
                Status::internal(format!("search failed: {e}"))
            })?;

        let elapsed = start.elapsed().as_micros() as u64;

        tracing::info!(
            worker_id = %self.worker_id,
            shard_id = self.engine.shard_id(),
            matches_found = results.len(),
            latency_micros = elapsed,
            "SearchRequest processed successfully"
        );

        Ok(Response::new(SearchResponse {
            results,
            shard_id: self.engine.shard_id(),
            query_latency_micros: elapsed,
        }))
    }

    async fn ping(
        &self,
        request: Request<PingRequest>,
    ) -> std::result::Result<Response<PingResponse>, Status> {
        let client_id = &request.get_ref().client_id;
        tracing::debug!(
            worker_id = %self.worker_id,
            shard_id = self.engine.shard_id(),
            client_id = %client_id,
            "Handled health check ping"
        );

        Ok(Response::new(PingResponse {
            worker_id: self.worker_id.clone(),
            shard_id: self.engine.shard_id(),
            num_vectors: self.engine.num_vectors() as u64,
            dimension: self.engine.dimension() as u32,
            ready: true,
        }))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> std::result::Result<Response<StatsResponse>, Status> {
        tracing::debug!(
            worker_id = %self.worker_id,
            shard_id = self.engine.shard_id(),
            num_vectors = self.engine.num_vectors(),
            dimension = self.engine.dimension(),
            "Handled stats request"
        );

        Ok(Response::new(StatsResponse {
            shard_id: self.engine.shard_id(),
            num_vectors: self.engine.num_vectors() as u64,
            dimension: self.engine.dimension() as u32,
            index_type: self.engine.index_type().to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vector_index::DistanceMetric;
    use vector_index::hnsw::HnswConfig;
    use vector_index::storage::HeapStorage;

    #[tokio::test]
    async fn test_worker_service_ping_and_search() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0]).unwrap();

        let engine = Arc::new(WorkerEngine::from_heap(
            10,
            storage,
            HnswConfig::default(),
            DistanceMetric::L2Squared,
        ));

        let service = WorkerServiceImpl::new("worker-1".to_string(), engine);

        // Test Ping
        let ping_res = service
            .ping(Request::new(PingRequest {
                client_id: "test".to_string(),
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(ping_res.worker_id, "worker-1");
        assert_eq!(ping_res.shard_id, 10);
        assert_eq!(ping_res.num_vectors, 2);
        assert_eq!(ping_res.dimension, 2);
        assert!(ping_res.ready);

        // Test Search
        let search_res = service
            .search(Request::new(SearchRequest {
                query_vector: vec![1.0, 0.0],
                k: 1,
                ef_search: 50,
                metric: 1, // L2Squared
                shard_id: 10,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(search_res.results.len(), 1);
        assert_eq!(search_res.results[0].id, 0);
        assert_eq!(search_res.shard_id, 10);

        // Test Stats
        let stats_res = service
            .get_stats(Request::new(StatsRequest { shard_id: 10 }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(stats_res.shard_id, 10);
        assert_eq!(stats_res.num_vectors, 2);
        assert_eq!(stats_res.dimension, 2);
        assert_eq!(stats_res.index_type, "HNSW");

        // Test Search Error: Empty Query
        let empty_err = service
            .search(Request::new(SearchRequest {
                query_vector: vec![],
                k: 1,
                ef_search: 50,
                metric: 1,
                shard_id: 10,
            }))
            .await;
        assert!(empty_err.is_err());

        // Test Search Error: Dimension Mismatch
        let dim_err = service
            .search(Request::new(SearchRequest {
                query_vector: vec![1.0, 2.0, 3.0],
                k: 1,
                ef_search: 50,
                metric: 1,
                shard_id: 10,
            }))
            .await;
        assert!(dim_err.is_err());
    }

    #[tokio::test]
    async fn test_worker_service_ivf_pq_backend() {
        use vector_index::ivf::IvfPqConfig;

        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0]).unwrap();

        let config = IvfPqConfig::new(1, 1, 1).with_sub_clusters(2);
        let engine = Arc::new(
            WorkerEngine::from_heap_ivf_pq(20, storage, config, DistanceMetric::L2Squared).unwrap(),
        );

        let service = WorkerServiceImpl::new("worker-ivf".to_string(), engine);

        // Ping
        let ping_res = service
            .ping(Request::new(PingRequest {
                client_id: "test".to_string(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(ping_res.worker_id, "worker-ivf");
        assert_eq!(ping_res.shard_id, 20);

        // Stats
        let stats_res = service
            .get_stats(Request::new(StatsRequest { shard_id: 20 }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(stats_res.shard_id, 20);
        assert_eq!(stats_res.index_type, "IVF-PQ");

        // Search with ef_search = 0 (default fallback)
        let search_res = service
            .search(Request::new(SearchRequest {
                query_vector: vec![1.0, 0.0],
                k: 1,
                ef_search: 0,
                metric: 1,
                shard_id: 20,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(search_res.results.len(), 1);
        assert_eq!(search_res.shard_id, 20);

        // Test Search Error: Metric Mismatch (e.g. asking for CosineSimilarity on an L2Squared shard)
        let metric_err = service
            .search(Request::new(SearchRequest {
                query_vector: vec![1.0, 0.0],
                k: 1,
                ef_search: 0,
                metric: vector_proto::DistanceMetric::CosineSimilarity as i32,
                shard_id: 20,
            }))
            .await;
        assert!(metric_err.is_err());
        let status = metric_err.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("metric mismatch"));
    }
}
