//! gRPC service implementation for VectorRS Coordinator Node.

use std::sync::Arc;
use tonic::{Request, Response, Status};
use vector_proto::{
    ClusterSearchRequest, ClusterSearchResponse, DistanceMetric as ProtoDistanceMetric,
    VectorCoordinatorService,
};

use crate::aggregator::ScatterGatherAggregator;

/// Implementation of the `VectorCoordinatorService` gRPC service.
pub struct CoordinatorServiceImpl {
    aggregator: Arc<ScatterGatherAggregator>,
}

impl CoordinatorServiceImpl {
    /// Creates a new `CoordinatorServiceImpl`.
    pub fn new(aggregator: Arc<ScatterGatherAggregator>) -> Self {
        Self { aggregator }
    }
}

#[tonic::async_trait]
impl VectorCoordinatorService for CoordinatorServiceImpl {
    async fn search_cluster(
        &self,
        request: Request<ClusterSearchRequest>,
    ) -> std::result::Result<Response<ClusterSearchResponse>, Status> {
        let req = request.into_inner();
        let query_dim = req.query_vector.len();

        tracing::info!(
            query_dimension = query_dim,
            k = req.k,
            ef_search = req.ef_search,
            metric = req.metric,
            "Received ClusterSearchRequest"
        );

        if req.query_vector.is_empty() {
            tracing::warn!("Rejected ClusterSearchRequest: query_vector must not be empty");
            return Err(Status::invalid_argument("query_vector must not be empty"));
        }
        if req.k == 0 {
            tracing::warn!("Rejected ClusterSearchRequest: k must be > 0");
            return Err(Status::invalid_argument("k must be > 0"));
        }

        let metric =
            ProtoDistanceMetric::try_from(req.metric).unwrap_or(ProtoDistanceMetric::Unspecified);
        let ef_search = if req.ef_search > 0 {
            req.ef_search as usize
        } else {
            50
        };

        let (response, last_error) = self
            .aggregator
            .search_cluster_with_error(req.query_vector, req.k as usize, ef_search, metric)
            .await;

        if response.total_queried_shards > 0 && response.successful_shards == 0 {
            if let Some(err) = last_error {
                tracing::warn!(
                    code = ?err.code(),
                    message = %err.message(),
                    "All worker shards rejected or failed the query"
                );
                return Err(Status::new(
                    err.code(),
                    format!("query failed across all shards: {}", err.message()),
                ));
            } else {
                return Err(Status::unavailable(
                    "no worker nodes responded successfully",
                ));
            }
        }

        tracing::info!(
            queried_shards = response.total_queried_shards,
            successful_shards = response.successful_shards,
            results_count = response.results.len(),
            latency_micros = response.query_latency_micros,
            "Dispatched and aggregated ClusterSearchResponse"
        );

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::WorkerRouter;
    use tonic::Code;

    #[tokio::test]
    async fn test_service_search_cluster_empty_query_error() {
        let router = Arc::new(WorkerRouter::new());
        let aggregator = Arc::new(ScatterGatherAggregator::new(router));
        let service = CoordinatorServiceImpl::new(aggregator);

        let req = ClusterSearchRequest {
            query_vector: vec![],
            k: 5,
            ef_search: 50,
            metric: ProtoDistanceMetric::L2Squared as i32,
        };

        let result = service.search_cluster(Request::new(req)).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("query_vector must not be empty"));
    }

    #[tokio::test]
    async fn test_service_search_cluster_zero_k_error() {
        let router = Arc::new(WorkerRouter::new());
        let aggregator = Arc::new(ScatterGatherAggregator::new(router));
        let service = CoordinatorServiceImpl::new(aggregator);

        let req = ClusterSearchRequest {
            query_vector: vec![1.0, 2.0],
            k: 0,
            ef_search: 50,
            metric: ProtoDistanceMetric::L2Squared as i32,
        };

        let result = service.search_cluster(Request::new(req)).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("k must be > 0"));
    }

    #[tokio::test]
    async fn test_service_search_cluster_default_ef_search_and_unspecified_metric() {
        let router = Arc::new(WorkerRouter::new());
        let aggregator = Arc::new(ScatterGatherAggregator::new(router));
        let service = CoordinatorServiceImpl::new(aggregator);

        let req = ClusterSearchRequest {
            query_vector: vec![1.0, 2.0, 3.0],
            k: 10,
            ef_search: 0, // Should trigger fallback to default (50)
            metric: 999,  // Invalid metric int -> Unspecified
        };

        let result = service.search_cluster(Request::new(req)).await;
        assert!(result.is_ok());
        let resp = result.unwrap().into_inner();
        assert_eq!(resp.total_queried_shards, 0);
        assert!(resp.results.is_empty());
    }
}
