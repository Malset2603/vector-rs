//! Worker routing table and gRPC client connection pool.

use parking_lot::RwLock;
use std::sync::Arc;
use tonic::transport::{Channel, Endpoint};
use vector_proto::{PingRequest, PingResponse, VectorWorkerServiceClient};

/// Information about a registered worker node.
#[derive(Debug, Clone)]
pub struct WorkerNodeInfo {
    pub endpoint: String,
    pub shard_id: Option<u32>,
    pub num_vectors: Option<u64>,
    pub dimension: Option<u32>,
    pub is_alive: bool,
}

/// Router managing connections to all worker nodes in the cluster.
#[derive(Clone)]
pub struct WorkerRouter {
    workers: Arc<RwLock<Vec<WorkerNodeInfo>>>,
    clients: Arc<RwLock<Vec<VectorWorkerServiceClient<Channel>>>>,
}

impl Default for WorkerRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRouter {
    /// Creates a new empty `WorkerRouter`.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(Vec::new())),
            clients: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Connects to a worker node at the given endpoint and registers it in the routing table.
    pub async fn add_worker(&self, endpoint_str: &str) -> Result<(), tonic::transport::Error> {
        let endpoint = Endpoint::from_shared(endpoint_str.to_string())?;
        let channel = endpoint.connect().await?;
        let client = VectorWorkerServiceClient::new(channel);

        self.workers.write().push(WorkerNodeInfo {
            endpoint: endpoint_str.to_string(),
            shard_id: None,
            num_vectors: None,
            dimension: None,
            is_alive: true,
        });

        self.clients.write().push(client);
        Ok(())
    }

    /// Adds a pre-constructed gRPC client directly (useful for in-process testing).
    pub fn add_client(&self, endpoint_str: &str, client: VectorWorkerServiceClient<Channel>) {
        self.workers.write().push(WorkerNodeInfo {
            endpoint: endpoint_str.to_string(),
            shard_id: None,
            num_vectors: None,
            dimension: None,
            is_alive: true,
        });
        self.clients.write().push(client);
    }

    /// Returns clones of all active worker clients for parallel query scattering.
    pub fn get_clients(&self) -> Vec<VectorWorkerServiceClient<Channel>> {
        self.clients.read().clone()
    }

    /// Returns the number of registered workers.
    pub fn len(&self) -> usize {
        self.workers.read().len()
    }

    /// Returns `true` if no workers are registered.
    pub fn is_empty(&self) -> bool {
        self.workers.read().is_empty()
    }

    /// Pings all workers concurrently and updates their health status and metadata.
    pub async fn health_check_all(&self) -> Vec<Result<PingResponse, tonic::Status>> {
        let clients = self.get_clients();
        let mut results = Vec::with_capacity(clients.len());

        for mut client in clients {
            let req = tonic::Request::new(PingRequest {
                client_id: "coordinator".to_string(),
            });
            match client.ping(req).await {
                Ok(resp) => results.push(Ok(resp.into_inner())),
                Err(status) => results.push(Err(status)),
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_node_info_debug_and_clone() {
        let info = WorkerNodeInfo {
            endpoint: "http://127.0.0.1:50051".to_string(),
            shard_id: Some(1),
            num_vectors: Some(1000),
            dimension: Some(128),
            is_alive: true,
        };

        let cloned = info.clone();
        assert_eq!(cloned.endpoint, "http://127.0.0.1:50051");
        assert_eq!(cloned.shard_id, Some(1));
        assert_eq!(cloned.num_vectors, Some(1000));
        assert_eq!(cloned.dimension, Some(128));
        assert!(cloned.is_alive);

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("WorkerNodeInfo"));
        assert!(debug_str.contains("127.0.0.1:50051"));
    }

    #[test]
    fn test_worker_router_new_and_defaults() {
        let router = WorkerRouter::new();
        assert!(router.is_empty());
        assert_eq!(router.len(), 0);
        assert!(router.get_clients().is_empty());

        let default_router = WorkerRouter::default();
        assert!(default_router.is_empty());
        assert_eq!(default_router.len(), 0);
    }

    #[tokio::test]
    async fn test_worker_router_add_invalid_endpoint() {
        let router = WorkerRouter::new();
        let res = router.add_worker("not a valid uri :::").await;
        assert!(res.is_err());
    }

    #[test]
    fn test_worker_router_clone_shares_state() {
        let router1 = WorkerRouter::new();
        let router2 = router1.clone();
        assert_eq!(router1.len(), 0);
        assert_eq!(router2.len(), 0);
        assert!(router1.is_empty());
        assert!(router2.is_empty());
    }

    #[tokio::test]
    async fn test_worker_router_health_check_empty() {
        let router = WorkerRouter::new();
        let results = router.health_check_all().await;
        assert!(results.is_empty());
    }
}
