//! # vector-worker
//!
//! High-performance distributed vector search worker node server.

pub mod engine;
pub mod service;

pub use engine::{StorageBackend, WorkerEngine};
pub use service::WorkerServiceImpl;

use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use vector_proto::VectorWorkerServiceServer;

/// Starts a VectorRS worker gRPC server on the given socket address.
pub async fn run_worker_server(
    addr: SocketAddr,
    worker_id: String,
    engine: Arc<WorkerEngine>,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = WorkerServiceImpl::new(worker_id, engine);

    Server::builder()
        .add_service(VectorWorkerServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
