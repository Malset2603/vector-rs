//! # vector-coordinator
//!
//! Scatter-gather query router and Top-K aggregator for the VectorRS distributed vector search engine.

pub mod aggregator;
pub mod router;
pub mod service;

pub use aggregator::ScatterGatherAggregator;
pub use router::{WorkerNodeInfo, WorkerRouter};
pub use service::CoordinatorServiceImpl;

use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use vector_proto::VectorCoordinatorServiceServer;

/// Starts a VectorRS Coordinator gRPC server on the given socket address.
pub async fn run_coordinator_server(
    addr: SocketAddr,
    aggregator: Arc<ScatterGatherAggregator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = CoordinatorServiceImpl::new(aggregator);

    Server::builder()
        .add_service(VectorCoordinatorServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
