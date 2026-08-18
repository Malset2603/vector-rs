//! Executable entrypoint for VectorRS Coordinator Node.

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use vector_coordinator::{ScatterGatherAggregator, WorkerRouter, run_coordinator_server};

#[derive(Parser, Debug)]
#[command(author, version, about = "VectorRS Distributed Coordinator Node")]
struct Args {
    /// Host address to bind to
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 50050)]
    port: u16,

    /// Comma-separated list of worker endpoints (e.g. http://127.0.0.1:50051,http://127.0.0.1:50052)
    #[arg(short, long, value_delimiter = ',')]
    workers: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,vector_coordinator=debug")),
        )
        .init();

    let args = Args::parse();
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    info!(
        bind_address = %addr,
        configured_workers = args.workers.len(),
        "Starting VectorRS Coordinator Node"
    );

    let router = Arc::new(WorkerRouter::new());

    for worker_ep in &args.workers {
        info!(worker_endpoint = %worker_ep, "Connecting to worker endpoint...");
        if let Err(e) = router.add_worker(worker_ep).await {
            warn!(
                worker_endpoint = %worker_ep,
                error = %e,
                "Failed to connect to worker endpoint"
            );
        } else {
            info!(worker_endpoint = %worker_ep, "Worker endpoint connected successfully");
        }
    }

    info!(
        active_workers = router.len(),
        "Worker router initialized successfully"
    );

    let aggregator = Arc::new(ScatterGatherAggregator::new(router));
    info!(
        bind_address = %addr,
        "VectorRS Coordinator server is ONLINE and listening for incoming gRPC queries"
    );

    if let Err(e) = run_coordinator_server(addr, aggregator).await {
        error!(error = %e, "Coordinator server crashed");
        return Err(e);
    }

    Ok(())
}
