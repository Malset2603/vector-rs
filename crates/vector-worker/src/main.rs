//! Executable entrypoint for VectorRS Worker Node.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use vector_index::DistanceMetric;
use vector_index::hnsw::HnswConfig;
use vector_index::ivf::IvfPqConfig;
use vector_index::storage::HeapStorage;
use vector_worker::{WorkerEngine, run_worker_server};

#[derive(Parser, Debug)]
#[command(author, version, about = "VectorRS Distributed Worker Node")]
struct Args {
    /// Host address to bind to
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Unique Worker identifier
    #[arg(short, long, default_value = "worker-0")]
    worker_id: String,

    /// Shard identifier assigned to this worker
    #[arg(short, long, default_value_t = 0)]
    shard_id: u32,

    /// Index algorithm type: "hnsw" or "ivf-pq"
    #[arg(long, default_value = "hnsw")]
    index_type: String,

    /// Distance metric: "l2", "dot", "cosine", "manhattan", "minkowski", "chebyshev", "hamming", "mahalanobis", "jaccard", "hellinger"
    #[arg(short = 'm', long, default_value = "l2")]
    metric: String,

    /// Path to memory-mapped vector storage file
    #[arg(long)]
    storage_file: Option<PathBuf>,

    /// Path to pre-built HNSW graph topology file
    #[arg(long)]
    graph_file: Option<PathBuf>,

    /// Path to pre-built IVF-PQ index file
    #[arg(long)]
    ivf_file: Option<PathBuf>,

    /// Number of coarse clusters for IVF-PQ
    #[arg(long, default_value_t = 64)]
    nlist: usize,

    /// Number of probed clusters for IVF-PQ
    #[arg(long, default_value_t = 8)]
    nprobe: usize,

    /// Number of sub-vectors for IVF-PQ
    #[arg(long, default_value_t = 8)]
    num_subvectors: usize,

    /// Vector dimension (if generating sample in-memory data)
    #[arg(short, long, default_value_t = 128)]
    dimension: usize,

    /// Enable CUDA GPU acceleration for worker operations
    #[arg(long)]
    use_cuda: bool,

    /// Number of GPU devices allocated for DDP multi-GPU processing
    #[arg(short = 'g', long = "num-gpus", alias = "gpus")]
    num_gpus: Option<usize>,
}

fn parse_metric_str(s: &str) -> DistanceMetric {
    match s.to_lowercase().as_str() {
        "l2" | "l2_squared" | "l2squared" | "euclidean" => DistanceMetric::L2Squared,
        "dot" | "dot_product" | "dotproduct" | "inner_product" => DistanceMetric::DotProduct,
        "cosine" | "cosine_similarity" | "cos" => DistanceMetric::CosineSimilarity,
        "manhattan" | "l1" => DistanceMetric::Manhattan,
        "minkowski" | "lp" => DistanceMetric::Minkowski,
        "chebyshev" | "linf" | "l_inf" => DistanceMetric::Chebyshev,
        "hamming" | "l0" => DistanceMetric::Hamming,
        "mahalanobis" => DistanceMetric::Mahalanobis,
        "jaccard" | "tanimoto" => DistanceMetric::Jaccard,
        "hellinger" => DistanceMetric::Hellinger,
        _ => {
            tracing::warn!("Unknown metric '{}', defaulting to L2Squared", s);
            DistanceMetric::L2Squared
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,vector_worker=debug")),
        )
        .init();

    let args = Args::parse();
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let metric = parse_metric_str(&args.metric);

    // Validate CUDA GPU hardware strictly if requested
    if args.use_cuda || args.num_gpus.is_some() {
        let requested_gpus = args.num_gpus.unwrap_or(1);
        let available_gpus = vector_cuda::CudaDeviceContext::device_count();

        info!(
            requested_gpus,
            available_gpus, "Validating CUDA GPU hardware configuration"
        );

        if available_gpus == 0 {
            error!(
                requested = requested_gpus,
                "CUDA GPU hardware acceleration requested, but no physical CUDA devices were found on this system"
            );
            eprintln!(
                "Error: CUDA GPU acceleration was requested (--use-cuda / --num-gpus {}), but 0 physical CUDA GPUs were found on this system.",
                requested_gpus
            );
            std::process::exit(1);
        }

        if requested_gpus > available_gpus {
            error!(
                requested = requested_gpus,
                available = available_gpus,
                "Requested GPU count exceeds available physical GPU hardware"
            );
            eprintln!(
                "Error: Requested {} GPUs via --num-gpus, but only {} physical CUDA GPU(s) available on this machine.",
                requested_gpus, available_gpus
            );
            std::process::exit(1);
        }

        info!(
            gpus = requested_gpus,
            "CUDA hardware validated successfully: {} physical GPU(s) active", requested_gpus
        );
    }

    info!(
        worker_id = %args.worker_id,
        bind_address = %addr,
        shard_id = args.shard_id,
        index_type = %args.index_type,
        metric = ?metric,
        use_cuda = args.use_cuda || args.num_gpus.is_some(),
        num_gpus = ?args.num_gpus,
        "Starting VectorRS Worker Node"
    );

    let is_ivf = args.index_type.eq_ignore_ascii_case("ivf-pq")
        || args.index_type.eq_ignore_ascii_case("ivf");

    let engine = if is_ivf {
        let ivf_config = IvfPqConfig::new(args.nlist, args.nprobe, args.num_subvectors);

        if let Some(storage_path) = args.storage_file {
            info!(
                storage_path = ?storage_path,
                shard_id = args.shard_id,
                nlist = args.nlist,
                nprobe = args.nprobe,
                metric = ?metric,
                "Loading memory-mapped IVF-PQ storage and building index"
            );
            WorkerEngine::from_mmap_ivf_pq(
                args.shard_id,
                storage_path,
                args.ivf_file,
                ivf_config,
                metric,
            )?
        } else {
            info!(
                dimension = args.dimension,
                shard_id = args.shard_id,
                metric = ?metric,
                "No storage file provided; initializing empty in-memory IVF-PQ shard"
            );
            let storage = HeapStorage::new(args.dimension);
            WorkerEngine::from_heap_ivf_pq(args.shard_id, storage, ivf_config, metric)?
        }
    } else {
        let hnsw_config = HnswConfig::default();

        if let Some(storage_path) = args.storage_file {
            info!(
                storage_path = ?storage_path,
                shard_id = args.shard_id,
                metric = ?metric,
                "Loading memory-mapped HNSW storage and building index topology"
            );
            WorkerEngine::from_mmap(
                args.shard_id,
                storage_path,
                args.graph_file,
                hnsw_config,
                metric,
            )?
        } else {
            info!(
                dimension = args.dimension,
                shard_id = args.shard_id,
                metric = ?metric,
                "No storage file provided; initializing empty in-memory HNSW shard"
            );
            let storage = HeapStorage::new(args.dimension);
            WorkerEngine::from_heap(args.shard_id, storage, hnsw_config, metric)
        }
    };

    info!(
        worker_id = %args.worker_id,
        shard_id = engine.shard_id(),
        index_type = %engine.index_type(),
        num_vectors = engine.num_vectors(),
        dimension = engine.dimension(),
        bind_address = %addr,
        "VectorRS Worker is ONLINE and listening for incoming gRPC queries"
    );

    if let Err(e) = run_worker_server(addr, args.worker_id.clone(), Arc::new(engine)).await {
        error!(
            worker_id = %args.worker_id,
            error = %e,
            "Worker server encountered a fatal error"
        );
    }

    Ok(())
}
