//! High-performance client and benchmark CLI for VectorRS Distributed Search Engine.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::Rng;
use vector_proto::{ClusterSearchRequest, DistanceMetric, VectorCoordinatorServiceClient};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "VectorRS Search Client & Cluster Benchmark Tool"
)]
struct Args {
    /// Coordinator gRPC endpoint URL
    #[arg(short, long, default_value = "http://127.0.0.1:50050")]
    coordinator: String,

    /// Top-K nearest neighbors to retrieve
    #[arg(short, long, default_value_t = 10)]
    k: u32,

    /// ef_search / nprobe search parameter
    #[arg(short, long, default_value_t = 64)]
    ef_search: u32,

    /// Distance metric: "l2", "dot", "cosine", "manhattan", "minkowski", "chebyshev", "hamming", "mahalanobis", "jaccard", "hellinger"
    #[arg(short = 'm', long, default_value = "l2")]
    metric: String,

    /// Vector dimension (if generating random queries)
    #[arg(short, long, default_value_t = 128)]
    dimension: usize,

    /// Number of queries to benchmark
    #[arg(short = 'n', long, default_value_t = 100)]
    num_queries: usize,

    /// Optional path to binary query vectors file (.bin)
    #[arg(long)]
    queries_file: Option<PathBuf>,
}

fn parse_proto_metric(s: &str) -> DistanceMetric {
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
        _ => DistanceMetric::L2Squared,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let proto_metric = parse_proto_metric(&args.metric);

    println!("==================================================");
    println!("     VectorRS Distributed Search Client          ");
    println!("==================================================");
    println!("Coordinator:  {}", args.coordinator);
    println!("Top-K:        {}", args.k);
    println!("ef_search:    {}", args.ef_search);
    println!(
        "Metric:       {:?} (code {})",
        proto_metric, proto_metric as i32
    );
    println!("Dimension:    {}", args.dimension);
    println!("Num Queries:  {}", args.num_queries);
    println!("--------------------------------------------------");

    println!("Connecting to Coordinator at {}...", args.coordinator);
    let mut client = VectorCoordinatorServiceClient::connect(args.coordinator.clone()).await?;
    println!("Connected successfully!\n");

    let mut rng = rand::thread_rng();
    let mut latencies_micros: Vec<u128> = Vec::with_capacity(args.num_queries);
    let mut successful_queries = 0;

    let overall_start = Instant::now();

    for i in 0..args.num_queries {
        let query_vector: Vec<f32> = (0..args.dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let req = tonic::Request::new(ClusterSearchRequest {
            query_vector,
            k: args.k,
            ef_search: args.ef_search,
            metric: proto_metric as i32,
        });

        let q_start = Instant::now();
        match client.search_cluster(req).await {
            Ok(resp) => {
                let dur = q_start.elapsed().as_micros();
                latencies_micros.push(dur);
                successful_queries += 1;

                if i == 0 {
                    let results = resp.into_inner().results;
                    println!("Sample Query 0 returned {} results:", results.len());
                    for (rank, item) in results.iter().enumerate().take(5) {
                        println!(
                            "  [Rank {}] Vector ID: {:<8} Score/Dist: {:.6} (Shard: {})",
                            rank + 1,
                            item.id,
                            item.distance,
                            item.shard_id
                        );
                    }
                    println!();
                }
            }
            Err(status) => {
                eprintln!("Query {} failed: {}", i, status.message());
            }
        }
    }

    let overall_duration = overall_start.elapsed();
    let qps = successful_queries as f64 / overall_duration.as_secs_f64();

    latencies_micros.sort_unstable();

    println!("==================================================");
    println!("               Benchmark Results                  ");
    println!("==================================================");
    println!(
        "Completed Queries:  {}/{}",
        successful_queries, args.num_queries
    );
    println!("Total Time Taken:   {:.2?}", overall_duration);
    println!("Throughput:         {:.2} QPS", qps);

    if !latencies_micros.is_empty() {
        let p50 = latencies_micros[latencies_micros.len() * 50 / 100];
        let p90 = latencies_micros[latencies_micros.len() * 90 / 100];
        let p95 = latencies_micros[latencies_micros.len() * 95 / 100];
        let p99 = latencies_micros[latencies_micros.len() * 99 / 100];
        let min = latencies_micros[0];
        let max = latencies_micros[latencies_micros.len() - 1];

        println!("\nLatency Distribution (Microseconds):");
        println!("  Min:  {:>8} µs", min);
        println!("  P50:  {:>8} µs", p50);
        println!("  P90:  {:>8} µs", p90);
        println!("  P95:  {:>8} µs", p95);
        println!("  P99:  {:>8} µs", p99);
        println!("  Max:  {:>8} µs", max);
    }
    println!("==================================================");

    Ok(())
}
