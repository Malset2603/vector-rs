use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::Rng;
use vector_cuda::{
    CudaKMeansEngine, CudaKnnEngine, DistributedKMeansEngine, DistributedKnnEngine, GpuShardMode,
};
use vector_index::DistanceMetric;

fn get_target_metric() -> DistanceMetric {
    let metric_str = std::env::var("CRITERION_METRIC")
        .or_else(|_| std::env::var("METRIC"))
        .unwrap_or_else(|_| "cosine".to_string())
        .to_lowercase();

    match metric_str.as_str() {
        "cosine" | "cos" | "cosinesimilarity" | "cosine_similarity" => {
            DistanceMetric::CosineSimilarity
        }
        "dot" | "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
        "l2" | "l2squared" | "l2_squared" | "euclidean" => DistanceMetric::L2Squared,
        "manhattan" | "l1" => DistanceMetric::Manhattan,
        _ => DistanceMetric::CosineSimilarity,
    }
}

fn get_target_dimension() -> usize {
    std::env::var("CRITERION_DIMENSION")
        .or_else(|_| std::env::var("DIMENSION"))
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(768)
}

fn get_target_num_vectors() -> usize {
    std::env::var("CRITERION_NUM_VECTORS")
        .or_else(|_| std::env::var("NUM_VECTORS"))
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(10_000)
}

fn get_target_batch_sizes() -> Vec<usize> {
    let batch_str = std::env::var("CRITERION_BATCH_SIZES")
        .or_else(|_| std::env::var("BATCH_SIZES"))
        .unwrap_or_else(|_| "1,2,4,8,16,32,64,128".to_string());

    let parsed: Vec<usize> = batch_str
        .split([',', ' ', ';'])
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    if parsed.is_empty() {
        vec![1, 2, 4, 8, 16, 32, 64, 128]
    } else {
        parsed
    }
}

fn get_target_clusters() -> Vec<usize> {
    let clusters_str = std::env::var("CRITERION_CLUSTERS")
        .or_else(|_| std::env::var("CRITERION_K_VALUES"))
        .or_else(|_| std::env::var("CLUSTERS"))
        .unwrap_or_else(|_| "16,32,64,128,256".to_string());

    let parsed: Vec<usize> = clusters_str
        .split([',', ' ', ';'])
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    if parsed.is_empty() {
        vec![16, 32, 64, 128, 256]
    } else {
        parsed
    }
}

fn get_target_num_gpus() -> usize {
    std::env::var("CRITERION_NUM_GPUS")
        .or_else(|_| std::env::var("NUM_GPUS"))
        .or_else(|_| std::env::var("GPUS"))
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

fn get_target_max_iterations() -> usize {
    std::env::var("CRITERION_MAX_ITERS")
        .or_else(|_| std::env::var("MAX_ITERS"))
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(15)
}

fn generate_flat_vectors(num_vectors: usize, dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..(num_vectors * dimension))
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

fn bench_cuda_knn_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_knn_batch_search");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let metric = get_target_metric();
    let batch_sizes = get_target_batch_sizes();

    if let Ok(sample_size_str) = std::env::var("CRITERION_SAMPLE_SIZE") {
        if let Ok(sample_size) = sample_size_str.parse::<usize>() {
            group.sample_size(sample_size);
        }
    }

    let dataset = generate_flat_vectors(n, dimension);
    let engine = CudaKnnEngine::new(&dataset, dimension, metric);
    let k = 10;

    for batch_size in batch_sizes {
        let queries = generate_flat_vectors(batch_size, dimension);
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_function(BenchmarkId::new("batch_size", batch_size), |b| {
            b.iter(|| black_box(engine.search_batch(black_box(&queries), black_box(k))));
        });
    }

    group.finish();
}

fn bench_cuda_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_kmeans_clustering");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let metric = get_target_metric();
    let clusters = get_target_clusters();
    let num_gpus = get_target_num_gpus();
    let max_iters = get_target_max_iterations();

    if let Ok(sample_size_str) = std::env::var("CRITERION_SAMPLE_SIZE") {
        if let Ok(sample_size) = sample_size_str.parse::<usize>() {
            group.sample_size(sample_size);
        }
    }

    let dataset = generate_flat_vectors(n, dimension);
    group.throughput(Throughput::Elements(n as u64));

    let engine = CudaKMeansEngine::new();

    // 1. CPU Rayon Multi-Core Baseline
    for &k in &clusters {
        group.bench_function(BenchmarkId::new("cpu", k), |b| {
            b.iter(|| {
                black_box(engine.fit_cpu(
                    black_box(&dataset),
                    black_box(dimension),
                    black_box(k),
                    black_box(max_iters),
                    black_box(1e-4),
                    metric,
                ))
            });
        });
    }

    // 2. 1x NVIDIA GPU (CUDA Single GPU)
    for &k in &clusters {
        group.bench_function(BenchmarkId::new("gpu_1", k), |b| {
            b.iter(|| {
                black_box(engine.fit(
                    black_box(&dataset),
                    black_box(dimension),
                    black_box(k),
                    black_box(max_iters),
                    black_box(1e-4),
                    metric,
                ))
            });
        });
    }

    // 3. 2x NVIDIA GPUs (Distributed DDP)
    if num_gpus >= 2 {
        let ddp_kmeans_2 = DistributedKMeansEngine::try_new(2)
            .unwrap_or_else(|_| DistributedKMeansEngine::emulator(2));
        for &k in &clusters {
            group.bench_function(BenchmarkId::new("gpu_2", k), |b| {
                b.iter(|| {
                    black_box(ddp_kmeans_2.fit(
                        black_box(&dataset),
                        black_box(dimension),
                        black_box(k),
                        black_box(max_iters),
                        black_box(1e-4),
                        metric,
                    ))
                });
            });
        }
    }

    group.finish();
}

fn bench_ddp_multi_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("ddp_multi_gpu_scaling");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let metric = get_target_metric();
    let k = 32;

    let dataset = generate_flat_vectors(n, dimension);
    let queries = generate_flat_vectors(32, dimension);

    for gpus in [1, 2, 4, 8] {
        let ddp_kmeans = DistributedKMeansEngine::try_new(gpus)
            .unwrap_or_else(|_| DistributedKMeansEngine::emulator(gpus));
        let ddp_knn_sharded =
            DistributedKnnEngine::try_new(&dataset, dimension, gpus, GpuShardMode::Sharded, metric)
                .unwrap_or_else(|_| {
                    DistributedKnnEngine::emulator(
                        &dataset,
                        dimension,
                        gpus,
                        GpuShardMode::Sharded,
                        metric,
                    )
                });

        group.bench_function(BenchmarkId::new("ddp_kmeans_gpus", gpus), |b| {
            b.iter(|| {
                black_box(ddp_kmeans.fit(
                    black_box(&dataset),
                    black_box(dimension),
                    black_box(k),
                    black_box(10),
                    black_box(1e-4),
                    metric,
                ))
            });
        });

        group.bench_function(BenchmarkId::new("ddp_knn_sharded_gpus", gpus), |b| {
            b.iter(|| black_box(ddp_knn_sharded.search_batch(black_box(&queries), black_box(10))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cuda_knn_batch,
    bench_cuda_kmeans,
    bench_ddp_multi_gpu
);
criterion_main!(benches);
