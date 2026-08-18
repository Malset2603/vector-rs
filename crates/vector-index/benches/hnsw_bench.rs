use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, black_box, criterion_group, criterion_main,
};
use rand::Rng;
use std::time::Duration;
use vector_index::DistanceMetric;
use vector_index::flat::FlatIndex;
use vector_index::hnsw::{HnswConfig, HnswIndex};
use vector_index::storage::HeapStorage;

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
        .unwrap_or(128)
}

fn get_target_num_vectors() -> usize {
    std::env::var("CRITERION_NUM_VECTORS")
        .or_else(|_| std::env::var("NUM_VECTORS"))
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1000)
}

fn get_target_sample_size() -> usize {
    std::env::var("CRITERION_SAMPLE_SIZE")
        .or_else(|_| std::env::var("SAMPLE_SIZE"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10)
        .max(10)
}

fn get_target_ef_search() -> Vec<usize> {
    std::env::var("CRITERION_EF_SEARCH")
        .or_else(|_| std::env::var("EF_SEARCH"))
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![10, 20, 50, 100, 150])
}

fn generate_dataset(num_vectors: usize, dimension: usize) -> HeapStorage {
    let mut storage = HeapStorage::with_capacity(dimension, num_vectors);
    let mut rng = rand::thread_rng();

    for _ in 0..num_vectors {
        let vec: Vec<f32> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
        storage.push(&vec).unwrap();
    }
    storage
}

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let sample_size = get_target_sample_size();
    let metric = get_target_metric();

    // Heavyweight construction uses Flat sampling to execute exactly 1 iteration per sample
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(sample_size);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(5000));

    let storage = generate_dataset(n, dimension);
    let config = HnswConfig::new(16, 100, 50);

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("sequential_build_1000x128d", |b| {
        b.iter(|| {
            black_box(HnswIndex::build_with_config(
                storage.clone(),
                config.clone(),
                metric,
            ))
        });
    });

    group.bench_function("parallel_build_1000x128d", |b| {
        b.iter(|| {
            black_box(HnswIndex::build_parallel_with_config(
                storage.clone(),
                config.clone(),
                metric,
            ))
        });
    });

    group.finish();
}

fn bench_search_qps(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_latency_and_qps");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let metric = get_target_metric();
    let ef_values = get_target_ef_search();

    let storage = generate_dataset(n, dimension);
    let query: Vec<f32> = (0..dimension).map(|i| (i as f32) * 0.01).collect();

    let flat = FlatIndex::new(storage.clone(), metric);
    let config = HnswConfig::new(16, 100, 50);
    let hnsw = HnswIndex::build_parallel_with_config(storage, config, metric);

    let k = 10;

    // Verify recall before benchmarking
    let gt = flat.search(&query, k).unwrap();
    let res = hnsw.search_default(&query, k).unwrap();
    let recall = HnswIndex::<HeapStorage>::evaluate_recall(&gt, &res);
    println!("Benchmark pre-check Recall@10: {:.2}%", recall * 100.0);

    group.bench_function(BenchmarkId::new("flat_brute_force", n), |b| {
        b.iter(|| black_box(flat.search(black_box(&query), black_box(k)).unwrap()));
    });

    for ef in ef_values {
        let res_ef = hnsw.search(&query, k, ef).unwrap();
        let recall_ef = HnswIndex::<HeapStorage>::evaluate_recall(&gt, &res_ef);
        println!(
            "HNSW (ef={}) Pre-check Recall@10: {:.2}%",
            ef,
            recall_ef * 100.0
        );

        group.bench_function(BenchmarkId::new("hnsw_search_ef", ef), |b| {
            b.iter(|| {
                black_box(
                    hnsw.search(black_box(&query), black_box(k), black_box(ef))
                        .unwrap(),
                )
            });
        });
    }

    group.bench_function(BenchmarkId::new("hnsw_search_ef50", n), |b| {
        b.iter(|| {
            black_box(
                hnsw.search_default(black_box(&query), black_box(k))
                    .unwrap(),
            )
        });
    });

    group.finish();
}

criterion_group!(benches, bench_hnsw_construction, bench_search_qps);
criterion_main!(benches);
