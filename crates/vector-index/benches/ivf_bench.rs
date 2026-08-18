use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, black_box, criterion_group, criterion_main,
};
use rand::Rng;
use std::time::Duration;
use vector_index::DistanceMetric;
use vector_index::flat::FlatIndex;
use vector_index::ivf::{IvfPqConfig, IvfPqIndex};
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

fn get_target_nprobe() -> Vec<usize> {
    std::env::var("CRITERION_NPROBE")
        .or_else(|_| std::env::var("NPROBE"))
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16])
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

fn bench_ivf_pq_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf_pq_construction");
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
    let config = IvfPqConfig::new(32, 4, 8)
        .with_sub_clusters(64)
        .with_max_kmeans_iters(15);

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("sequential_build_1000x128d_nlist32_m8", |b| {
        b.iter(|| {
            black_box(
                IvfPqIndex::build_sequential_with_config(storage.clone(), config.clone(), metric)
                    .unwrap(),
            )
        });
    });

    group.bench_function("parallel_build_1000x128d_nlist32_m8", |b| {
        b.iter(|| {
            black_box(
                IvfPqIndex::build_parallel_with_config(storage.clone(), config.clone(), metric)
                    .unwrap(),
            )
        });
    });

    group.finish();
}

fn bench_ivf_pq_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf_pq_search_nprobe");
    let dimension = get_target_dimension();
    let n = get_target_num_vectors();
    let metric = get_target_metric();
    let nprobe_values = get_target_nprobe();

    let storage = generate_dataset(n, dimension);
    let query: Vec<f32> = (0..dimension).map(|i| (i as f32) * 0.01).collect();

    let flat = FlatIndex::new(storage.clone(), metric);
    let config = IvfPqConfig::new(32, 4, 8)
        .with_sub_clusters(64)
        .with_max_kmeans_iters(15);

    let ivf = IvfPqIndex::build_with_config(storage, config, metric).unwrap();
    let k = 10;

    // Verify recall against ground truth
    let gt = flat.search(&query, k).unwrap();
    for nprobe in nprobe_values {
        let res = ivf.search_with_nprobe(&query, k, nprobe).unwrap();
        let recall = IvfPqIndex::<HeapStorage>::evaluate_recall(&gt, &res);
        println!(
            "IVF-PQ (nprobe={}) Pre-check Recall@10: {:.2}%",
            nprobe,
            recall * 100.0
        );

        group.bench_function(BenchmarkId::new("nprobe", nprobe), |b| {
            b.iter(|| {
                black_box(
                    ivf.search_with_nprobe(black_box(&query), black_box(k), black_box(nprobe))
                        .unwrap(),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ivf_pq_construction, bench_ivf_pq_search);
criterion_main!(benches);
