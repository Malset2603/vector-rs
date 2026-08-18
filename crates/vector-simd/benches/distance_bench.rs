//! Criterion benchmarks comparing Scalar vs SIMD distance computation performance.
//!
//! Run with: `cargo bench -p vector-simd`
//!
//! Measures throughput for all 10 distance/similarity metrics across
//! standard embedding dimensions (128-D SIFT, 768-D BERT, 1536-D OpenAI).

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;
use vector_simd::{DistanceEngine, SimdBackend};

/// Returns the dimensions to evaluate. Defaults to `[128, 768, 1536]`,
/// or a single dimension if `CRITERION_DIMENSION` / `DIMENSION` is set.
fn get_target_dimensions() -> Vec<usize> {
    if let Ok(val) = std::env::var("CRITERION_DIMENSION").or_else(|_| std::env::var("DIMENSION")) {
        if let Ok(dim) = val.trim().parse::<usize>() {
            return vec![dim];
        }
    }
    vec![128, 768, 1536]
}

/// Checks if a given metric benchmark group should be executed.
/// Reads from `CRITERION_METRICS` / `METRICS` env var (comma-separated list),
/// or defaults to running all if unset.
fn should_bench_metric(group_name: &str) -> bool {
    if let Ok(val) = std::env::var("CRITERION_METRICS").or_else(|_| std::env::var("METRICS")) {
        let metrics: Vec<&str> = val
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !metrics.is_empty() {
            return metrics.iter().any(|&m| m.eq_ignore_ascii_case(group_name));
        }
    }
    true
}

/// Generates a deterministic pseudo-random vector of the given dimension.
fn make_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i * seed + 13) % 1000) as f32 * 0.001)
        .collect()
}

/// Generates a positive pseudo-random vector for metrics that require non-negative inputs (e.g., Hellinger).
fn make_positive_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| (((i * seed + 17) % 1000) as f32 * 0.001).abs() + 0.001)
        .collect()
}

fn bench_dot_product(c: &mut Criterion) {
    if !should_bench_metric("dot_product") {
        return;
    }
    let mut group = c.benchmark_group("dot_product");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.dot_product(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.dot_product(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_l2_squared(c: &mut Criterion) {
    if !should_bench_metric("l2_squared") {
        return;
    }
    let mut group = c.benchmark_group("l2_squared");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.l2_squared(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.l2_squared(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    if !should_bench_metric("cosine_similarity") {
        return;
    }
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.cosine_similarity(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.cosine_similarity(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_manhattan(c: &mut Criterion) {
    if !should_bench_metric("manhattan") {
        return;
    }
    let mut group = c.benchmark_group("manhattan");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.manhattan(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.manhattan(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_minkowski(c: &mut Criterion) {
    if !should_bench_metric("minkowski_p3") {
        return;
    }
    let mut group = c.benchmark_group("minkowski_p3");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.minkowski(black_box(&a), black_box(&b), 3.0));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.minkowski(black_box(&a), black_box(&b), 3.0));
                },
            );
        }
    }

    group.finish();
}

fn bench_chebyshev(c: &mut Criterion) {
    if !should_bench_metric("chebyshev") {
        return;
    }
    let mut group = c.benchmark_group("chebyshev");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.chebyshev(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.chebyshev(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_hamming(c: &mut Criterion) {
    if !should_bench_metric("hamming") {
        return;
    }
    let mut group = c.benchmark_group("hamming");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.hamming(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.hamming(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_mahalanobis(c: &mut Criterion) {
    if !should_bench_metric("mahalanobis") {
        return;
    }
    let mut group = c.benchmark_group("mahalanobis");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.mahalanobis(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.mahalanobis(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_jaccard(c: &mut Criterion) {
    if !should_bench_metric("jaccard") {
        return;
    }
    let mut group = c.benchmark_group("jaccard");

    for dim in get_target_dimensions() {
        let a = make_vector(dim, 7);
        let b = make_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.jaccard(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.jaccard(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn bench_hellinger(c: &mut Criterion) {
    if !should_bench_metric("hellinger") {
        return;
    }
    let mut group = c.benchmark_group("hellinger");

    for dim in get_target_dimensions() {
        let a = make_positive_vector(dim, 7);
        let b = make_positive_vector(dim, 13);

        group.throughput(Throughput::Elements(dim as u64));

        let scalar = DistanceEngine::scalar();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar.hellinger(black_box(&a), black_box(&b)));
        });

        let auto = DistanceEngine::auto();
        if auto.backend() != SimdBackend::Scalar {
            group.bench_with_input(
                BenchmarkId::new(format!("{}", auto.backend()), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| auto.hellinger(black_box(&a), black_box(&b)));
                },
            );
        }
    }

    group.finish();
}

fn custom_criterion() -> Criterion {
    let sample_size = std::env::var("CRITERION_SAMPLE_SIZE")
        .or_else(|_| std::env::var("SAMPLE_SIZE"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);

    let warmup_ms = std::env::var("CRITERION_WARMUP_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500);

    let measure_ms = std::env::var("CRITERION_MEASURE_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1000);

    Criterion::default()
        .warm_up_time(Duration::from_millis(warmup_ms))
        .measurement_time(Duration::from_millis(measure_ms))
        .sample_size(sample_size.max(10))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_dot_product,
              bench_l2_squared,
              bench_cosine_similarity,
              bench_manhattan,
              bench_minkowski,
              bench_chebyshev,
              bench_hamming,
              bench_mahalanobis,
              bench_jaccard,
              bench_hellinger
}
criterion_main!(benches);
