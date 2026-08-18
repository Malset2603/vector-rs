use vector_cuda::{
    CudaDeviceContext, CudaError, CudaKnnEngine, DistributedKMeansEngine, DistributedKnnEngine,
    GpuShardMode,
};
use vector_index::DistanceMetric;

#[test]
fn test_end_to_end_clustering_and_sharded_search() {
    let dimension = 4;
    let n_per_cluster = 30;

    let mut data = Vec::new();
    // Cluster 0 around 0.0
    for _ in 0..n_per_cluster {
        data.extend_from_slice(&[0.1, 0.0, -0.1, 0.05]);
    }
    // Cluster 1 around 10.0
    for _ in 0..n_per_cluster {
        data.extend_from_slice(&[9.9, 10.1, 10.0, 9.95]);
    }
    // Cluster 2 around 20.0
    for _ in 0..n_per_cluster {
        data.extend_from_slice(&[20.0, 19.9, 20.1, 20.05]);
    }

    // 1. Distributed K-Means Clustering across 4 GPU ranks (emulator mode)
    let kmeans_engine = DistributedKMeansEngine::emulator(4);
    let kmeans_res = kmeans_engine.fit(&data, dimension, 3, 30, 1e-5, DistanceMetric::L2Squared);
    assert_eq!(kmeans_res.k, 3);
    assert_eq!(kmeans_res.dimension, dimension);

    // 2. Distributed K-NN Search across 3 GPU shards
    let knn_sharded = DistributedKnnEngine::emulator(
        &data,
        dimension,
        3,
        GpuShardMode::Sharded,
        DistanceMetric::L2Squared,
    );
    assert_eq!(knn_sharded.num_vectors(), 90);

    // Query near cluster 1
    let query_c1 = [10.0, 10.0, 10.0, 10.0];
    let top_results = knn_sharded.search(&query_c1, 5);
    assert_eq!(top_results.len(), 5);

    // All top 5 results should belong to cluster 1 (indices 30..60)
    for res in top_results {
        assert!(
            res.id >= 30 && res.id < 60,
            "Expected result id in cluster 1 range [30, 60), got {}",
            res.id
        );
        assert!(res.distance < 1.0);
    }
}

#[test]
fn test_single_gpu_vs_multi_gpu_consistency() {
    let dimension = 3;
    let mut data = Vec::new();
    for i in 0..60 {
        let v = [
            (i as f32) * 0.5,
            ((i * i) as f32) * 0.05,
            (i as f32 + 3.0).sin(),
        ];
        data.extend_from_slice(&v);
    }

    let single_gpu = CudaKnnEngine::new(&data, dimension, DistanceMetric::L2Squared);
    let ddp_sharded = DistributedKnnEngine::emulator(
        &data,
        dimension,
        4,
        GpuShardMode::Sharded,
        DistanceMetric::L2Squared,
    );
    let ddp_replicated = DistributedKnnEngine::emulator(
        &data,
        dimension,
        4,
        GpuShardMode::Replicated,
        DistanceMetric::L2Squared,
    );

    let batch_queries = vec![1.0, 2.0, 0.5, 15.0, 45.0, -0.5, 25.0, 120.0, 0.8];

    let single_res = single_gpu.search_batch(&batch_queries, 5);
    let sharded_res = ddp_sharded.search_batch(&batch_queries, 5);
    let replicated_res = ddp_replicated.search_batch(&batch_queries, 5);

    assert_eq!(single_res.len(), 3);
    assert_eq!(sharded_res.len(), 3);
    assert_eq!(replicated_res.len(), 3);

    for q in 0..3 {
        for rank in 0..5 {
            assert_eq!(
                sharded_res[q][rank].id, single_res[q][rank].id,
                "Mismatch in query {} rank {} between single and sharded",
                q, rank
            );
            assert!(
                (sharded_res[q][rank].distance - single_res[q][rank].distance).abs() < 1e-4,
                "Distance mismatch in query {} rank {}",
                q,
                rank
            );

            assert_eq!(
                replicated_res[q][rank].id, single_res[q][rank].id,
                "Mismatch in query {} rank {} between single and replicated",
                q, rank
            );
            assert!((replicated_res[q][rank].distance - single_res[q][rank].distance).abs() < 1e-4);
        }
    }
}

#[test]
fn test_all_metrics_consistency() {
    let dimension = 2;
    let data = vec![0.8, 0.6, 0.8, 0.0, 0.0, 0.6, 0.1, 0.1, -1.0, -1.0];

    let metrics = [
        DistanceMetric::L2Squared,
        DistanceMetric::CosineSimilarity,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
        DistanceMetric::Minkowski,
        DistanceMetric::Chebyshev,
        DistanceMetric::Hamming,
        DistanceMetric::Mahalanobis,
        DistanceMetric::Jaccard,
        DistanceMetric::Hellinger,
    ];

    let query = [0.8, 0.6];

    for metric in metrics {
        let single_gpu = CudaKnnEngine::new(&data, dimension, metric);
        let ddp_gpu =
            DistributedKnnEngine::emulator(&data, dimension, 2, GpuShardMode::Sharded, metric);

        let res_single = single_gpu.search(&query, 3);
        let res_ddp = ddp_gpu.search(&query, 3);

        assert_eq!(res_single.len(), 3);
        assert_eq!(res_ddp.len(), 3);

        for i in 0..3 {
            assert!(
                (res_single[i].distance - res_ddp[i].distance).abs() < 1e-4,
                "Metric {:?} rank {}",
                metric,
                i
            );
        }
    }
}

#[test]
fn test_strict_hardware_validation_error() {
    let available = CudaDeviceContext::device_count();
    let requested = available + 10;

    let res = DistributedKMeansEngine::try_new(requested);
    assert!(res.is_err());
    assert_eq!(
        res.unwrap_err(),
        CudaError::InsufficientDevices {
            requested,
            available,
        }
    );
}
