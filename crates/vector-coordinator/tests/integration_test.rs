use std::net::SocketAddr;
use std::sync::Arc;
use vector_coordinator::{ScatterGatherAggregator, WorkerRouter};
use vector_index::DistanceMetric;
use vector_index::hnsw::HnswConfig;
use vector_index::storage::HeapStorage;
use vector_proto::DistanceMetric as ProtoDistanceMetric;
use vector_worker::WorkerEngine;

/// Helper: Spawns a worker node in a background Tokio task on an OS-assigned ephemeral port.
async fn spawn_test_worker(
    shard_id: u32,
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    metric: DistanceMetric,
) -> SocketAddr {
    let mut storage = HeapStorage::new(dimension);
    for v in vectors {
        storage.push(&v).unwrap();
    }

    let engine = Arc::new(WorkerEngine::from_heap(
        shard_id,
        storage,
        HnswConfig::new(8, 50, 50),
        metric,
    ));

    // Get an available local port
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener); // release so tonic can bind

    let engine_clone = engine.clone();
    let worker_id = format!("worker-shard-{}", shard_id);
    tokio::spawn(async move {
        let _ = vector_worker::run_worker_server(addr, worker_id, engine_clone).await;
    });

    addr
}

#[tokio::test]
async fn test_distributed_scatter_gather_3_shards() {
    let dim = 4;

    // Shard 0 vectors
    let shard0_vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

    // Shard 1 vectors
    let shard1_vectors = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];

    // Shard 2 vectors
    let shard2_vectors = vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]];

    let addr0 = spawn_test_worker(0, shard0_vectors, dim, DistanceMetric::L2Squared).await;
    let addr1 = spawn_test_worker(1, shard1_vectors, dim, DistanceMetric::L2Squared).await;
    let addr2 = spawn_test_worker(2, shard2_vectors, dim, DistanceMetric::L2Squared).await;

    // Wait slightly for servers to accept connections
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let router = Arc::new(WorkerRouter::new());
    router
        .add_worker(&format!("http://{}", addr0))
        .await
        .unwrap();
    router
        .add_worker(&format!("http://{}", addr1))
        .await
        .unwrap();
    router
        .add_worker(&format!("http://{}", addr2))
        .await
        .unwrap();

    assert_eq!(router.len(), 3);

    // Verify health check
    let pings = router.health_check_all().await;
    assert_eq!(pings.len(), 3);
    for p in pings {
        assert!(p.unwrap().ready);
    }

    let aggregator = ScatterGatherAggregator::new(router);

    // Query closest to [1.0, 0.0, 0.0, 0.0]
    let query = vec![1.0, 0.1, 0.0, 0.0];
    let k = 3;

    let resp = aggregator
        .search_cluster(query, k, 50, ProtoDistanceMetric::L2Squared)
        .await;

    assert_eq!(resp.total_queried_shards, 3);
    assert_eq!(resp.successful_shards, 3);
    assert_eq!(resp.results.len(), 3);

    // Top result should be from Shard 0 (id=0: [1,0,0,0])
    assert_eq!(resp.results[0].shard_id, 0);
    assert_eq!(resp.results[0].id, 0);
    assert!(resp.results[0].distance < 0.05);

    // Second result should be from Shard 2 (id=0: [1,1,0,0]) or Shard 0 (id=1)
    assert!(resp.results[1].distance <= resp.results[2].distance);
}

#[tokio::test]
async fn test_scatter_gather_empty_workers() {
    let router = Arc::new(WorkerRouter::new());
    let aggregator = ScatterGatherAggregator::new(router);

    let resp = aggregator
        .search_cluster(vec![1.0, 2.0], 5, 50, ProtoDistanceMetric::L2Squared)
        .await;

    assert_eq!(resp.total_queried_shards, 0);
    assert_eq!(resp.results.len(), 0);
}

#[tokio::test]
async fn test_distributed_scatter_gather_dot_product_and_cosine() {
    let dim = 3;

    // Shard 0: Unit X
    let s0 = vec![vec![1.0, 0.0, 0.0]];
    // Shard 1: Unit Y
    let s1 = vec![vec![0.0, 1.0, 0.0]];
    // Shard 2: Unit Z
    let s2 = vec![vec![0.0, 0.0, 1.0]];

    let a0 = spawn_test_worker(0, s0, dim, DistanceMetric::CosineSimilarity).await;
    let a1 = spawn_test_worker(1, s1, dim, DistanceMetric::CosineSimilarity).await;
    let a2 = spawn_test_worker(2, s2, dim, DistanceMetric::CosineSimilarity).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let router = Arc::new(WorkerRouter::new());
    router.add_worker(&format!("http://{}", a0)).await.unwrap();
    router.add_worker(&format!("http://{}", a1)).await.unwrap();
    router.add_worker(&format!("http://{}", a2)).await.unwrap();

    let aggregator = ScatterGatherAggregator::new(router);

    // Query along X axis
    let query = vec![1.0, 0.0, 0.0];
    let resp = aggregator
        .search_cluster(query, 3, 50, ProtoDistanceMetric::CosineSimilarity)
        .await;

    assert_eq!(resp.results.len(), 3);
    // Highest cosine similarity (1.0) must be Shard 0
    assert_eq!(resp.results[0].shard_id, 0);
    assert_eq!(resp.results[0].id, 0);
    assert!((resp.results[0].distance - 1.0).abs() < 1e-4);
}

#[tokio::test]
async fn test_coordinator_service_direct_grpc_call() {
    use vector_coordinator::CoordinatorServiceImpl;
    use vector_proto::{ClusterSearchRequest, VectorCoordinatorService};

    let dim = 2;
    let s0 = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let s1 = vec![vec![10.0, 10.0], vec![20.0, 20.0]];

    let a0 = spawn_test_worker(0, s0, dim, DistanceMetric::L2Squared).await;
    let a1 = spawn_test_worker(1, s1, dim, DistanceMetric::L2Squared).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let router = Arc::new(WorkerRouter::new());
    router.add_worker(&format!("http://{}", a0)).await.unwrap();
    router.add_worker(&format!("http://{}", a1)).await.unwrap();

    let aggregator = Arc::new(ScatterGatherAggregator::new(router));
    let service = CoordinatorServiceImpl::new(aggregator);

    let req = tonic::Request::new(ClusterSearchRequest {
        query_vector: vec![0.1, 0.1],
        k: 2,
        ef_search: 32,
        metric: ProtoDistanceMetric::L2Squared as i32,
    });

    let res = service.search_cluster(req).await;
    assert!(res.is_ok());
    let search_resp = res.unwrap().into_inner();

    assert_eq!(search_resp.total_queried_shards, 2);
    assert_eq!(search_resp.successful_shards, 2);
    assert_eq!(search_resp.results.len(), 2);
    assert_eq!(search_resp.results[0].shard_id, 0);
    assert_eq!(search_resp.results[0].id, 0);
}

#[tokio::test]
async fn test_distributed_scatter_gather_large_k_handling() {
    let dim = 2;
    let s0 = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
    let s1 = vec![vec![3.0, 3.0], vec![4.0, 4.0]];

    let a0 = spawn_test_worker(0, s0, dim, DistanceMetric::L2Squared).await;
    let a1 = spawn_test_worker(1, s1, dim, DistanceMetric::L2Squared).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let router = Arc::new(WorkerRouter::new());
    router.add_worker(&format!("http://{}", a0)).await.unwrap();
    router.add_worker(&format!("http://{}", a1)).await.unwrap();

    let aggregator = ScatterGatherAggregator::new(router);

    // Request k = 50 when cluster total is 4
    let resp = aggregator
        .search_cluster(vec![0.0, 0.0], 50, 50, ProtoDistanceMetric::L2Squared)
        .await;

    assert_eq!(resp.total_queried_shards, 2);
    assert_eq!(resp.successful_shards, 2);
    assert_eq!(resp.results.len(), 4);
    assert_eq!(resp.results[0].shard_id, 0);
    assert_eq!(resp.results[0].id, 0);
    assert_eq!(resp.results[3].shard_id, 1);
    assert_eq!(resp.results[3].id, 1);
}
