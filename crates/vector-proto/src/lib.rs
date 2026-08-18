//! # vector-proto
//!
//! Generated Protocol Buffers bindings and gRPC service client/server definitions
//! for the VectorRS distributed vector search engine.

pub mod proto {
    tonic::include_proto!("vector_proto");
}

pub use proto::vector_coordinator_service_client::VectorCoordinatorServiceClient;
pub use proto::vector_coordinator_service_server::{
    VectorCoordinatorService, VectorCoordinatorServiceServer,
};
pub use proto::vector_worker_service_client::VectorWorkerServiceClient;
pub use proto::vector_worker_service_server::{VectorWorkerService, VectorWorkerServiceServer};
pub use proto::*;

pub fn parse_distance_metric(val: i32) -> DistanceMetric {
    DistanceMetric::try_from(val).unwrap_or(DistanceMetric::Unspecified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn test_metric_conversion() {
        assert_eq!(parse_distance_metric(0), DistanceMetric::Unspecified);
        assert_eq!(parse_distance_metric(1), DistanceMetric::L2Squared);
        assert_eq!(parse_distance_metric(2), DistanceMetric::DotProduct);
        assert_eq!(parse_distance_metric(3), DistanceMetric::CosineSimilarity);
        assert_eq!(parse_distance_metric(4), DistanceMetric::Manhattan);
        assert_eq!(parse_distance_metric(5), DistanceMetric::Minkowski);
        assert_eq!(parse_distance_metric(6), DistanceMetric::Chebyshev);
        assert_eq!(parse_distance_metric(7), DistanceMetric::Hamming);
        assert_eq!(parse_distance_metric(8), DistanceMetric::Mahalanobis);
        assert_eq!(parse_distance_metric(9), DistanceMetric::Jaccard);
        assert_eq!(parse_distance_metric(10), DistanceMetric::Hellinger);
        assert_eq!(parse_distance_metric(-1), DistanceMetric::Unspecified);
        assert_eq!(parse_distance_metric(-100), DistanceMetric::Unspecified);
        assert_eq!(parse_distance_metric(11), DistanceMetric::Unspecified);
        assert_eq!(parse_distance_metric(99), DistanceMetric::Unspecified);
    }

    #[test]
    fn test_distance_metric_from_i32_and_str() {
        assert_eq!(DistanceMetric::try_from(1), Ok(DistanceMetric::L2Squared));
        assert_eq!(DistanceMetric::try_from(2), Ok(DistanceMetric::DotProduct));
        assert_eq!(
            DistanceMetric::try_from(3),
            Ok(DistanceMetric::CosineSimilarity)
        );
        assert_eq!(DistanceMetric::try_from(4), Ok(DistanceMetric::Manhattan));
        assert_eq!(DistanceMetric::try_from(5), Ok(DistanceMetric::Minkowski));
        assert_eq!(DistanceMetric::try_from(6), Ok(DistanceMetric::Chebyshev));
        assert_eq!(DistanceMetric::try_from(7), Ok(DistanceMetric::Hamming));
        assert_eq!(DistanceMetric::try_from(8), Ok(DistanceMetric::Mahalanobis));
        assert_eq!(DistanceMetric::try_from(9), Ok(DistanceMetric::Jaccard));
        assert_eq!(DistanceMetric::try_from(10), Ok(DistanceMetric::Hellinger));
        assert_eq!(DistanceMetric::try_from(0), Ok(DistanceMetric::Unspecified));
        assert!(DistanceMetric::try_from(99).is_err());

        assert_eq!(
            DistanceMetric::L2Squared.as_str_name(),
            "DISTANCE_METRIC_L2_SQUARED"
        );
        assert_eq!(
            DistanceMetric::DotProduct.as_str_name(),
            "DISTANCE_METRIC_DOT_PRODUCT"
        );
        assert_eq!(
            DistanceMetric::CosineSimilarity.as_str_name(),
            "DISTANCE_METRIC_COSINE_SIMILARITY"
        );
        assert_eq!(
            DistanceMetric::Manhattan.as_str_name(),
            "DISTANCE_METRIC_MANHATTAN"
        );
        assert_eq!(
            DistanceMetric::Minkowski.as_str_name(),
            "DISTANCE_METRIC_MINKOWSKI"
        );
        assert_eq!(
            DistanceMetric::Chebyshev.as_str_name(),
            "DISTANCE_METRIC_CHEBYSHEV"
        );
        assert_eq!(
            DistanceMetric::Hamming.as_str_name(),
            "DISTANCE_METRIC_HAMMING"
        );
        assert_eq!(
            DistanceMetric::Mahalanobis.as_str_name(),
            "DISTANCE_METRIC_MAHALANOBIS"
        );
        assert_eq!(
            DistanceMetric::Jaccard.as_str_name(),
            "DISTANCE_METRIC_JACCARD"
        );
        assert_eq!(
            DistanceMetric::Hellinger.as_str_name(),
            "DISTANCE_METRIC_HELLINGER"
        );
        assert_eq!(
            DistanceMetric::Unspecified.as_str_name(),
            "DISTANCE_METRIC_UNSPECIFIED"
        );
    }

    #[test]
    fn test_search_result_item_prost_roundtrip() {
        let item = SearchResultItem {
            id: 42,
            distance: 0.12345,
            shard_id: 3,
        };

        let mut buf = Vec::new();
        item.encode(&mut buf).unwrap();
        assert_eq!(buf.len(), item.encoded_len());

        let decoded = SearchResultItem::decode(&buf[..]).unwrap();
        assert_eq!(decoded.id, 42);
        assert!((decoded.distance - 0.12345).abs() < 1e-6);
        assert_eq!(decoded.shard_id, 3);
    }

    #[test]
    fn test_search_request_and_response_prost_roundtrip() {
        let req = SearchRequest {
            query_vector: vec![1.0, 2.5, -3.0, 0.0],
            k: 10,
            ef_search: 64,
            metric: DistanceMetric::CosineSimilarity as i32,
            shard_id: 2,
        };

        let mut req_buf = Vec::new();
        req.encode(&mut req_buf).unwrap();
        let decoded_req = SearchRequest::decode(&req_buf[..]).unwrap();

        assert_eq!(decoded_req.query_vector, vec![1.0, 2.5, -3.0, 0.0]);
        assert_eq!(decoded_req.k, 10);
        assert_eq!(decoded_req.ef_search, 64);
        assert_eq!(decoded_req.metric, DistanceMetric::CosineSimilarity as i32);
        assert_eq!(decoded_req.shard_id, 2);

        let resp = SearchResponse {
            results: vec![
                SearchResultItem {
                    id: 1,
                    distance: 0.1,
                    shard_id: 2,
                },
                SearchResultItem {
                    id: 2,
                    distance: 0.2,
                    shard_id: 2,
                },
            ],
            shard_id: 2,
            query_latency_micros: 150,
        };

        let mut resp_buf = Vec::new();
        resp.encode(&mut resp_buf).unwrap();
        let decoded_resp = SearchResponse::decode(&resp_buf[..]).unwrap();

        assert_eq!(decoded_resp.results.len(), 2);
        assert_eq!(decoded_resp.results[0].id, 1);
        assert_eq!(decoded_resp.results[1].id, 2);
        assert_eq!(decoded_resp.shard_id, 2);
        assert_eq!(decoded_resp.query_latency_micros, 150);
    }

    #[test]
    fn test_ping_and_stats_messages_prost_roundtrip() {
        let ping_req = PingRequest {
            client_id: "coordinator-1".to_string(),
        };
        let mut p_buf = Vec::new();
        ping_req.encode(&mut p_buf).unwrap();
        let dec_ping_req = PingRequest::decode(&p_buf[..]).unwrap();
        assert_eq!(dec_ping_req.client_id, "coordinator-1");

        let ping_resp = PingResponse {
            worker_id: "worker-shard-0".to_string(),
            shard_id: 0,
            num_vectors: 100_000,
            dimension: 128,
            ready: true,
        };
        let mut pr_buf = Vec::new();
        ping_resp.encode(&mut pr_buf).unwrap();
        let dec_ping_resp = PingResponse::decode(&pr_buf[..]).unwrap();
        assert_eq!(dec_ping_resp.worker_id, "worker-shard-0");
        assert_eq!(dec_ping_resp.shard_id, 0);
        assert_eq!(dec_ping_resp.num_vectors, 100_000);
        assert_eq!(dec_ping_resp.dimension, 128);
        assert!(dec_ping_resp.ready);

        let stats_req = StatsRequest { shard_id: 5 };
        let mut sr_buf = Vec::new();
        stats_req.encode(&mut sr_buf).unwrap();
        let dec_stats_req = StatsRequest::decode(&sr_buf[..]).unwrap();
        assert_eq!(dec_stats_req.shard_id, 5);

        let stats_resp = StatsResponse {
            shard_id: 5,
            num_vectors: 50_000,
            dimension: 64,
            index_type: "HNSW".to_string(),
        };
        let mut sresp_buf = Vec::new();
        stats_resp.encode(&mut sresp_buf).unwrap();
        let dec_stats_resp = StatsResponse::decode(&sresp_buf[..]).unwrap();
        assert_eq!(dec_stats_resp.shard_id, 5);
        assert_eq!(dec_stats_resp.num_vectors, 50_000);
        assert_eq!(dec_stats_resp.dimension, 64);
        assert_eq!(dec_stats_resp.index_type, "HNSW");
    }

    #[test]
    fn test_cluster_search_messages_prost_roundtrip() {
        let req = ClusterSearchRequest {
            query_vector: vec![0.5, 0.5, 0.5],
            k: 5,
            ef_search: 100,
            metric: DistanceMetric::DotProduct as i32,
        };
        let mut req_buf = Vec::new();
        req.encode(&mut req_buf).unwrap();
        let dec_req = ClusterSearchRequest::decode(&req_buf[..]).unwrap();
        assert_eq!(dec_req.query_vector, vec![0.5, 0.5, 0.5]);
        assert_eq!(dec_req.k, 5);
        assert_eq!(dec_req.ef_search, 100);
        assert_eq!(dec_req.metric, DistanceMetric::DotProduct as i32);

        let resp = ClusterSearchResponse {
            results: vec![SearchResultItem {
                id: 10,
                distance: 0.99,
                shard_id: 1,
            }],
            total_queried_shards: 4,
            successful_shards: 4,
            query_latency_micros: 2500,
        };
        let mut resp_buf = Vec::new();
        resp.encode(&mut resp_buf).unwrap();
        let dec_resp = ClusterSearchResponse::decode(&resp_buf[..]).unwrap();
        assert_eq!(dec_resp.results.len(), 1);
        assert_eq!(dec_resp.total_queried_shards, 4);
        assert_eq!(dec_resp.successful_shards, 4);
        assert_eq!(dec_resp.query_latency_micros, 2500);
    }

    #[test]
    fn test_message_defaults() {
        let item = SearchResultItem::default();
        assert_eq!(item.id, 0);
        assert_eq!(item.distance, 0.0);
        assert_eq!(item.shard_id, 0);

        let req = SearchRequest::default();
        assert!(req.query_vector.is_empty());
        assert_eq!(req.k, 0);
        assert_eq!(req.metric, 0);

        let resp = SearchResponse::default();
        assert!(resp.results.is_empty());
        assert_eq!(resp.shard_id, 0);

        let cluster_req = ClusterSearchRequest::default();
        assert!(cluster_req.query_vector.is_empty());
        assert_eq!(cluster_req.k, 0);

        let cluster_resp = ClusterSearchResponse::default();
        assert!(cluster_resp.results.is_empty());
        assert_eq!(cluster_resp.total_queried_shards, 0);
    }
}
