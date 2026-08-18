//! High-performance parallel k-Means clustering engine.
//!
//! Provides Lloyd's algorithm with SIMD-accelerated distance evaluations and
//! multi-threaded parallel assignment/update steps via `rayon`.

use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use vector_simd::DistanceEngine;

use crate::DistanceMetric;

/// Results from fitting a k-Means clustering model.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult {
    /// Flat buffer containing $k$ centroid vectors, length $k \times D$.
    pub centroids: Vec<f32>,
    /// Dimensionality of each centroid.
    pub dimension: usize,
    /// Number of clusters ($k$).
    pub k: usize,
    /// Number of Lloyd's iterations completed before convergence or limit.
    pub iterations: usize,
    /// Sum of squared errors (inertia) upon completion.
    pub inertia: f32,
}

impl KMeansResult {
    /// Returns a slice view of centroid $i \in [0, k-1]$.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        let start = i * self.dimension;
        &self.centroids[start..start + self.dimension]
    }
}

/// Standalone k-Means clustering engine.
pub struct KMeans;

impl KMeans {
    /// Fits $k$ centroids on the provided flat data buffer containing $N$ vectors of dimension $D$.
    pub fn fit(
        data: &[f32],
        dimension: usize,
        k: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> KMeansResult {
        let n = data.len() / dimension;
        if n == 0 || k == 0 {
            return KMeansResult {
                centroids: Vec::new(),
                dimension,
                k: 0,
                iterations: 0,
                inertia: 0.0,
            };
        }

        let effective_k = k.min(n);
        let engine = DistanceEngine::auto();

        // 1. Centroid Initialization: Random sample from dataset
        let mut centroids = Self::init_centroids_random(data, dimension, n, effective_k);

        let mut iterations = 0;
        let mut final_inertia = f32::MAX;

        for iter in 0..max_iters {
            iterations = iter + 1;

            // 2. Parallel Assignment: Map each vector to its nearest centroid
            // Thread-local accumulation of (count, sum_vector)
            let (cluster_counts, cluster_sums, total_inertia) = (0..n)
                .into_par_iter()
                .fold(
                    || {
                        (
                            vec![0usize; effective_k],
                            vec![0.0f32; effective_k * dimension],
                            0.0f32,
                        )
                    },
                    |(mut counts, mut sums, mut inertia), i| {
                        let vec_slice = &data[i * dimension..(i + 1) * dimension];
                        let (best_cluster, dist) = Self::find_nearest_centroid(
                            vec_slice, &centroids, dimension, metric, &engine,
                        );

                        counts[best_cluster] += 1;
                        let sum_start = best_cluster * dimension;
                        for d in 0..dimension {
                            sums[sum_start + d] += vec_slice[d];
                        }
                        inertia += dist;

                        (counts, sums, inertia)
                    },
                )
                .reduce(
                    || {
                        (
                            vec![0usize; effective_k],
                            vec![0.0f32; effective_k * dimension],
                            0.0f32,
                        )
                    },
                    |(mut c1, mut s1, i1), (c2, s2, i2)| {
                        for k_idx in 0..effective_k {
                            c1[k_idx] += c2[k_idx];
                        }
                        for idx in 0..(effective_k * dimension) {
                            s1[idx] += s2[idx];
                        }
                        (c1, s1, i1 + i2)
                    },
                );

            final_inertia = total_inertia;

            // 3. Update Centroids & Check Convergence
            let mut max_shift = 0.0f32;
            let mut rng = rand::thread_rng();

            for (k_idx, &count) in cluster_counts.iter().enumerate().take(effective_k) {
                let start = k_idx * dimension;

                if count > 0 {
                    let inv_count = 1.0 / (count as f32);
                    let mut shift = 0.0f32;

                    for d in 0..dimension {
                        let new_val = cluster_sums[start + d] * inv_count;
                        let diff = new_val - centroids[start + d];
                        shift += diff * diff;
                        centroids[start + d] = new_val;
                    }

                    if shift > max_shift {
                        max_shift = shift;
                    }
                } else {
                    // Empty cluster: reseed with a random vector from the dataset
                    let random_idx = rng.gen_range(0..n);
                    let sample_start = random_idx * dimension;
                    centroids[start..start + dimension]
                        .copy_from_slice(&data[sample_start..sample_start + dimension]);
                }
            }

            if max_shift <= tolerance {
                break;
            }
        }

        KMeansResult {
            centroids,
            dimension,
            k: effective_k,
            iterations,
            inertia: final_inertia,
        }
    }

    /// Fits $k$ centroids sequentially (single-threaded) on the provided flat data buffer.
    pub fn fit_sequential(
        data: &[f32],
        dimension: usize,
        k: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> KMeansResult {
        let n = data.len() / dimension;
        if n == 0 || k == 0 {
            return KMeansResult {
                centroids: Vec::new(),
                dimension,
                k: 0,
                iterations: 0,
                inertia: 0.0,
            };
        }

        let effective_k = k.min(n);
        let engine = DistanceEngine::auto();

        // 1. Centroid Initialization: Random sample from dataset
        let mut centroids = Self::init_centroids_random(data, dimension, n, effective_k);

        let mut iterations = 0;
        let mut final_inertia = f32::MAX;

        for iter in 0..max_iters {
            iterations = iter + 1;

            // 2. Sequential Assignment: Map each vector to its nearest centroid
            let mut cluster_counts = vec![0usize; effective_k];
            let mut cluster_sums = vec![0.0f32; effective_k * dimension];
            let mut total_inertia = 0.0f32;

            for i in 0..n {
                let vec_slice = &data[i * dimension..(i + 1) * dimension];
                let (best_cluster, dist) =
                    Self::find_nearest_centroid(vec_slice, &centroids, dimension, metric, &engine);

                cluster_counts[best_cluster] += 1;
                let sum_start = best_cluster * dimension;
                for d in 0..dimension {
                    cluster_sums[sum_start + d] += vec_slice[d];
                }
                total_inertia += dist;
            }

            final_inertia = total_inertia;

            // 3. Update Centroids & Check Convergence
            let mut max_shift = 0.0f32;
            let mut rng = rand::thread_rng();

            for (k_idx, &count) in cluster_counts.iter().enumerate().take(effective_k) {
                let start = k_idx * dimension;

                if count > 0 {
                    let inv_count = 1.0 / (count as f32);
                    let mut shift = 0.0f32;

                    for d in 0..dimension {
                        let new_val = cluster_sums[start + d] * inv_count;
                        let diff = new_val - centroids[start + d];
                        shift += diff * diff;
                        centroids[start + d] = new_val;
                    }

                    if shift > max_shift {
                        max_shift = shift;
                    }
                } else {
                    // Empty cluster: reseed with a random vector from the dataset
                    let random_idx = rng.gen_range(0..n);
                    let sample_start = random_idx * dimension;
                    centroids[start..start + dimension]
                        .copy_from_slice(&data[sample_start..sample_start + dimension]);
                }
            }

            if max_shift <= tolerance {
                break;
            }
        }

        KMeansResult {
            centroids,
            dimension,
            k: effective_k,
            iterations,
            inertia: final_inertia,
        }
    }

    /// Finds the closest centroid index and distance for a given vector.
    #[inline]
    pub fn find_nearest_centroid(
        vector: &[f32],
        centroids: &[f32],
        dimension: usize,
        metric: DistanceMetric,
        engine: &DistanceEngine,
    ) -> (usize, f32) {
        let k = centroids.len() / dimension;
        let mut best_idx = 0;
        let mut best_score = if metric.higher_is_better() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };

        for c_idx in 0..k {
            let c_start = c_idx * dimension;
            let c_slice = &centroids[c_start..c_start + dimension];

            let dist = match metric {
                DistanceMetric::L2Squared => engine.l2_squared(vector, c_slice),
                DistanceMetric::DotProduct => engine.dot_product(vector, c_slice),
                DistanceMetric::CosineSimilarity => engine.cosine_similarity(vector, c_slice),
                DistanceMetric::Manhattan => engine.manhattan(vector, c_slice),
                DistanceMetric::Minkowski => engine.minkowski(vector, c_slice, 3.0),
                DistanceMetric::Chebyshev => engine.chebyshev(vector, c_slice),
                DistanceMetric::Hamming => engine.hamming(vector, c_slice),
                DistanceMetric::Mahalanobis => engine.mahalanobis(vector, c_slice),
                DistanceMetric::Jaccard => engine.jaccard(vector, c_slice),
                DistanceMetric::Hellinger => engine.hellinger(vector, c_slice),
            };

            let is_better = if metric.higher_is_better() {
                dist > best_score
            } else {
                dist < best_score
            };

            if is_better {
                best_score = dist;
                best_idx = c_idx;
            }
        }

        (best_idx, best_score)
    }

    /// Finds the top-$P$ (`nprobe`) nearest centroids for a query vector, sorted by proximity.
    pub fn find_top_centroids(
        query: &[f32],
        centroids: &[f32],
        dimension: usize,
        nprobe: usize,
        metric: DistanceMetric,
        engine: &DistanceEngine,
    ) -> Vec<(usize, f32)> {
        let k = centroids.len() / dimension;
        let target_p = nprobe.min(k);
        if target_p == 0 {
            return Vec::new();
        }

        let mut distances: Vec<(usize, f32)> = (0..k)
            .map(|c_idx| {
                let c_start = c_idx * dimension;
                let c_slice = &centroids[c_start..c_start + dimension];

                let score = match metric {
                    DistanceMetric::L2Squared => engine.l2_squared(query, c_slice),
                    DistanceMetric::DotProduct => engine.dot_product(query, c_slice),
                    DistanceMetric::CosineSimilarity => engine.cosine_similarity(query, c_slice),
                    DistanceMetric::Manhattan => engine.manhattan(query, c_slice),
                    DistanceMetric::Minkowski => engine.minkowski(query, c_slice, 3.0),
                    DistanceMetric::Chebyshev => engine.chebyshev(query, c_slice),
                    DistanceMetric::Hamming => engine.hamming(query, c_slice),
                    DistanceMetric::Mahalanobis => engine.mahalanobis(query, c_slice),
                    DistanceMetric::Jaccard => engine.jaccard(query, c_slice),
                    DistanceMetric::Hellinger => engine.hellinger(query, c_slice),
                };

                (c_idx, score)
            })
            .collect();

        if metric.higher_is_better() {
            distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        distances.truncate(target_p);
        distances
    }

    /// Initializes centroids by random sampling without replacement from dataset.
    fn init_centroids_random(data: &[f32], dimension: usize, n: usize, k: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Vec::with_capacity(k * dimension);
        for &idx in indices.iter().take(k) {
            let start = idx * dimension;
            centroids.extend_from_slice(&data[start..start + dimension]);
        }

        centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_empty() {
        let res = KMeans::fit(&[], 4, 2, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res.k, 0);
        assert!(res.centroids.is_empty());
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1];
        let res = KMeans::fit(&data, 4, 1, 10, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res.k, 1);
        assert_eq!(res.dimension, 4);

        let c = res.centroid(0);
        assert!((c[0] - 1.05).abs() < 1e-3);
        assert!((c[1] - 2.05).abs() < 1e-3);
    }

    #[test]
    fn test_kmeans_two_distinct_clusters() {
        // Cluster 1 around (0, 0), Cluster 2 around (10, 10)
        let data = vec![
            0.0, 0.1, 0.1, 0.0, -0.1, 0.0, 10.0, 10.1, 10.1, 10.0, 9.9, 10.0,
        ];

        let res = KMeans::fit(&data, 2, 2, 20, 1e-5, DistanceMetric::L2Squared);
        assert_eq!(res.k, 2);

        let c0 = res.centroid(0);
        let c1 = res.centroid(1);

        // One centroid should be near 0, other near 10
        let (near_0, near_10) = if c0[0] < 5.0 { (c0, c1) } else { (c1, c0) };

        assert!(near_0[0].abs() < 0.5);
        assert!(near_0[1].abs() < 0.5);
        assert!((near_10[0] - 10.0).abs() < 0.5);
        assert!((near_10[1] - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_find_top_centroids() {
        let engine = DistanceEngine::auto();
        let centroids = vec![
            0.0, 0.0, // Centroid 0
            5.0, 5.0, // Centroid 1
            10.0, 10.0, // Centroid 2
        ];

        let query = [1.0, 1.0];
        let top2 = KMeans::find_top_centroids(
            &query,
            &centroids,
            2,
            2,
            DistanceMetric::L2Squared,
            &engine,
        );

        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // Closest is (0,0)
        assert_eq!(top2[1].0, 1); // Second is (5,5)
    }

    #[test]
    fn test_kmeans_result_accessors_and_traits() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let res1 = KMeans::fit(&data, 2, 2, 5, 1e-4, DistanceMetric::L2Squared);
        let res2 = res1.clone();
        assert_eq!(res1, res2);

        assert_eq!(res1.centroid(0).len(), 2);
        assert_eq!(res1.centroid(1).len(), 2);

        let debug_str = format!("{:?}", res1);
        assert!(debug_str.contains("KMeansResult"));
    }

    #[test]
    fn test_kmeans_k_greater_than_n() {
        let data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        // 3 vectors in 2D, request k = 10
        let res = KMeans::fit(&data, 2, 10, 5, 1e-4, DistanceMetric::L2Squared);
        assert_eq!(res.k, 3);
        assert_eq!(res.centroids.len(), 3 * 2);
    }

    #[test]
    fn test_kmeans_cosine_and_dot_product_metrics() {
        let data = vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9];

        let res_cos = KMeans::fit(&data, 2, 2, 15, 1e-4, DistanceMetric::CosineSimilarity);
        assert_eq!(res_cos.k, 2);

        let res_dot = KMeans::fit(&data, 2, 2, 15, 1e-4, DistanceMetric::DotProduct);
        assert_eq!(res_dot.k, 2);
    }

    #[test]
    fn test_find_top_centroids_similarity_and_edge_cases() {
        let engine = DistanceEngine::auto();
        let centroids = vec![
            1.0, 0.0, // Centroid 0 (X axis)
            0.0, 1.0, // Centroid 1 (Y axis)
            -1.0, 0.0, // Centroid 2 (-X axis)
        ];

        let query = [0.99, 0.01];

        // Cosine similarity: highest similarity first
        let top = KMeans::find_top_centroids(
            &query,
            &centroids,
            2,
            2,
            DistanceMetric::CosineSimilarity,
            &engine,
        );
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // highest similarity with Centroid 0

        // Zero nprobe
        let top_zero = KMeans::find_top_centroids(
            &query,
            &centroids,
            2,
            0,
            DistanceMetric::CosineSimilarity,
            &engine,
        );
        assert!(top_zero.is_empty());

        // nprobe > k
        let top_all = KMeans::find_top_centroids(
            &query,
            &centroids,
            2,
            100,
            DistanceMetric::CosineSimilarity,
            &engine,
        );
        assert_eq!(top_all.len(), 3);
    }
}
