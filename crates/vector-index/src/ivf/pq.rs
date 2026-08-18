//! Product Quantization (PQ) engine for vector compression and Asymmetric Distance Computation (ADC).
//!
//! Divides high-dimensional vectors into $M$ sub-vectors and quantizes each sub-space
//! into 1-byte (`u8`) codebook centroid indices, delivering up to 96% memory reduction.

use rayon::prelude::*;
use vector_simd::DistanceEngine;

use super::kmeans::KMeans;
use crate::DistanceMetric;

/// Trained Product Quantizer holding codebooks for all sub-vector spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductQuantizer {
    /// Total dimensionality of original vectors ($D$).
    pub dimension: usize,
    /// Number of sub-vector segments ($M$).
    pub num_subvectors: usize,
    /// Dimensionality of each sub-vector ($d_s = D / M$).
    pub sub_dimension: usize,
    /// Number of centroids per codebook ($K^*$, typically 256).
    pub sub_clusters: usize,
    /// Flat buffer of all codebooks.
    /// Total elements: $M \times K^* \times d_s$.
    pub codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Creates a new `ProductQuantizer` from pre-trained codebooks.
    pub fn new(
        dimension: usize,
        num_subvectors: usize,
        sub_clusters: usize,
        codebooks: Vec<f32>,
    ) -> Self {
        assert_eq!(dimension % num_subvectors, 0);
        let sub_dimension = dimension / num_subvectors;
        assert_eq!(
            codebooks.len(),
            num_subvectors * sub_clusters * sub_dimension
        );

        Self {
            dimension,
            num_subvectors,
            sub_dimension,
            sub_clusters,
            codebooks,
        }
    }

    /// Trains a `ProductQuantizer` on training vectors/residuals in parallel across sub-spaces.
    pub fn train(
        data: &[f32],
        dimension: usize,
        num_subvectors: usize,
        sub_clusters: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(dimension % num_subvectors, 0);
        let sub_dimension = dimension / num_subvectors;
        let n = data.len() / dimension;

        // Train codebooks for each sub-space in parallel using rayon
        let sub_codebooks: Vec<Vec<f32>> = (0..num_subvectors)
            .into_par_iter()
            .map(|m| {
                // 1. Extract m-th sub-vector across all N training samples
                let mut sub_data = Vec::with_capacity(n * sub_dimension);
                for i in 0..n {
                    let vec_start = i * dimension + m * sub_dimension;
                    sub_data.extend_from_slice(&data[vec_start..vec_start + sub_dimension]);
                }

                // 2. Run k-Means on the sub-space
                let km = KMeans::fit(
                    &sub_data,
                    sub_dimension,
                    sub_clusters,
                    max_iters,
                    tolerance,
                    metric,
                );

                // If fewer centroids than sub_clusters were generated, pad with zeros
                let mut cb = km.centroids;
                if km.k < sub_clusters {
                    cb.resize(sub_clusters * sub_dimension, 0.0);
                }
                cb
            })
            .collect();

        // Flatten all sub-codebooks into a single contiguous array
        let mut codebooks = Vec::with_capacity(num_subvectors * sub_clusters * sub_dimension);
        for cb in sub_codebooks {
            codebooks.extend_from_slice(&cb);
        }

        Self {
            dimension,
            num_subvectors,
            sub_dimension,
            sub_clusters,
            codebooks,
        }
    }

    /// Trains a `ProductQuantizer` sequentially (single-threaded) on training vectors/residuals.
    pub fn train_sequential(
        data: &[f32],
        dimension: usize,
        num_subvectors: usize,
        sub_clusters: usize,
        max_iters: usize,
        tolerance: f32,
        metric: DistanceMetric,
    ) -> Self {
        assert_eq!(dimension % num_subvectors, 0);
        let sub_dimension = dimension / num_subvectors;
        let n = data.len() / dimension;

        // Train codebooks for each sub-space sequentially
        let sub_codebooks: Vec<Vec<f32>> = (0..num_subvectors)
            .map(|m| {
                // 1. Extract m-th sub-vector across all N training samples
                let mut sub_data = Vec::with_capacity(n * sub_dimension);
                for i in 0..n {
                    let vec_start = i * dimension + m * sub_dimension;
                    sub_data.extend_from_slice(&data[vec_start..vec_start + sub_dimension]);
                }

                // 2. Run k-Means sequentially on the sub-space
                let km = KMeans::fit_sequential(
                    &sub_data,
                    sub_dimension,
                    sub_clusters,
                    max_iters,
                    tolerance,
                    metric,
                );

                // If fewer centroids than sub_clusters were generated, pad with zeros
                let mut cb = km.centroids;
                if km.k < sub_clusters {
                    cb.resize(sub_clusters * sub_dimension, 0.0);
                }
                cb
            })
            .collect();

        // Flatten all sub-codebooks into a single contiguous array
        let mut codebooks = Vec::with_capacity(num_subvectors * sub_clusters * sub_dimension);
        for cb in sub_codebooks {
            codebooks.extend_from_slice(&cb);
        }

        Self {
            dimension,
            num_subvectors,
            sub_dimension,
            sub_clusters,
            codebooks,
        }
    }

    /// Returns a slice view of centroid $k$ in sub-space $m$.
    #[inline]
    pub fn get_sub_centroid(&self, m: usize, k: usize) -> &[f32] {
        let start = (m * self.sub_clusters + k) * self.sub_dimension;
        &self.codebooks[start..start + self.sub_dimension]
    }

    /// Encodes a single vector/residual into an $M$-byte PQ code (`[u8; M]`).
    #[inline]
    pub fn encode(&self, vector: &[f32], out_code: &mut [u8], metric: DistanceMetric) {
        assert_eq!(vector.len(), self.dimension);
        assert_eq!(out_code.len(), self.num_subvectors);

        let engine = DistanceEngine::auto();

        for m in 0..self.num_subvectors {
            let sub_vec = &vector[m * self.sub_dimension..(m + 1) * self.sub_dimension];
            let codebook_start = m * self.sub_clusters * self.sub_dimension;
            let codebook_slice = &self.codebooks
                [codebook_start..codebook_start + self.sub_clusters * self.sub_dimension];

            let (best_cluster, _) = KMeans::find_nearest_centroid(
                sub_vec,
                codebook_slice,
                self.sub_dimension,
                metric,
                &engine,
            );

            out_code[m] = best_cluster as u8;
        }
    }

    /// Encodes $N$ vectors/residuals in parallel into a flat byte slice of length $N \times M$.
    pub fn encode_batch(&self, data: &[f32], out_codes: &mut [u8], metric: DistanceMetric) {
        let n = data.len() / self.dimension;
        assert_eq!(out_codes.len(), n * self.num_subvectors);

        out_codes
            .par_chunks_mut(self.num_subvectors)
            .enumerate()
            .for_each(|(i, code_slice)| {
                let vec_slice = &data[i * self.dimension..(i + 1) * self.dimension];
                self.encode(vec_slice, code_slice, metric);
            });
    }

    /// Reconstructs / decodes an $M$-byte code into an approximate $D$-dimensional `f32` vector.
    pub fn decode(&self, code: &[u8], out_vector: &mut [f32]) {
        assert_eq!(code.len(), self.num_subvectors);
        assert_eq!(out_vector.len(), self.dimension);

        for (m, &byte) in code.iter().enumerate().take(self.num_subvectors) {
            let k = byte as usize;
            let centroid = self.get_sub_centroid(m, k);
            let out_start = m * self.sub_dimension;
            out_vector[out_start..out_start + self.sub_dimension].copy_from_slice(centroid);
        }
    }

    /// Computes the Asymmetric Distance Computation (ADC) Lookup Table for a query/residual vector.
    ///
    /// Table shape: $M \times K^*$, flat length $M \times sub\_clusters$.
    /// Entry `lut[m * sub_clusters + k]` stores the distance from query sub-vector $m$ to centroid $k$.
    pub fn compute_adc_lut(
        &self,
        query: &[f32],
        lut: &mut [f32],
        metric: DistanceMetric,
        engine: &DistanceEngine,
    ) {
        assert_eq!(query.len(), self.dimension);
        assert_eq!(lut.len(), self.num_subvectors * self.sub_clusters);

        for m in 0..self.num_subvectors {
            let q_sub = &query[m * self.sub_dimension..(m + 1) * self.sub_dimension];
            let lut_offset = m * self.sub_clusters;

            for k in 0..self.sub_clusters {
                let centroid = self.get_sub_centroid(m, k);
                let dist = match metric {
                    DistanceMetric::L2Squared => engine.l2_squared(q_sub, centroid),
                    DistanceMetric::DotProduct => engine.dot_product(q_sub, centroid),
                    DistanceMetric::CosineSimilarity => engine.cosine_similarity(q_sub, centroid),
                    DistanceMetric::Manhattan => engine.manhattan(q_sub, centroid),
                    DistanceMetric::Minkowski => engine.minkowski(q_sub, centroid, 3.0),
                    DistanceMetric::Chebyshev => engine.chebyshev(q_sub, centroid),
                    DistanceMetric::Hamming => engine.hamming(q_sub, centroid),
                    DistanceMetric::Mahalanobis => engine.mahalanobis(q_sub, centroid),
                    DistanceMetric::Jaccard => engine.jaccard(q_sub, centroid),
                    DistanceMetric::Hellinger => engine.hellinger(q_sub, centroid),
                };
                lut[lut_offset + k] = dist;
            }
        }
    }

    /// Fast ADC distance calculation using precomputed lookup table.
    ///
    /// Computes $\sum_{m=0}^{M-1} LUT[m][code_m]$ via direct table index lookup and accumulation.
    #[inline]
    pub fn compute_distance_with_lut(&self, code: &[u8], lut: &[f32]) -> f32 {
        let mut total = 0.0f32;
        let sub_clusters = self.sub_clusters;

        // Unroll 4-way for high throughput
        let chunks = code.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut m = 0;

        for quad in chunks {
            let d0 = lut[m * sub_clusters + quad[0] as usize];
            let d1 = lut[(m + 1) * sub_clusters + quad[1] as usize];
            let d2 = lut[(m + 2) * sub_clusters + quad[2] as usize];
            let d3 = lut[(m + 3) * sub_clusters + quad[3] as usize];
            total += (d0 + d1) + (d2 + d3);
            m += 4;
        }

        for &c in remainder {
            total += lut[m * sub_clusters + c as usize];
            m += 1;
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_training_and_encoding() {
        let dimension = 8;
        let num_subvectors = 2; // sub_dimension = 4
        let sub_clusters = 4;

        // Generate synthetic training vectors
        let mut data = Vec::new();
        for i in 0..20 {
            let val = i as f32;
            data.extend_from_slice(&[val, val + 1.0, val + 2.0, val + 3.0, val, val, val, val]);
        }

        let pq = ProductQuantizer::train(
            &data,
            dimension,
            num_subvectors,
            sub_clusters,
            15,
            1e-4,
            DistanceMetric::L2Squared,
        );

        assert_eq!(pq.dimension, 8);
        assert_eq!(pq.num_subvectors, 2);
        assert_eq!(pq.sub_dimension, 4);
        assert_eq!(pq.sub_clusters, 4);
        assert_eq!(pq.codebooks.len(), 2 * 4 * 4);

        // Test single vector encoding
        let mut code = [0u8; 2];
        let sample = &data[0..8];
        pq.encode(sample, &mut code, DistanceMetric::L2Squared);
        assert!((code[0] as usize) < sub_clusters);
        assert!((code[1] as usize) < sub_clusters);

        // Test reconstruction
        let mut decoded = vec![0.0f32; 8];
        pq.decode(&code, &mut decoded);
        assert_eq!(decoded.len(), 8);

        // Test batch encoding
        let mut batch_codes = vec![0u8; 20 * 2];
        pq.encode_batch(&data, &mut batch_codes, DistanceMetric::L2Squared);
        assert_eq!(batch_codes[0..2], code);
    }

    #[test]
    fn test_pq_adc_lut() {
        let dimension = 4;
        let num_subvectors = 2; // sub_dimension = 2
        let sub_clusters = 2;

        let codebooks = vec![
            // Sub-space 0
            0.0, 0.0, // Centroid 0
            1.0, 1.0, // Centroid 1
            // Sub-space 1
            0.0, 0.0, // Centroid 0
            2.0, 2.0, // Centroid 1
        ];

        let pq = ProductQuantizer::new(dimension, num_subvectors, sub_clusters, codebooks);
        let engine = DistanceEngine::auto();

        let query = [1.0, 1.0, 2.0, 2.0];
        let mut lut = vec![0.0f32; 2 * 2];
        pq.compute_adc_lut(&query, &mut lut, DistanceMetric::L2Squared, &engine);

        // LUT check:
        // Sub 0 query: [1.0, 1.0] -> dist to [0,0] = 2.0, dist to [1,1] = 0.0
        assert_eq!(lut[0], 2.0);
        assert_eq!(lut[1], 0.0);
        // Sub 1 query: [2.0, 2.0] -> dist to [0,0] = 8.0, dist to [2,2] = 0.0
        assert_eq!(lut[2], 8.0);
        assert_eq!(lut[3], 0.0);

        // Test ADC distance for code [1, 1] (which corresponds to exact centroids)
        let dist = pq.compute_distance_with_lut(&[1, 1], &lut);
        assert_eq!(dist, 0.0); // 0.0 + 0.0

        // Test ADC distance for code [0, 0]
        let dist_0 = pq.compute_distance_with_lut(&[0, 0], &lut);
        assert_eq!(dist_0, 10.0); // 2.0 + 8.0
    }

    #[test]
    fn test_pq_get_sub_centroid_accessor() {
        let codebooks = vec![
            1.0, 2.0, // m=0, k=0
            3.0, 4.0, // m=0, k=1
            5.0, 6.0, // m=1, k=0
            7.0, 8.0, // m=1, k=1
        ];
        let pq = ProductQuantizer::new(4, 2, 2, codebooks);
        assert_eq!(pq.get_sub_centroid(0, 0), &[1.0, 2.0]);
        assert_eq!(pq.get_sub_centroid(0, 1), &[3.0, 4.0]);
        assert_eq!(pq.get_sub_centroid(1, 0), &[5.0, 6.0]);
        assert_eq!(pq.get_sub_centroid(1, 1), &[7.0, 8.0]);
    }

    #[test]
    fn test_pq_compute_distance_with_lut_remainder_unroll() {
        // M = 5 subvectors (4-way chunk + 1 remainder), sub_clusters = 2, sub_dim = 1 (dim = 5)
        let mut codebooks = Vec::new();
        for _ in 0..5 {
            codebooks.push(0.0); // k=0
            codebooks.push(1.0); // k=1
        }

        let pq = ProductQuantizer::new(5, 5, 2, codebooks);
        let engine = DistanceEngine::auto();

        let query = [1.0, 1.0, 1.0, 1.0, 1.0];
        let mut lut = vec![0.0f32; 5 * 2];
        pq.compute_adc_lut(&query, &mut lut, DistanceMetric::L2Squared, &engine);

        // Distance for code [1, 1, 1, 1, 1] (all exact matches) should be 0.0
        let dist_exact = pq.compute_distance_with_lut(&[1, 1, 1, 1, 1], &lut);
        assert!((dist_exact - 0.0).abs() < 1e-5);

        // Distance for code [0, 0, 0, 0, 0] should be 5 * (1-0)^2 = 5.0
        let dist_diff = pq.compute_distance_with_lut(&[0, 0, 0, 0, 0], &lut);
        assert!((dist_diff - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_pq_cosine_and_dot_product_metrics() {
        let dimension = 4;
        let num_subvectors = 2;
        let sub_clusters = 2;

        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        }

        let pq_cos = ProductQuantizer::train(
            &data,
            dimension,
            num_subvectors,
            sub_clusters,
            10,
            1e-4,
            DistanceMetric::CosineSimilarity,
        );
        let mut code_cos = [0u8; 2];
        pq_cos.encode(
            &[1.0, 0.0, 0.0, 1.0],
            &mut code_cos,
            DistanceMetric::CosineSimilarity,
        );

        let pq_dot = ProductQuantizer::train(
            &data,
            dimension,
            num_subvectors,
            sub_clusters,
            10,
            1e-4,
            DistanceMetric::DotProduct,
        );
        let mut code_dot = [0u8; 2];
        pq_dot.encode(
            &[1.0, 0.0, 0.0, 1.0],
            &mut code_dot,
            DistanceMetric::DotProduct,
        );
    }
}
