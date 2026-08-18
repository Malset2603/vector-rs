//! Configuration parameters for Inverted File with Product Quantization (IVF-PQ) index.

use crate::types::{Result, VectorIndexError};

/// Default number of coarse Voronoi clusters (inverted lists).
pub const DEFAULT_NLIST: usize = 64;

/// Default number of coarse clusters probed during query search.
pub const DEFAULT_NPROBE: usize = 8;

/// Default number of sub-vector partitions for Product Quantization ($M$).
pub const DEFAULT_NUM_SUBVECTORS: usize = 8;

/// Default number of centroids per sub-space codebook ($K^* = 256$ for 1-byte encoding).
pub const DEFAULT_SUB_CLUSTERS: usize = 256;

/// Default maximum number of iterations for k-Means clustering.
pub const DEFAULT_MAX_KMEANS_ITERS: usize = 25;

/// Default centroid movement convergence tolerance for k-Means.
pub const DEFAULT_KMEANS_TOLERANCE: f32 = 1e-4;

/// Configuration options for building and querying an IVF-PQ index.
///
/// Controls the trade-off between memory compression ratio, index construction time,
/// query throughput (QPS), and retrieval recall.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfPqConfig {
    /// Number of coarse Voronoi partition clusters (inverted lists).
    pub nlist: usize,
    /// Number of nearest coarse clusters probed during query retrieval.
    pub nprobe: usize,
    /// Number of sub-vector segments for Product Quantization ($M$).
    /// The vector dimensionality $D$ must be evenly divisible by $M$.
    pub num_subvectors: usize,
    /// Number of centroids per sub-space codebook ($K^*$, max 256 for `u8` codes).
    pub sub_clusters: usize,
    /// Maximum number of Lloyd's iterations during k-Means clustering.
    pub max_kmeans_iters: usize,
    /// Convergence tolerance for k-Means centroid relocation.
    pub kmeans_tolerance: f32,
    /// Optional limit on the number of vectors randomly sampled for training.
    pub max_train_points: Option<usize>,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            nlist: DEFAULT_NLIST,
            nprobe: DEFAULT_NPROBE,
            num_subvectors: DEFAULT_NUM_SUBVECTORS,
            sub_clusters: DEFAULT_SUB_CLUSTERS,
            max_kmeans_iters: DEFAULT_MAX_KMEANS_ITERS,
            kmeans_tolerance: DEFAULT_KMEANS_TOLERANCE,
            max_train_points: None,
        }
    }
}

impl IvfPqConfig {
    /// Creates a new `IvfPqConfig` with the primary tuning parameters.
    pub fn new(nlist: usize, nprobe: usize, num_subvectors: usize) -> Self {
        Self {
            nlist,
            nprobe,
            num_subvectors,
            ..Default::default()
        }
    }

    /// Sets the number of centroids per sub-space codebook.
    pub fn with_sub_clusters(mut self, sub_clusters: usize) -> Self {
        self.sub_clusters = sub_clusters;
        self
    }

    /// Sets the maximum number of k-Means iterations.
    pub fn with_max_kmeans_iters(mut self, iters: usize) -> Self {
        self.max_kmeans_iters = iters;
        self
    }

    /// Sets the convergence tolerance for k-Means clustering.
    pub fn with_kmeans_tolerance(mut self, tol: f32) -> Self {
        self.kmeans_tolerance = tol;
        self
    }

    /// Sets a cap on the number of training points sampled during clustering.
    pub fn with_max_train_points(mut self, max_points: Option<usize>) -> Self {
        self.max_train_points = max_points;
        self
    }

    /// Validates the configuration against a target vector dimensionality $D$.
    pub fn validate(&self, dimension: usize) -> Result<()> {
        if self.nlist == 0 {
            return Err(VectorIndexError::InvalidConfig {
                reason: "nlist must be greater than 0".to_string(),
            });
        }
        if self.nprobe == 0 {
            return Err(VectorIndexError::InvalidConfig {
                reason: "nprobe must be greater than 0".to_string(),
            });
        }
        if self.nprobe > self.nlist {
            return Err(VectorIndexError::InvalidConfig {
                reason: format!(
                    "nprobe ({}) cannot be greater than nlist ({})",
                    self.nprobe, self.nlist
                ),
            });
        }
        if self.num_subvectors == 0 {
            return Err(VectorIndexError::InvalidConfig {
                reason: "num_subvectors must be greater than 0".to_string(),
            });
        }
        if !dimension.is_multiple_of(self.num_subvectors) {
            return Err(VectorIndexError::InvalidConfig {
                reason: format!(
                    "vector dimension ({}) must be divisible by num_subvectors ({})",
                    dimension, self.num_subvectors
                ),
            });
        }
        if self.sub_clusters == 0 || self.sub_clusters > 256 {
            return Err(VectorIndexError::InvalidConfig {
                reason: format!(
                    "sub_clusters must be between 1 and 256, got {}",
                    self.sub_clusters
                ),
            });
        }
        if self.max_kmeans_iters == 0 {
            return Err(VectorIndexError::InvalidConfig {
                reason: "max_kmeans_iters must be greater than 0".to_string(),
            });
        }
        if self.kmeans_tolerance < 0.0 {
            return Err(VectorIndexError::InvalidConfig {
                reason: "kmeans_tolerance must be non-negative".to_string(),
            });
        }

        Ok(())
    }

    /// Returns the dimensionality of each sub-vector $d_s = D / M$.
    #[inline]
    pub fn sub_dimension(&self, dimension: usize) -> usize {
        dimension / self.num_subvectors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IvfPqConfig::default();
        assert_eq!(config.nlist, 64);
        assert_eq!(config.nprobe, 8);
        assert_eq!(config.num_subvectors, 8);
        assert_eq!(config.sub_clusters, 256);
        assert_eq!(config.max_kmeans_iters, 25);
        assert!(config.validate(128).is_ok());
    }

    #[test]
    fn test_custom_builders() {
        let config = IvfPqConfig::new(32, 4, 4)
            .with_sub_clusters(64)
            .with_max_kmeans_iters(15)
            .with_kmeans_tolerance(1e-5)
            .with_max_train_points(Some(5000));

        assert_eq!(config.nlist, 32);
        assert_eq!(config.nprobe, 4);
        assert_eq!(config.num_subvectors, 4);
        assert_eq!(config.sub_clusters, 64);
        assert_eq!(config.max_kmeans_iters, 15);
        assert_eq!(config.kmeans_tolerance, 1e-5);
        assert_eq!(config.max_train_points, Some(5000));
        assert_eq!(config.sub_dimension(128), 32);
        assert!(config.validate(128).is_ok());
    }

    #[test]
    fn test_validation_errors() {
        assert!(IvfPqConfig::new(0, 1, 4).validate(128).is_err());
        assert!(IvfPqConfig::new(10, 0, 4).validate(128).is_err());
        assert!(IvfPqConfig::new(10, 11, 4).validate(128).is_err());
        assert!(IvfPqConfig::new(10, 5, 0).validate(128).is_err());
        assert!(IvfPqConfig::new(10, 5, 5).validate(128).is_err()); // 128 not divisible by 5
        assert!(
            IvfPqConfig::new(10, 5, 4)
                .with_sub_clusters(0)
                .validate(128)
                .is_err()
        );
        assert!(
            IvfPqConfig::new(10, 5, 4)
                .with_sub_clusters(300)
                .validate(128)
                .is_err()
        );
        assert!(
            IvfPqConfig::new(10, 5, 4)
                .with_max_kmeans_iters(0)
                .validate(128)
                .is_err()
        );
        assert!(
            IvfPqConfig::new(10, 5, 4)
                .with_kmeans_tolerance(-1.0)
                .validate(128)
                .is_err()
        );
    }

    #[test]
    fn test_config_traits() {
        let cfg1 = IvfPqConfig::new(16, 2, 4);
        let cfg2 = cfg1.clone();
        assert_eq!(cfg1, cfg2);
        let debug_str = format!("{:?}", cfg1);
        assert!(debug_str.contains("IvfPqConfig"));
        assert!(debug_str.contains("nlist: 16"));
    }

    #[test]
    fn test_sub_dimension_high_dim() {
        let config = IvfPqConfig::new(64, 8, 16);
        assert_eq!(config.sub_dimension(1536), 96);
        assert!(config.validate(1536).is_ok());
    }
}
