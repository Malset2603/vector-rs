//! Configuration parameters for Hierarchical Navigable Small World (HNSW) indexing.

use crate::types::{Result, VectorIndexError};

/// Default maximum number of bidirectional connections for each node on layers $l > 0$.
pub const DEFAULT_M: usize = 16;

/// Default maximum number of bidirectional connections on layer 0 ($2 \times M$).
pub const DEFAULT_M0: usize = 32;

/// Default size of the dynamic candidate list during graph construction.
pub const DEFAULT_EF_CONSTRUCTION: usize = 100;

/// Default size of the dynamic candidate list during nearest-neighbor search.
pub const DEFAULT_EF_SEARCH: usize = 50;

/// Configuration options for the HNSW index.
///
/// Controls trade-offs between index construction speed, search latency,
/// memory consumption, and recall accuracy.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswConfig {
    /// Maximum number of bidirectional links per node on layers $> 0$.
    pub m: usize,
    /// Maximum number of bidirectional links per node on layer $0$.
    /// Defaults to $2 \times M$.
    pub m0: usize,
    /// Size of dynamic candidate list during index construction ($ef_{construction}$).
    pub ef_construction: usize,
    /// Default size of dynamic candidate list during search ($ef_{search}$).
    pub ef_search: usize,
    /// Level normalization factor $m_l = \frac{1}{\ln(M)}$.
    pub ml: f64,
    /// Whether to use the diversity-preserving neighbor selection heuristic (Algorithm 4 in HNSW paper).
    pub use_heuristic: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH)
    }
}

impl HnswConfig {
    /// Creates a new `HnswConfig` with the given parameters and calculated defaults.
    ///
    /// `m0` is set to $2 \times M$, and $m_l$ is set to $\frac{1}{\ln(M)}$.
    ///
    /// # Panics
    ///
    /// Panics if `m < 2`, `ef_construction == 0`, or `ef_search == 0`.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        assert!(m >= 2, "m must be at least 2");
        assert!(ef_construction > 0, "ef_construction must be > 0");
        assert!(ef_search > 0, "ef_search must be > 0");

        let m0 = 2 * m;
        let ml = 1.0 / (m as f64).ln();

        Self {
            m,
            m0,
            ef_construction,
            ef_search,
            ml,
            use_heuristic: true,
        }
    }

    /// Sets the maximum number of links per node on layer 0 (`m0`).
    pub fn with_m0(mut self, m0: usize) -> Self {
        assert!(m0 >= self.m, "m0 should be >= m");
        self.m0 = m0;
        self
    }

    /// Sets $ef_{construction}$.
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        assert!(ef_construction > 0, "ef_construction must be > 0");
        self.ef_construction = ef_construction;
        self
    }

    /// Sets $ef_{search}$.
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        assert!(ef_search > 0, "ef_search must be > 0");
        self.ef_search = ef_search;
        self
    }

    /// Enables or disables the diversity-preserving neighbor selection heuristic.
    pub fn with_heuristic(mut self, use_heuristic: bool) -> Self {
        self.use_heuristic = use_heuristic;
        self
    }

    /// Sets the level generation multiplier $m_l$.
    pub fn with_ml(mut self, ml: f64) -> Self {
        assert!(ml > 0.0, "ml must be > 0.0");
        self.ml = ml;
        self
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return Err(VectorIndexError::InvalidHeader {
                path: std::path::PathBuf::from("config"),
                reason: format!("m must be >= 2, got {}", self.m),
            });
        }
        if self.m0 < self.m {
            return Err(VectorIndexError::InvalidHeader {
                path: std::path::PathBuf::from("config"),
                reason: format!("m0 ({}) must be >= m ({})", self.m0, self.m),
            });
        }
        if self.ef_construction == 0 {
            return Err(VectorIndexError::InvalidHeader {
                path: std::path::PathBuf::from("config"),
                reason: "ef_construction must be > 0".to_string(),
            });
        }
        if self.ef_search == 0 {
            return Err(VectorIndexError::InvalidHeader {
                path: std::path::PathBuf::from("config"),
                reason: "ef_search must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HnswConfig::default();
        assert_eq!(config.m, DEFAULT_M);
        assert_eq!(config.m0, DEFAULT_M0);
        assert_eq!(config.ef_construction, DEFAULT_EF_CONSTRUCTION);
        assert_eq!(config.ef_search, DEFAULT_EF_SEARCH);
        assert!(config.use_heuristic);
        assert!((config.ml - (1.0 / (DEFAULT_M as f64).ln())).abs() < 1e-6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_custom_config() {
        let config = HnswConfig::new(32, 200, 100)
            .with_m0(64)
            .with_heuristic(false);

        assert_eq!(config.m, 32);
        assert_eq!(config.m0, 64);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 100);
        assert!(!config.use_heuristic);
        assert!(config.validate().is_ok());
    }

    #[test]
    #[should_panic(expected = "m must be at least 2")]
    fn test_invalid_m() {
        HnswConfig::new(1, 100, 50);
    }

    #[test]
    #[should_panic(expected = "ef_construction must be > 0")]
    fn test_zero_ef_construction() {
        HnswConfig::new(16, 0, 50);
    }

    #[test]
    #[should_panic(expected = "ef_search must be > 0")]
    fn test_zero_ef_search() {
        HnswConfig::new(16, 100, 0);
    }

    #[test]
    #[should_panic(expected = "m0 should be >= m")]
    fn test_invalid_m0_panic() {
        HnswConfig::new(16, 100, 50).with_m0(10);
    }

    #[test]
    #[should_panic(expected = "ml must be > 0.0")]
    fn test_invalid_ml_panic() {
        HnswConfig::new(16, 100, 50).with_ml(0.0);
    }

    #[test]
    fn test_config_builder_setters() {
        let config = HnswConfig::default()
            .with_ef_construction(150)
            .with_ef_search(75)
            .with_ml(0.5)
            .with_heuristic(false);

        assert_eq!(config.ef_construction, 150);
        assert_eq!(config.ef_search, 75);
        assert_eq!(config.ml, 0.5);
        assert!(!config.use_heuristic);
    }

    #[test]
    fn test_validate_method_error_cases() {
        let mut bad_config = HnswConfig::default();
        bad_config.m = 1;
        assert!(bad_config.validate().is_err());

        let mut bad_config2 = HnswConfig::default();
        bad_config2.m0 = 5;
        assert!(bad_config2.validate().is_err());

        let mut bad_config3 = HnswConfig::default();
        bad_config3.ef_construction = 0;
        assert!(bad_config3.validate().is_err());

        let mut bad_config4 = HnswConfig::default();
        bad_config4.ef_search = 0;
        assert!(bad_config4.validate().is_err());
    }
}
