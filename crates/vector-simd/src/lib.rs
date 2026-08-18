//! # vector-simd
//!
//! High-performance SIMD-accelerated vector distance computations for the VectorRS engine.
//!
//! Provides hardware-accelerated distance and similarity metrics:
//! - **Dot Product** — inner product of two vectors
//! - **L2 Squared** — squared Euclidean distance
//! - **Cosine Similarity** — normalized directional cosine metric
//! - **Manhattan** — L1 absolute coordinate distance
//! - **Minkowski** — Lp distance
//! - **Chebyshev** — L-infinity maximum coordinate distance
//! - **Hamming** — coordinate mismatch distance
//! - **Mahalanobis** — standardized covariance distance
//! - **Jaccard** — generalized continuous Tanimoto distance
//! - **Hellinger** — probability distribution distance

pub mod scalar;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx2;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;

/// Identifies which SIMD backend is active for distance computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// x86/x86_64 AVX-512F (512-bit, 16 × f32 per instruction)
    Avx512,
    /// x86/x86_64 AVX2 + FMA (256-bit, 8 × f32 per instruction)
    Avx2,
    /// AArch64 NEON (128-bit, 4 × f32 per instruction)
    Neon,
    /// Portable scalar fallback (1 × f32 per operation)
    Scalar,
}

impl std::fmt::Display for SimdBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimdBackend::Avx512 => write!(f, "AVX-512F"),
            SimdBackend::Avx2 => write!(f, "AVX2+FMA"),
            SimdBackend::Neon => write!(f, "NEON"),
            SimdBackend::Scalar => write!(f, "Scalar"),
        }
    }
}

/// Runtime-dispatched vector distance computation engine.
#[derive(Debug, Clone, Copy)]
pub struct DistanceEngine {
    backend: SimdBackend,
}

impl DistanceEngine {
    /// Creates a new `DistanceEngine` that automatically detects the best available SIMD backend.
    pub fn auto() -> Self {
        Self {
            backend: detect_best_backend(),
        }
    }

    /// Creates a `DistanceEngine` that always uses the scalar fallback.
    pub fn scalar() -> Self {
        Self {
            backend: SimdBackend::Scalar,
        }
    }

    /// Creates a `DistanceEngine` with an explicitly specified backend.
    pub fn with_backend(backend: SimdBackend) -> Self {
        match backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => {
                assert!(
                    is_x86_feature_detected!("avx512f"),
                    "AVX-512F is not supported on this CPU"
                );
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => {
                assert!(
                    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
                    "AVX2+FMA is not supported on this CPU"
                );
            }
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => {}
            #[cfg(not(target_arch = "aarch64"))]
            SimdBackend::Neon => {
                panic!("NEON is only supported on AArch64");
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            SimdBackend::Avx512 | SimdBackend::Avx2 => {
                panic!("AVX is only supported on x86/x86_64");
            }
            SimdBackend::Scalar => {}
        }
        Self { backend }
    }

    /// Returns which SIMD backend this engine is using.
    pub fn backend(&self) -> SimdBackend {
        self.backend
    }

    /// Computes the dot product of two f32 slices.
    #[inline]
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::dot_product(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::dot_product(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::dot_product(a, b) },
            SimdBackend::Scalar => scalar::dot_product(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::dot_product(a, b),
        }
    }

    /// Computes the squared Euclidean (L2²) distance between two f32 slices.
    #[inline]
    pub fn l2_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::l2_squared(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::l2_squared(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::l2_squared(a, b) },
            SimdBackend::Scalar => scalar::l2_squared(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::l2_squared(a, b),
        }
    }

    /// Computes cosine similarity between two f32 slices.
    #[inline]
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::cosine_similarity(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::cosine_similarity(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::cosine_similarity(a, b) },
            SimdBackend::Scalar => scalar::cosine_similarity(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::cosine_similarity(a, b),
        }
    }

    /// Computes the Manhattan (L1) distance between two f32 slices.
    #[inline]
    pub fn manhattan(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::manhattan(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::manhattan(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::manhattan(a, b) },
            SimdBackend::Scalar => scalar::manhattan(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::manhattan(a, b),
        }
    }

    /// Computes the Minkowski (Lp) distance between two f32 slices.
    #[inline]
    pub fn minkowski(&self, a: &[f32], b: &[f32], p: f32) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::minkowski(a, b, p) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::minkowski(a, b, p) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::minkowski(a, b, p) },
            SimdBackend::Scalar => scalar::minkowski(a, b, p),
            #[allow(unreachable_patterns)]
            _ => scalar::minkowski(a, b, p),
        }
    }

    /// Computes the Chebyshev (L-infinity) distance between two f32 slices.
    #[inline]
    pub fn chebyshev(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::chebyshev(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::chebyshev(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::chebyshev(a, b) },
            SimdBackend::Scalar => scalar::chebyshev(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::chebyshev(a, b),
        }
    }

    /// Computes the Hamming distance between two f32 slices.
    #[inline]
    pub fn hamming(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::hamming(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::hamming(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::hamming(a, b) },
            SimdBackend::Scalar => scalar::hamming(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::hamming(a, b),
        }
    }

    /// Computes the Mahalanobis distance between two f32 slices using a precision matrix or diagonal inverse variance weights.
    ///
    /// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
    #[inline]
    pub fn mahalanobis_with_inv_cov(&self, a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::mahalanobis_with_inv_cov(a, b, inv_cov) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::mahalanobis_with_inv_cov(a, b, inv_cov) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::mahalanobis_with_inv_cov(a, b, inv_cov) },
            SimdBackend::Scalar => scalar::mahalanobis_with_inv_cov(a, b, inv_cov),
            #[allow(unreachable_patterns)]
            _ => scalar::mahalanobis_with_inv_cov(a, b, inv_cov),
        }
    }

    /// Computes the Standardized Mahalanobis distance between two f32 slices.
    ///
    /// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
    #[inline]
    pub fn mahalanobis(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::mahalanobis(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::mahalanobis(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::mahalanobis(a, b) },
            SimdBackend::Scalar => scalar::mahalanobis(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::mahalanobis(a, b),
        }
    }

    /// Computes the Generalized Jaccard distance between two f32 slices.
    #[inline]
    pub fn jaccard(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::jaccard(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::jaccard(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::jaccard(a, b) },
            SimdBackend::Scalar => scalar::jaccard(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::jaccard(a, b),
        }
    }

    /// Computes the Hellinger distance between two f32 slices.
    #[inline]
    pub fn hellinger(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.backend {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx512 => unsafe { avx512::hellinger(a, b) },
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Avx2 => unsafe { avx2::hellinger(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { neon::hellinger(a, b) },
            SimdBackend::Scalar => scalar::hellinger(a, b),
            #[allow(unreachable_patterns)]
            _ => scalar::hellinger(a, b),
        }
    }
}

/// Detects the best available SIMD backend on the current CPU at runtime.
fn detect_best_backend() -> SimdBackend {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdBackend::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdBackend::Avx2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return SimdBackend::Neon;
    }

    #[allow(unreachable_code)]
    SimdBackend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    #[test]
    fn test_auto_detection() {
        let engine = DistanceEngine::auto();
        let backend = engine.backend();
        println!("Detected SIMD backend: {backend}");
        assert!(matches!(
            backend,
            SimdBackend::Avx512 | SimdBackend::Avx2 | SimdBackend::Neon | SimdBackend::Scalar
        ));
    }

    #[test]
    fn test_scalar_engine() {
        let engine = DistanceEngine::scalar();
        assert_eq!(engine.backend(), SimdBackend::Scalar);
    }

    #[test]
    fn test_engine_dot_product() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = engine.dot_product(&a, &b);
        let expected = scalar::dot_product(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_l2_squared() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = engine.l2_squared(&a, &b);
        let expected = scalar::l2_squared(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_cosine_similarity() {
        let engine = DistanceEngine::auto();
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=32).map(|i| (i as f32) * 2.0).collect();
        let result = engine.cosine_similarity(&a, &b);
        let expected = scalar::cosine_similarity(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_manhattan() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let result = engine.manhattan(&a, &b);
        let expected = scalar::manhattan(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_chebyshev() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 10.0, 3.0, 4.0];
        let b = [4.0, 2.0, 2.0, 1.0];
        let result = engine.chebyshev(&a, &b);
        let expected = scalar::chebyshev(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_jaccard() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 2.0, 0.0, 4.0];
        let b = [1.0, 0.0, 0.0, 2.0];
        let result = engine.jaccard(&a, &b);
        let expected = scalar::jaccard(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_hellinger() {
        let engine = DistanceEngine::auto();
        let a = [0.5, 0.5];
        let b = [0.5, 0.5];
        let result = engine.hellinger(&a, &b);
        let expected = scalar::hellinger(&a, &b);
        assert!((result - expected).abs() < EPSILON);
    }

    #[test]
    fn test_engine_mahalanobis() {
        let engine = DistanceEngine::auto();
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        let result = engine.mahalanobis(&a, &b);
        let expected = scalar::mahalanobis(&a, &b);
        assert!((result - expected).abs() < EPSILON);

        let inv_cov = [2.0, 0.5];
        let res_cov = engine.mahalanobis_with_inv_cov(&a, &b, &inv_cov);
        let exp_cov = scalar::mahalanobis_with_inv_cov(&a, &b, &inv_cov);
        assert!((res_cov - exp_cov).abs() < EPSILON);
    }

    #[test]
    fn test_engine_high_dimensional() {
        let engine = DistanceEngine::auto();
        let a: Vec<f32> = (0..1536)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
            .collect();
        let b: Vec<f32> = (0..1536)
            .map(|i| ((i * 13 + 5) % 100) as f32 * 0.01)
            .collect();

        let dot = engine.dot_product(&a, &b);
        let expected_dot = scalar::dot_product(&a, &b);
        let rel = (dot - expected_dot).abs() / expected_dot.abs();
        assert!(rel < 1e-3, "dot: {dot} vs {expected_dot}, rel={rel}");

        let man = engine.manhattan(&a, &b);
        let expected_man = scalar::manhattan(&a, &b);
        assert!((man - expected_man).abs() / expected_man < 1e-3);
    }
}
