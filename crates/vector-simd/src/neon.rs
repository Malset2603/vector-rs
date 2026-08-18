#![allow(unsafe_op_in_unsafe_fn)]
/// ARM NEON accelerated vector distance computations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Computes the dot product of two f32 slices using ARM NEON intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;

        let a0 = vld1q_f32(a_ptr.add(base));
        let b0 = vld1q_f32(b_ptr.add(base));
        sum0 = vfmaq_f32(sum0, a0, b0);

        let a1 = vld1q_f32(a_ptr.add(base + 4));
        let b1 = vld1q_f32(b_ptr.add(base + 4));
        sum1 = vfmaq_f32(sum1, a1, b1);

        let a2 = vld1q_f32(a_ptr.add(base + 8));
        let b2 = vld1q_f32(b_ptr.add(base + 8));
        sum2 = vfmaq_f32(sum2, a2, b2);

        let a3 = vld1q_f32(a_ptr.add(base + 12));
        let b3 = vld1q_f32(b_ptr.add(base + 12));
        sum3 = vfmaq_f32(sum3, a3, b3);
    }

    let mut i = chunks * 16;
    while i + 4 <= n {
        let a_v = vld1q_f32(a_ptr.add(i));
        let b_v = vld1q_f32(b_ptr.add(i));
        sum0 = vfmaq_f32(sum0, a_v, b_v);
        i += 4;
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    let mut result = vaddvq_f32(sum0);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Computes the squared Euclidean (L2²) distance using ARM NEON intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;

        let diff0 = vsubq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        let diff1 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 4)),
            vld1q_f32(b_ptr.add(base + 4)),
        );
        sum1 = vfmaq_f32(sum1, diff1, diff1);

        let diff2 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 8)),
            vld1q_f32(b_ptr.add(base + 8)),
        );
        sum2 = vfmaq_f32(sum2, diff2, diff2);

        let diff3 = vsubq_f32(
            vld1q_f32(a_ptr.add(base + 12)),
            vld1q_f32(b_ptr.add(base + 12)),
        );
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    let mut i = chunks * 16;
    while i + 4 <= n {
        let diff = vsubq_f32(vld1q_f32(a_ptr.add(i)), vld1q_f32(b_ptr.add(i)));
        sum0 = vfmaq_f32(sum0, diff, diff);
        i += 4;
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    let mut result = vaddvq_f32(sum0);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

/// Computes cosine similarity using ARM NEON intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot = vdupq_n_f32(0.0);
    let mut na = vdupq_n_f32(0.0);
    let mut nb = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        let a_v = vld1q_f32(a_ptr.add(base));
        let b_v = vld1q_f32(b_ptr.add(base));
        dot = vfmaq_f32(dot, a_v, b_v);
        na = vfmaq_f32(na, a_v, a_v);
        nb = vfmaq_f32(nb, b_v, b_v);
    }

    let mut dot_result = vaddvq_f32(dot);
    let mut norm_a_result = vaddvq_f32(na);
    let mut norm_b_result = vaddvq_f32(nb);

    let mut i = chunks * 4;
    while i < n {
        dot_result += a[i] * b[i];
        norm_a_result += a[i] * a[i];
        norm_b_result += b[i] * b[i];
        i += 1;
    }

    let denominator = norm_a_result.sqrt() * norm_b_result.sqrt();
    assert!(
        denominator > 0.0,
        "cannot compute cosine similarity for zero-magnitude vector"
    );
    dot_result / denominator
}

/// Computes Manhattan (L1) distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::manhattan(a, b)
}

/// Computes Minkowski ($L_p$) distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
/// - `p > 0.0`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn minkowski(a: &[f32], b: &[f32], p: f32) -> f32 {
    crate::scalar::minkowski(a, b, p)
}

/// Computes Chebyshev (L-infinity) distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn chebyshev(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::chebyshev(a, b)
}

/// Computes thresholded Hamming distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hamming(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::hamming(a, b)
}

/// Computes Mahalanobis distance with covariance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length $d$.
/// - `inv_cov` has length $d$ or $d \times d$.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn mahalanobis_with_inv_cov(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    crate::scalar::mahalanobis_with_inv_cov(a, b, inv_cov)
}

/// Computes Standardized Mahalanobis distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn mahalanobis(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::mahalanobis(a, b)
}

/// Computes Generalized / Weighted Jaccard distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn jaccard(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::jaccard(a, b)
}

/// Computes Hellinger distance using ARM NEON / scalar fallback.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports ARM NEON instructions (`target_feature = "neon"`).
/// - `a` and `b` have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn hellinger(a: &[f32], b: &[f32]) -> f32 {
    crate::scalar::hellinger(a, b)
}
