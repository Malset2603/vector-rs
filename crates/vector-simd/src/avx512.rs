#![allow(unsafe_op_in_unsafe_fn)]
//! AVX-512 accelerated vector distance computations.
//!
//! Uses 512-bit SIMD registers to process 16 × f32 values per instruction cycle.
//! Employs 4-way loop unrolling with independent accumulators to maximize
//! Instruction-Level Parallelism (ILP).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// Computes the dot product of two f32 slices using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    let chunks = n / 64;
    for i in 0..chunks {
        let base = i * 64;

        let a0 = _mm512_loadu_ps(a_ptr.add(base));
        let b0 = _mm512_loadu_ps(b_ptr.add(base));
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        let a1 = _mm512_loadu_ps(a_ptr.add(base + 16));
        let b1 = _mm512_loadu_ps(b_ptr.add(base + 16));
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);

        let a2 = _mm512_loadu_ps(a_ptr.add(base + 32));
        let b2 = _mm512_loadu_ps(b_ptr.add(base + 32));
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);

        let a3 = _mm512_loadu_ps(a_ptr.add(base + 48));
        let b3 = _mm512_loadu_ps(b_ptr.add(base + 48));
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }

    let mut i = chunks * 64;
    while i + 16 <= n {
        let a_v = _mm512_loadu_ps(a_ptr.add(i));
        let b_v = _mm512_loadu_ps(b_ptr.add(i));
        sum0 = _mm512_fmadd_ps(a_v, b_v, sum0);
        i += 16;
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    let mut result = _mm512_reduce_add_ps(sum0);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Computes the squared Euclidean (L2²) distance using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    let chunks = n / 64;
    for i in 0..chunks {
        let base = i * 64;

        let diff0 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base)),
            _mm512_loadu_ps(b_ptr.add(base)),
        );
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        let diff1 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base + 16)),
            _mm512_loadu_ps(b_ptr.add(base + 16)),
        );
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        let diff2 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base + 32)),
            _mm512_loadu_ps(b_ptr.add(base + 32)),
        );
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

        let diff3 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base + 48)),
            _mm512_loadu_ps(b_ptr.add(base + 48)),
        );
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    let mut i = chunks * 64;
    while i + 16 <= n {
        let diff = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(i)), _mm512_loadu_ps(b_ptr.add(i)));
        sum0 = _mm512_fmadd_ps(diff, diff, sum0);
        i += 16;
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    let mut result = _mm512_reduce_add_ps(sum0);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

/// Computes cosine similarity using AVX-512 intrinsics.
///
/// Returns `0.0` gracefully if either vector has near-zero magnitude.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot0 = _mm512_setzero_ps();
    let mut dot1 = _mm512_setzero_ps();
    let mut na0 = _mm512_setzero_ps();
    let mut na1 = _mm512_setzero_ps();
    let mut nb0 = _mm512_setzero_ps();
    let mut nb1 = _mm512_setzero_ps();

    let chunks = n / 32;
    for i in 0..chunks {
        let base = i * 32;

        let a0 = _mm512_loadu_ps(a_ptr.add(base));
        let b0 = _mm512_loadu_ps(b_ptr.add(base));
        dot0 = _mm512_fmadd_ps(a0, b0, dot0);
        na0 = _mm512_fmadd_ps(a0, a0, na0);
        nb0 = _mm512_fmadd_ps(b0, b0, nb0);

        let a1 = _mm512_loadu_ps(a_ptr.add(base + 16));
        let b1 = _mm512_loadu_ps(b_ptr.add(base + 16));
        dot1 = _mm512_fmadd_ps(a1, b1, dot1);
        na1 = _mm512_fmadd_ps(a1, a1, na1);
        nb1 = _mm512_fmadd_ps(b1, b1, nb1);
    }

    let mut i = chunks * 32;
    while i + 16 <= n {
        let a_v = _mm512_loadu_ps(a_ptr.add(i));
        let b_v = _mm512_loadu_ps(b_ptr.add(i));
        dot0 = _mm512_fmadd_ps(a_v, b_v, dot0);
        na0 = _mm512_fmadd_ps(a_v, a_v, na0);
        nb0 = _mm512_fmadd_ps(b_v, b_v, nb0);
        i += 16;
    }

    dot0 = _mm512_add_ps(dot0, dot1);
    na0 = _mm512_add_ps(na0, na1);
    nb0 = _mm512_add_ps(nb0, nb1);

    let mut dot_result = _mm512_reduce_add_ps(dot0);
    let mut norm_a_result = _mm512_reduce_add_ps(na0);
    let mut norm_b_result = _mm512_reduce_add_ps(nb0);

    while i < n {
        dot_result += a[i] * b[i];
        norm_a_result += a[i] * a[i];
        norm_b_result += b[i] * b[i];
        i += 1;
    }

    let denominator = norm_a_result.sqrt() * norm_b_result.sqrt();
    if denominator <= 1e-9 {
        0.0
    } else {
        dot_result / denominator
    }
}

/// Computes Manhattan (L1) distance using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();

    let chunks = n / 32;
    for i in 0..chunks {
        let base = i * 32;
        let diff0 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base)),
            _mm512_loadu_ps(b_ptr.add(base)),
        );
        let abs0 = _mm512_and_ps(diff0, sign_mask);
        sum0 = _mm512_add_ps(sum0, abs0);

        let diff1 = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base + 16)),
            _mm512_loadu_ps(b_ptr.add(base + 16)),
        );
        let abs1 = _mm512_and_ps(diff1, sign_mask);
        sum1 = _mm512_add_ps(sum1, abs1);
    }

    let mut i = chunks * 32;
    while i + 16 <= n {
        let diff = _mm512_sub_ps(_mm512_loadu_ps(a_ptr.add(i)), _mm512_loadu_ps(b_ptr.add(i)));
        let abs_val = _mm512_and_ps(diff, sign_mask);
        sum0 = _mm512_add_ps(sum0, abs_val);
        i += 16;
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    let mut result = _mm512_reduce_add_ps(sum0);

    while i < n {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Computes Minkowski ($L_p, p=3$) distance using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
/// - `p > 0.0`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn minkowski(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    assert!(p > 0.0, "p must be greater than 0");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    if (p - 3.0).abs() < 1e-5 {
        let mut sum_vec0 = _mm512_setzero_ps();
        let mut sum_vec1 = _mm512_setzero_ps();

        let chunks = n / 32;
        for i in 0..chunks {
            let base = i * 32;
            let diff0 = _mm512_and_ps(
                _mm512_sub_ps(
                    _mm512_loadu_ps(a_ptr.add(base)),
                    _mm512_loadu_ps(b_ptr.add(base)),
                ),
                sign_mask,
            );
            let diff0_sq = _mm512_mul_ps(diff0, diff0);
            sum_vec0 = _mm512_fmadd_ps(diff0_sq, diff0, sum_vec0);

            let diff1 = _mm512_and_ps(
                _mm512_sub_ps(
                    _mm512_loadu_ps(a_ptr.add(base + 16)),
                    _mm512_loadu_ps(b_ptr.add(base + 16)),
                ),
                sign_mask,
            );
            let diff1_sq = _mm512_mul_ps(diff1, diff1);
            sum_vec1 = _mm512_fmadd_ps(diff1_sq, diff1, sum_vec1);
        }

        sum_vec0 = _mm512_add_ps(sum_vec0, sum_vec1);
        let mut sum = _mm512_reduce_add_ps(sum_vec0);

        let mut i = chunks * 32;
        while i < n {
            let diff = (a[i] - b[i]).abs();
            sum += diff * diff * diff;
            i += 1;
        }

        sum.cbrt()
    } else {
        crate::scalar::minkowski(a, b, p)
    }
}

/// Computes Chebyshev (L-infinity) distance using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn chebyshev(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    if a.is_empty() {
        return 0.0;
    }

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    let mut max_vec = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        let diff = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base)),
            _mm512_loadu_ps(b_ptr.add(base)),
        );
        let abs_val = _mm512_and_ps(diff, sign_mask);
        max_vec = _mm512_max_ps(max_vec, abs_val);
    }

    let mut max_val = _mm512_reduce_max_ps(max_vec);

    let mut i = chunks * 16;
    while i < n {
        let diff = (a[i] - b[i]).abs();
        if diff > max_val {
            max_val = diff;
        }
        i += 1;
    }

    max_val
}

/// Computes thresholded Hamming distance using AVX-512 intrinsics.
///
/// Counts coordinate positions where $|a_i - b_i| > 10^{-6}$.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn hamming(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    let eps_vec = _mm512_set1_ps(1e-6);

    let mut count = 0.0_f32;

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        let diff = _mm512_sub_ps(
            _mm512_loadu_ps(a_ptr.add(base)),
            _mm512_loadu_ps(b_ptr.add(base)),
        );
        let abs_diff = _mm512_and_ps(diff, sign_mask);
        let mask = _mm512_cmp_ps_mask(abs_diff, eps_vec, _CMP_GT_OQ);
        count += mask.count_ones() as f32;
    }

    let mut i = chunks * 16;
    while i < n {
        if (a[i] - b[i]).abs() > 1e-6 {
            count += 1.0;
        }
        i += 1;
    }

    count
}

/// Computes Mahalanobis distance with precision matrix or diagonal inverse variances using AVX-512 intrinsics.
///
/// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length $d$.
/// - `inv_cov` has length $d$ (diagonal) or $d \times d$ (full matrix).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mahalanobis_with_inv_cov(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    let d = a.len();
    assert_eq!(d, b.len(), "vector dimensions must match");

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let inv_ptr = inv_cov.as_ptr();

    if inv_cov.len() == d {
        // Diagonal inverse variance: \sum w_i * (a_i - b_i)^2
        let mut sum_vec0 = _mm512_setzero_ps();
        let mut sum_vec1 = _mm512_setzero_ps();

        let chunks = d / 32;
        for i in 0..chunks {
            let base = i * 32;
            let diff0 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base)),
                _mm512_loadu_ps(b_ptr.add(base)),
            );
            let w0 = _mm512_loadu_ps(inv_ptr.add(base));
            let diff0_sq = _mm512_mul_ps(diff0, diff0);
            sum_vec0 = _mm512_fmadd_ps(diff0_sq, w0, sum_vec0);

            let diff1 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base + 16)),
                _mm512_loadu_ps(b_ptr.add(base + 16)),
            );
            let w1 = _mm512_loadu_ps(inv_ptr.add(base + 16));
            let diff1_sq = _mm512_mul_ps(diff1, diff1);
            sum_vec1 = _mm512_fmadd_ps(diff1_sq, w1, sum_vec1);
        }

        sum_vec0 = _mm512_add_ps(sum_vec0, sum_vec1);
        let mut sum = _mm512_reduce_add_ps(sum_vec0);

        let mut i = chunks * 32;
        while i < d {
            let diff = a[i] - b[i];
            sum += inv_cov[i] * diff * diff;
            i += 1;
        }

        sum.max(0.0).sqrt()
    } else if inv_cov.len() == d * d {
        crate::scalar::mahalanobis_with_inv_cov(a, b, inv_cov)
    } else {
        panic!(
            "invalid inv_cov dimension: expected {} or {}, got {}",
            d,
            d * d,
            inv_cov.len()
        );
    }
}

/// Computes Standardized Mahalanobis distance assuming unit diagonal covariance ($\Sigma = I$).
///
/// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn mahalanobis(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Computes Generalized / Weighted Jaccard (Tanimoto) distance using AVX-512 intrinsics.
///
/// $$J_D(\mathbf{u}, \mathbf{v}) = 1.0 - \frac{\sum \min(|u_i|, |v_i|)}{\sum \max(|u_i|, |v_i|) + \epsilon}$$
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn jaccard(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    let mut min_vec = _mm512_setzero_ps();
    let mut max_vec = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        let abs_a = _mm512_and_ps(_mm512_loadu_ps(a_ptr.add(base)), sign_mask);
        let abs_b = _mm512_and_ps(_mm512_loadu_ps(b_ptr.add(base)), sign_mask);
        min_vec = _mm512_add_ps(min_vec, _mm512_min_ps(abs_a, abs_b));
        max_vec = _mm512_add_ps(max_vec, _mm512_max_ps(abs_a, abs_b));
    }

    let sum_min = _mm512_reduce_add_ps(min_vec);
    let mut sum_max = _mm512_reduce_add_ps(max_vec);

    let mut i = chunks * 16;
    let mut remainder_min = 0.0_f32;
    while i < n {
        let abs_a = a[i].abs();
        let abs_b = b[i].abs();
        remainder_min += abs_a.min(abs_b);
        sum_max += abs_a.max(abs_b);
        i += 1;
    }

    let total_min = sum_min + remainder_min;
    if sum_max <= 1e-9 {
        0.0
    } else {
        1.0 - (total_min / sum_max)
    }
}

/// Computes Hellinger distance using AVX-512 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX-512F instructions (`target_feature = "avx512f"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn hellinger(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    let mut sum_vec = _mm512_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        let abs_a = _mm512_and_ps(_mm512_loadu_ps(a_ptr.add(base)), sign_mask);
        let abs_b = _mm512_and_ps(_mm512_loadu_ps(b_ptr.add(base)), sign_mask);
        let diff = _mm512_sub_ps(_mm512_sqrt_ps(abs_a), _mm512_sqrt_ps(abs_b));
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    let mut sum = _mm512_reduce_add_ps(sum_vec);

    let mut i = chunks * 16;
    while i < n {
        let diff = a[i].abs().sqrt() - b[i].abs().sqrt();
        sum += diff * diff;
        i += 1;
    }

    (0.5 * sum).sqrt()
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn has_avx512f() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[test]
    fn test_avx512_metrics_match_scalar() {
        if !has_avx512f() {
            return;
        }
        for dim in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 768] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1).collect();
            let b: Vec<f32> = (0..dim)
                .map(|i| ((i * 13 + 5) % 100) as f32 * 0.1)
                .collect();

            let dot_simd = unsafe { dot_product(&a, &b) };
            let dot_scalar = crate::scalar::dot_product(&a, &b);
            let rel_dot = (dot_simd - dot_scalar).abs() / dot_scalar.abs().max(1.0);
            assert!(rel_dot < 1e-3);

            let l2_simd = unsafe { l2_squared(&a, &b) };
            let l2_scalar = crate::scalar::l2_squared(&a, &b);
            let rel_l2 = (l2_simd - l2_scalar).abs() / l2_scalar.max(1.0);
            assert!(rel_l2 < 1e-3);

            let man_simd = unsafe { manhattan(&a, &b) };
            let man_scalar = crate::scalar::manhattan(&a, &b);
            let rel_man = (man_simd - man_scalar).abs() / man_scalar.max(1.0);
            assert!(rel_man < 1e-3);

            let mink_simd = unsafe { minkowski(&a, &b, 3.0) };
            let mink_scalar = crate::scalar::minkowski(&a, &b, 3.0);
            let rel_mink = (mink_simd - mink_scalar).abs() / mink_scalar.max(1.0);
            assert!(rel_mink < 1e-3);

            let cheb_simd = unsafe { chebyshev(&a, &b) };
            let cheb_scalar = crate::scalar::chebyshev(&a, &b);
            assert!((cheb_simd - cheb_scalar).abs() < EPSILON);

            let ham_simd = unsafe { hamming(&a, &b) };
            let ham_scalar = crate::scalar::hamming(&a, &b);
            assert!((ham_simd - ham_scalar).abs() < EPSILON);

            let jacc_simd = unsafe { jaccard(&a, &b) };
            let jacc_scalar = crate::scalar::jaccard(&a, &b);
            assert!((jacc_simd - jacc_scalar).abs() < EPSILON);

            let hell_simd = unsafe { hellinger(&a, &b) };
            let hell_scalar = crate::scalar::hellinger(&a, &b);
            let rel_hell = (hell_simd - hell_scalar).abs() / hell_scalar.max(1.0);
            assert!(rel_hell < 1e-3);

            let inv_diag: Vec<f32> = (0..dim).map(|i| (i % 5 + 1) as f32 * 0.25).collect();
            let mah_simd = unsafe { mahalanobis_with_inv_cov(&a, &b, &inv_diag) };
            let mah_scalar = crate::scalar::mahalanobis_with_inv_cov(&a, &b, &inv_diag);
            let rel_mah = (mah_simd - mah_scalar).abs() / mah_scalar.max(1.0);
            assert!(rel_mah < 1e-3);
        }
    }
}
