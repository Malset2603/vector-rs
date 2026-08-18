#![allow(unsafe_op_in_unsafe_fn)]
//! AVX2 + FMA accelerated vector distance computations.
//!
//! Uses 256-bit SIMD registers to process 8 × f32 values per instruction cycle.
//! Employs loop unrolling with independent accumulators to maximize
//! Instruction-Level Parallelism (ILP).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// Computes the dot product of two f32 slices using AVX2 + FMA intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = n / 32;
    for i in 0..chunks {
        let base = i * 32;

        let a0 = _mm256_loadu_ps(a_ptr.add(base));
        let b0 = _mm256_loadu_ps(b_ptr.add(base));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_loadu_ps(a_ptr.add(base + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(base + 8));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_loadu_ps(a_ptr.add(base + 16));
        let b2 = _mm256_loadu_ps(b_ptr.add(base + 16));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_loadu_ps(a_ptr.add(base + 24));
        let b3 = _mm256_loadu_ps(b_ptr.add(base + 24));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }

    let mut i = chunks * 32;
    while i + 8 <= n {
        let a_v = _mm256_loadu_ps(a_ptr.add(i));
        let b_v = _mm256_loadu_ps(b_ptr.add(i));
        sum0 = _mm256_fmadd_ps(a_v, b_v, sum0);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    let mut result = hsum256_ps(sum0);

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Computes the squared Euclidean (L2²) distance using AVX2 + FMA intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = n / 32;
    for i in 0..chunks {
        let base = i * 32;

        let diff0 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
        );
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let diff1 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base + 8)),
            _mm256_loadu_ps(b_ptr.add(base + 8)),
        );
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let diff2 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base + 16)),
            _mm256_loadu_ps(b_ptr.add(base + 16)),
        );
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let diff3 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base + 24)),
            _mm256_loadu_ps(b_ptr.add(base + 24)),
        );
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    let mut i = chunks * 32;
    while i + 8 <= n {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(a_ptr.add(i)), _mm256_loadu_ps(b_ptr.add(i)));
        sum0 = _mm256_fmadd_ps(diff, diff, sum0);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    let mut result = hsum256_ps(sum0);

    while i < n {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result
}

/// Computes cosine similarity using AVX2 + FMA intrinsics.
///
/// Returns `0.0` gracefully if either vector has near-zero magnitude.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut na0 = _mm256_setzero_ps();
    let mut na1 = _mm256_setzero_ps();
    let mut nb0 = _mm256_setzero_ps();
    let mut nb1 = _mm256_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;

        let a0 = _mm256_loadu_ps(a_ptr.add(base));
        let b0 = _mm256_loadu_ps(b_ptr.add(base));
        dot0 = _mm256_fmadd_ps(a0, b0, dot0);
        na0 = _mm256_fmadd_ps(a0, a0, na0);
        nb0 = _mm256_fmadd_ps(b0, b0, nb0);

        let a1 = _mm256_loadu_ps(a_ptr.add(base + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(base + 8));
        dot1 = _mm256_fmadd_ps(a1, b1, dot1);
        na1 = _mm256_fmadd_ps(a1, a1, na1);
        nb1 = _mm256_fmadd_ps(b1, b1, nb1);
    }

    let mut i = chunks * 16;
    while i + 8 <= n {
        let a_v = _mm256_loadu_ps(a_ptr.add(i));
        let b_v = _mm256_loadu_ps(b_ptr.add(i));
        dot0 = _mm256_fmadd_ps(a_v, b_v, dot0);
        na0 = _mm256_fmadd_ps(a_v, a_v, na0);
        nb0 = _mm256_fmadd_ps(b_v, b_v, nb0);
        i += 8;
    }

    dot0 = _mm256_add_ps(dot0, dot1);
    na0 = _mm256_add_ps(na0, na1);
    nb0 = _mm256_add_ps(nb0, nb1);

    let mut dot_result = hsum256_ps(dot0);
    let mut norm_a_result = hsum256_ps(na0);
    let mut norm_b_result = hsum256_ps(nb0);

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

/// Computes Manhattan (L1) distance using AVX2 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 instructions (`target_feature = "avx2"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        let diff0 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
        );
        let abs0 = _mm256_and_ps(diff0, sign_mask);
        sum0 = _mm256_add_ps(sum0, abs0);

        let diff1 = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base + 8)),
            _mm256_loadu_ps(b_ptr.add(base + 8)),
        );
        let abs1 = _mm256_and_ps(diff1, sign_mask);
        sum1 = _mm256_add_ps(sum1, abs1);
    }

    let mut i = chunks * 16;
    while i + 8 <= n {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(a_ptr.add(i)), _mm256_loadu_ps(b_ptr.add(i)));
        let abs_val = _mm256_and_ps(diff, sign_mask);
        sum0 = _mm256_add_ps(sum0, abs_val);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    let mut result = hsum256_ps(sum0);

    while i < n {
        result += (a[i] - b[i]).abs();
        i += 1;
    }

    result
}

/// Computes Minkowski ($L_p, p=3$) distance using AVX2 + FMA intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length.
/// - `p > 0.0`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn minkowski(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    assert!(p > 0.0, "p must be greater than 0");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    if (p - 3.0).abs() < 1e-5 {
        let mut sum_vec0 = _mm256_setzero_ps();
        let mut sum_vec1 = _mm256_setzero_ps();

        let chunks = n / 16;
        for i in 0..chunks {
            let base = i * 16;
            let diff0 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(base)),
                    _mm256_loadu_ps(b_ptr.add(base)),
                ),
                sign_mask,
            );
            let diff0_sq = _mm256_mul_ps(diff0, diff0);
            sum_vec0 = _mm256_fmadd_ps(diff0_sq, diff0, sum_vec0);

            let diff1 = _mm256_and_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(a_ptr.add(base + 8)),
                    _mm256_loadu_ps(b_ptr.add(base + 8)),
                ),
                sign_mask,
            );
            let diff1_sq = _mm256_mul_ps(diff1, diff1);
            sum_vec1 = _mm256_fmadd_ps(diff1_sq, diff1, sum_vec1);
        }

        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        let mut sum = hsum256_ps(sum_vec0);

        let mut i = chunks * 16;
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

/// Computes Chebyshev (L-infinity) distance using AVX2 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 instructions (`target_feature = "avx2"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn chebyshev(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    if a.is_empty() {
        return 0.0;
    }

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    let mut max_vec = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
        );
        let abs_val = _mm256_and_ps(diff, sign_mask);
        max_vec = _mm256_max_ps(max_vec, abs_val);
    }

    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), max_vec);
    let mut max_val = arr.iter().copied().fold(0.0f32, f32::max);

    let mut i = chunks * 8;
    while i < n {
        let diff = (a[i] - b[i]).abs();
        if diff > max_val {
            max_val = diff;
        }
        i += 1;
    }

    max_val
}

/// Computes thresholded Hamming distance using AVX2 intrinsics.
///
/// Counts coordinate positions where $|a_i - b_i| > 10^{-6}$.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 instructions (`target_feature = "avx2"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn hamming(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    let eps_vec = _mm256_set1_ps(1e-6);
    let one_vec = _mm256_set1_ps(1.0);

    let mut count_vec = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let diff = _mm256_sub_ps(
            _mm256_loadu_ps(a_ptr.add(base)),
            _mm256_loadu_ps(b_ptr.add(base)),
        );
        let abs_diff = _mm256_and_ps(diff, sign_mask);
        let mask = _mm256_cmp_ps(abs_diff, eps_vec, _CMP_GT_OQ);
        let count_inc = _mm256_and_ps(mask, one_vec);
        count_vec = _mm256_add_ps(count_vec, count_inc);
    }

    let mut count = hsum256_ps(count_vec);

    let mut i = chunks * 8;
    while i < n {
        if (a[i] - b[i]).abs() > 1e-6 {
            count += 1.0;
        }
        i += 1;
    }

    count
}

/// Computes Mahalanobis distance with precision matrix or diagonal inverse variances using AVX2 + FMA intrinsics.
///
/// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length $d$.
/// - `inv_cov` has length $d$ (diagonal) or $d \times d$ (full matrix).
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mahalanobis_with_inv_cov(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    let d = a.len();
    assert_eq!(d, b.len(), "vector dimensions must match");

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let inv_ptr = inv_cov.as_ptr();

    if inv_cov.len() == d {
        // Diagonal inverse variance: \sum w_i * (a_i - b_i)^2
        let mut sum_vec0 = _mm256_setzero_ps();
        let mut sum_vec1 = _mm256_setzero_ps();

        let chunks = d / 16;
        for i in 0..chunks {
            let base = i * 16;
            let diff0 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base)),
                _mm256_loadu_ps(b_ptr.add(base)),
            );
            let w0 = _mm256_loadu_ps(inv_ptr.add(base));
            let diff0_sq = _mm256_mul_ps(diff0, diff0);
            sum_vec0 = _mm256_fmadd_ps(diff0_sq, w0, sum_vec0);

            let diff1 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base + 8)),
                _mm256_loadu_ps(b_ptr.add(base + 8)),
            );
            let w1 = _mm256_loadu_ps(inv_ptr.add(base + 8));
            let diff1_sq = _mm256_mul_ps(diff1, diff1);
            sum_vec1 = _mm256_fmadd_ps(diff1_sq, w1, sum_vec1);
        }

        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        let mut sum = hsum256_ps(sum_vec0);

        let mut i = chunks * 16;
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
/// - The CPU supports AVX2 and FMA instructions (`target_feature = "avx2,fma"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mahalanobis(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Computes Generalized / Weighted Jaccard (Tanimoto) distance using AVX2 intrinsics.
///
/// $$J_D(\mathbf{u}, \mathbf{v}) = 1.0 - \frac{\sum \min(|u_i|, |v_i|)}{\sum \max(|u_i|, |v_i|) + \epsilon}$$
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 instructions (`target_feature = "avx2"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn jaccard(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    let mut min_vec = _mm256_setzero_ps();
    let mut max_vec = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let abs_a = _mm256_and_ps(_mm256_loadu_ps(a_ptr.add(base)), sign_mask);
        let abs_b = _mm256_and_ps(_mm256_loadu_ps(b_ptr.add(base)), sign_mask);
        min_vec = _mm256_add_ps(min_vec, _mm256_min_ps(abs_a, abs_b));
        max_vec = _mm256_add_ps(max_vec, _mm256_max_ps(abs_a, abs_b));
    }

    let mut sum_min = hsum256_ps(min_vec);
    let mut sum_max = hsum256_ps(max_vec);

    let mut i = chunks * 8;
    while i < n {
        let abs_a = a[i].abs();
        let abs_b = b[i].abs();
        sum_min += abs_a.min(abs_b);
        sum_max += abs_a.max(abs_b);
        i += 1;
    }

    if sum_max <= 1e-9 {
        0.0
    } else {
        1.0 - (sum_min / sum_max)
    }
}

/// Computes Hellinger distance using AVX2 intrinsics.
///
/// # Safety
///
/// The caller must ensure that:
/// - The CPU supports AVX2 instructions (`target_feature = "avx2"`).
/// - `a` and `b` have the same length.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn hellinger(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let n = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    let mut sum_vec = _mm256_setzero_ps();

    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let abs_a = _mm256_and_ps(_mm256_loadu_ps(a_ptr.add(base)), sign_mask);
        let abs_b = _mm256_and_ps(_mm256_loadu_ps(b_ptr.add(base)), sign_mask);
        let diff = _mm256_sub_ps(_mm256_sqrt_ps(abs_a), _mm256_sqrt_ps(abs_b));
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
    }

    let mut sum = hsum256_ps(sum_vec);

    let mut i = chunks * 8;
    while i < n {
        let diff = a[i].abs().sqrt() - b[i].abs().sqrt();
        sum += diff * diff;
        i += 1;
    }

    (0.5 * sum).sqrt()
}

/// Horizontal sum of all 8 lanes in a 256-bit AVX register → single f32.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);

    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(result)
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn has_avx2_fma() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[test]
    fn test_avx2_metrics_match_scalar() {
        if !has_avx2_fma() {
            return;
        }
        for dim in [1, 7, 8, 15, 16, 32, 64, 128, 768] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1).collect();
            let b: Vec<f32> = (0..dim)
                .map(|i| ((i * 13 + 5) % 100) as f32 * 0.1)
                .collect();

            let man_simd = unsafe { manhattan(&a, &b) };
            let man_scalar = crate::scalar::manhattan(&a, &b);
            let rel_man = (man_simd - man_scalar).abs() / man_scalar.max(1.0);
            assert!(
                rel_man < 1e-3,
                "manhattan dim={dim}: simd={man_simd}, scalar={man_scalar}"
            );

            let mink_simd = unsafe { minkowski(&a, &b, 3.0) };
            let mink_scalar = crate::scalar::minkowski(&a, &b, 3.0);
            let rel_mink = (mink_simd - mink_scalar).abs() / mink_scalar.max(1.0);
            assert!(
                rel_mink < 1e-3,
                "minkowski dim={dim}: simd={mink_simd}, scalar={mink_scalar}"
            );

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
            assert!(
                rel_hell < 1e-3,
                "hellinger dim={dim}: simd={hell_simd}, scalar={hell_scalar}"
            );

            let inv_diag: Vec<f32> = (0..dim).map(|i| (i % 5 + 1) as f32 * 0.25).collect();
            let mah_simd = unsafe { mahalanobis_with_inv_cov(&a, &b, &inv_diag) };
            let mah_scalar = crate::scalar::mahalanobis_with_inv_cov(&a, &b, &inv_diag);
            let rel_mah = (mah_simd - mah_scalar).abs() / mah_scalar.max(1.0);
            assert!(
                rel_mah < 1e-3,
                "mahalanobis dim={dim}: simd={mah_simd}, scalar={mah_scalar}"
            );
        }
    }
}
