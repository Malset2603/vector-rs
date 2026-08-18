//! Scalar (non-SIMD) fallback implementations of vector distance metrics.
//!
//! These serve as the reference baseline for correctness validation and
//! as a fallback on platforms where no SIMD intrinsics are available.

/// Computes the dot product of two f32 slices.
///
/// $$\text{dot}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{d} u_i \cdot v_i$$
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Computes the squared Euclidean (L2²) distance between two f32 slices.
///
/// $$L_2^2(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{d} (u_i - v_i)^2$$
///
/// Returns the **squared** distance to avoid an unnecessary `sqrt` operation
/// since ordering is preserved under monotonic transformations.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Computes cosine similarity between two f32 slices.
///
/// $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}|| \cdot ||\mathbf{v}||}$$
///
/// Returns a value in the range [-1.0, 1.0]. If either vector has near-zero magnitude, returns `0.0`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator <= 1e-9 {
        0.0
    } else {
        dot / denominator
    }
}

/// Computes the Manhattan ($L_1$) distance between two f32 slices.
///
/// $$L_1(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{d} |u_i - v_i|$$
#[inline]
pub fn manhattan(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

/// Computes the Minkowski ($L_p$) distance between two f32 slices for any parameter $p > 0$.
///
/// $$L_p(\mathbf{u}, \mathbf{v}) = \left(\sum_{i=1}^{d} |u_i - v_i|^p\right)^{1/p}$$
#[inline]
pub fn minkowski(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    assert!(p > 0.0, "p must be greater than 0");

    if (p - 1.0).abs() < 1e-5 {
        manhattan(a, b)
    } else if (p - 2.0).abs() < 1e-5 {
        l2_squared(a, b).sqrt()
    } else if (p - 3.0).abs() < 1e-5 {
        let mut sum = 0.0_f32;
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            sum += diff * diff * diff;
        }
        sum.cbrt()
    } else {
        let mut sum = 0.0_f32;
        for i in 0..a.len() {
            sum += (a[i] - b[i]).abs().powf(p);
        }
        sum.powf(1.0 / p)
    }
}

/// Computes the Chebyshev ($L_\infty$) distance between two f32 slices.
///
/// $$L_\infty(\mathbf{u}, \mathbf{v}) = \max_{i=1..d} |u_i - v_i|$$
#[inline]
pub fn chebyshev(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");
    if a.is_empty() {
        return 0.0;
    }

    let mut max_diff = 0.0_f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

/// Computes the Thresholded Float Hamming distance ($L_0$ mismatch count) between two f32 slices.
///
/// Counts the number of coordinate positions where $|a_i - b_i| > 10^{-6}$.
#[inline]
pub fn hamming(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut count = 0.0_f32;
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-6 {
            count += 1.0;
        }
    }
    count
}

/// Computes the Mahalanobis distance between two f32 slices given an inverse covariance matrix $\Sigma^{-1}$
/// or diagonal inverse variance weights.
///
/// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
///
/// - If `inv_cov` has length $D \times D$, it represents the full precision matrix $\Sigma^{-1}$.
/// - If `inv_cov` has length $D$, it represents diagonal precision vector $\text{diag}(\sigma_1^{-2}, \dots, \sigma_D^{-2})$.
///
/// # Panics
///
/// Panics if dimensions mismatch or if `inv_cov` is not positive semi-definite (quadratic form $< 0$).
#[inline]
pub fn mahalanobis_with_inv_cov(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    let d = a.len();
    assert_eq!(d, b.len(), "vector dimensions must match");

    if inv_cov.len() == d * d {
        // Full D x D Precision Matrix
        let mut quad_form = 0.0_f32;
        for i in 0..d {
            let diff_i = a[i] - b[i];
            let row_offset = i * d;
            let mut row_sum = 0.0_f32;
            for j in 0..d {
                row_sum += inv_cov[row_offset + j] * (a[j] - b[j]);
            }
            quad_form += diff_i * row_sum;
        }
        assert!(
            quad_form >= -1e-5,
            "invalid precision matrix: quadratic form evaluates to negative value {}",
            quad_form
        );
        quad_form.max(0.0).sqrt()
    } else if inv_cov.len() == d {
        // Diagonal precision weights
        let mut quad_form = 0.0_f32;
        for i in 0..d {
            let diff = a[i] - b[i];
            quad_form += inv_cov[i] * diff * diff;
        }
        assert!(
            quad_form >= -1e-5,
            "invalid precision weights: negative variance encountered"
        );
        quad_form.max(0.0).sqrt()
    } else {
        panic!(
            "invalid inv_cov dimension: expected {} or {}, got {}",
            d,
            d * d,
            inv_cov.len()
        );
    }
}

/// Computes the Standardized Mahalanobis distance assuming unit diagonal covariance ($\Sigma = I$).
///
/// $$D_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^\top \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$
#[inline]
pub fn mahalanobis(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Computes the Generalized / Weighted Jaccard (Tanimoto) distance between two f32 slices.
///
/// $$J_D(\mathbf{u}, \mathbf{v}) = 1.0 - \frac{\sum \min(|u_i|, |v_i|)}{\sum \max(|u_i|, |v_i|) + \epsilon}$$
#[inline]
pub fn jaccard(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum_min = 0.0_f32;
    let mut sum_max = 0.0_f32;

    for i in 0..a.len() {
        let abs_a = a[i].abs();
        let abs_b = b[i].abs();
        sum_min += abs_a.min(abs_b);
        sum_max += abs_a.max(abs_b);
    }

    if sum_max <= 1e-9 {
        0.0
    } else {
        1.0 - (sum_min / sum_max)
    }
}

/// Computes the Hellinger distance between two f32 slices.
///
/// $$H(\mathbf{u}, \mathbf{v}) = \frac{1}{\sqrt{2}} \sqrt{\sum_{i=1}^{d} (\sqrt{|u_i|} - \sqrt{|v_i|})^2}$$
#[inline]
pub fn hellinger(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimensions must match");

    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i].abs().sqrt() - b[i].abs().sqrt();
        sum += diff * diff;
    }
    (0.5 * sum).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_squared_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        let result = l2_squared(&a, &b);
        assert!((result - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < EPSILON);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_manhattan() {
        let a = [1.0, 5.0, 2.0];
        let b = [4.0, 1.0, 6.0];
        assert!((manhattan(&a, &b) - 11.0).abs() < EPSILON);
        assert!(manhattan(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn test_minkowski() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 2.0, 7.0];
        let res = minkowski(&a, &b, 3.0);
        let expected = (27.0_f32 + 64.0_f32).powf(1.0 / 3.0);
        assert!((res - expected).abs() < EPSILON);

        // p = 1 and p = 2 branches
        assert!((minkowski(&a, &b, 1.0) - manhattan(&a, &b)).abs() < EPSILON);
        assert!((minkowski(&a, &b, 2.0) - l2_squared(&a, &b).sqrt()).abs() < EPSILON);
    }

    #[test]
    fn test_chebyshev() {
        let a = [1.0, 10.0, 3.0];
        let b = [5.0, 2.0, 4.0];
        assert!((chebyshev(&a, &b) - 8.0).abs() < EPSILON);
        assert!(chebyshev(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn test_hamming() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 9.0, 3.0, 8.0];
        assert!((hamming(&a, &b) - 2.0).abs() < EPSILON);
        assert!(hamming(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn test_mahalanobis() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let res = mahalanobis(&a, &b);
        assert!((res - 5.0).abs() < EPSILON);

        // Diagonal precision weights
        let inv_diag = [1.0, 0.25];
        let res_diag = mahalanobis_with_inv_cov(&a, &b, &inv_diag);
        assert!((res_diag - 13.0_f32.sqrt()).abs() < EPSILON);

        // Full 2x2 precision matrix
        let inv_full = [2.0, 1.0, 1.0, 2.0];
        let res_full = mahalanobis_with_inv_cov(&a, &b, &inv_full);
        assert!((res_full - 74.0_f32.sqrt()).abs() < EPSILON);
    }

    #[test]
    fn test_jaccard() {
        let a = [1.0, 2.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        assert!((jaccard(&a, &b) - (2.0 / 3.0)).abs() < EPSILON);
        assert!(jaccard(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn test_hellinger() {
        let a = [0.25, 0.25];
        let b = [0.25, 0.25];
        assert!(hellinger(&a, &b).abs() < EPSILON);
    }
}
