// VectorRS CUDA Kernels: Multi-Metric Parallel k-Means Assignment and Centroid Accumulation
//
// Supports all 10 distance/similarity metrics:
// - Metric 0: L2 Squared (argmin)
// - Metric 1: Dot Product (argmax)
// - Metric 2: Cosine Similarity (argmax)
// - Metric 3: Manhattan (argmin)
// - Metric 4: Minkowski p=3 (argmin, via cbrtf)
// - Metric 5: Chebyshev (argmin)
// - Metric 6: Hamming (argmin)
// - Metric 7: Standardized Mahalanobis (argmin)
// - Metric 8: Weighted Jaccard (argmin)
// - Metric 9: Hellinger (argmin)

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_SIZE 256

// Parallel k-Means Assignment Kernel
extern "C" __global__ void kmeans_assign_and_accumulate(
    const float* __restrict__ data,          // [N x D]
    const float* __restrict__ centroids,     // [C x D]
    int* __restrict__ assignments,           // [N] output cluster index per vector
    float* __restrict__ cluster_sums,        // [C x D] accumulated coordinate sums
    int* __restrict__ cluster_counts,        // [C] accumulated vector counts per cluster
    int N, int C, int D,
    int metric                               // 0..9
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Vector index in [0, N)
    if (idx >= N) return;

    const float* vec = data + idx * D;

    float vec_norm_sq = 0.0f;
    if (metric == 2) {
        for (int d = 0; d < D; ++d) {
            vec_norm_sq += vec[d] * vec[d];
        }
    }

    int best_c = 0;
    float best_score = (metric == 1 || metric == 2) ? -FLT_MAX : FLT_MAX;

    for (int c = 0; c < C; ++c) {
        const float* cent = centroids + c * D;
        
        switch (metric) {
            case 0: { // L2 Squared Distance
                float dist = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float diff = vec[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
            case 1: { // Dot Product
                float dot = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    dot += vec[d] * cent[d];
                }
                if (dot > best_score) {
                    best_score = dot;
                    best_c = c;
                }
                break;
            }
            case 2: { // Cosine Similarity
                float dot = 0.0f;
                float cent_norm_sq = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    dot += vec[d] * cent[d];
                    cent_norm_sq += cent[d] * cent[d];
                }
                float denom = sqrtf(vec_norm_sq) * sqrtf(cent_norm_sq);
                float cos_sim = (denom > 1e-9f) ? (dot / denom) : 0.0f;
                if (cos_sim > best_score) {
                    best_score = cos_sim;
                    best_c = c;
                }
                break;
            }
            case 3: { // Manhattan (L1)
                float dist = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    dist += fabsf(vec[d] - cent[d]);
                }
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
            case 4: { // Minkowski (p=3)
                float sum = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float diff = fabsf(vec[d] - cent[d]);
                    sum += diff * diff * diff;
                }
                float dist = cbrtf(sum);
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
            case 5: { // Chebyshev (L_inf)
                float max_diff = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float diff = fabsf(vec[d] - cent[d]);
                    if (diff > max_diff) max_diff = diff;
                }
                if (max_diff < best_score) {
                    best_score = max_diff;
                    best_c = c;
                }
                break;
            }
            case 6: { // Thresholded Hamming
                float count = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    if (fabsf(vec[d] - cent[d]) > 1e-6f) count += 1.0f;
                }
                if (count < best_score) {
                    best_score = count;
                    best_c = c;
                }
                break;
            }
            case 7: { // Standardized Mahalanobis
                float sum = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float diff = vec[d] - cent[d];
                    sum += diff * diff;
                }
                float dist = sqrtf(sum);
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
            case 8: { // Weighted Jaccard
                float sum_min = 0.0f, sum_max = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float a = fabsf(vec[d]), b = fabsf(cent[d]);
                    sum_min += fminf(a, b);
                    sum_max += fmaxf(a, b);
                }
                float dist = (sum_max <= 1e-9f) ? 0.0f : (1.0f - sum_min / sum_max);
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
            case 9: { // Hellinger
                float sum = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float diff = sqrtf(fabsf(vec[d])) - sqrtf(fabsf(cent[d]));
                    sum += diff * diff;
                }
                float dist = sqrtf(0.5f * sum);
                if (dist < best_score) {
                    best_score = dist;
                    best_c = c;
                }
                break;
            }
        }
    }

    assignments[idx] = best_c;

    // Atomic update of centroid accumulators
    atomicAdd(&cluster_counts[best_c], 1);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&cluster_sums[best_c * D + d], vec[d]);
    }
}
