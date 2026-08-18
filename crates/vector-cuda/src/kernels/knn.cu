// VectorRS CUDA Kernels: High-Performance Multi-Metric Exact K-NN Search
//
// Optimized Tiled Shared-Memory Architecture for All 10 Distance & Similarity Metrics:
// - Metric 0: Squared Euclidean Distance (L2^2): ||q - x||^2 = ||q||^2 + ||x||^2 - 2<q, x>
// - Metric 1: Dot Product (Inner Product): <q, x>
// - Metric 2: Cosine Similarity: <q, x> / (||q|| * ||x||) (zero-norm convention: 0.0)
// - Metric 3: Manhattan Distance (L1): sum |q_i - x_i| (Tiled Shared Memory)
// - Metric 4: Minkowski Distance (Lp, p=3): cbrtf(sum |q_i - x_i|^3)
// - Metric 5: Chebyshev Distance (L_inf): max |q_i - x_i|
// - Metric 6: Thresholded Hamming Distance: sum I(|q_i - x_i| > eps)
// - Metric 7: Standardized Mahalanobis Distance: sqrt((q - x)^T \Sigma^{-1} (q - x))
// - Metric 8: Weighted / Generalized Jaccard Distance: 1.0 - sum min(|q_i|, |x_i|) / sum max(|q_i|, |x_i|)
// - Metric 9: Hellinger Distribution Distance: sqrt(0.5 * sum (sqrt(|q_i|) - sqrt(|x_i|))^2)

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define TILE_DIM 32

extern "C" __global__ void knn_compute_distance_matrix(
    const float* __restrict__ queries,     // [Q x D]
    const float* __restrict__ dataset,     // [N x D]
    const float* __restrict__ query_norms, // [Q] precomputed squared norms sum(q_i^2)
    const float* __restrict__ data_norms,  // [N] precomputed squared norms sum(x_j^2)
    float* __restrict__ dist_matrix,       // [Q x N] output score matrix
    int Q, int N, int D,
    int metric                             // 0..9
) {
    __shared__ float s_query[TILE_DIM][TILE_DIM];
    __shared__ float s_data[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Query index q in [0, Q)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Data vector index n in [0, N)

    // Accumulators for tiled evaluation
    float acc_primary = 0.0f;
    float acc_secondary = 0.0f;

    int num_tiles = (D + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        int d_q = t * TILE_DIM + threadIdx.x;
        if (row < Q && d_q < D) {
            s_query[threadIdx.y][threadIdx.x] = queries[row * D + d_q];
        } else {
            s_query[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int d_n = t * TILE_DIM + threadIdx.y;
        if (col < N && d_n < D) {
            s_data[threadIdx.y][threadIdx.x] = dataset[col * D + d_n];
        } else {
            s_data[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int valid_d = D - t * TILE_DIM;
        int limit = (valid_d < TILE_DIM) ? valid_d : TILE_DIM;

        #pragma unroll
        for (int k = 0; k < limit; ++k) {
            float q_val = s_query[threadIdx.y][k];
            float d_val = s_data[k][threadIdx.x];

            switch (metric) {
                case 0: // L2 (GEMM dot product)
                case 1: // Dot Product
                case 2: // Cosine Similarity
                    acc_primary += q_val * d_val;
                    break;
                case 3: // Manhattan (L1)
                    acc_primary += fabsf(q_val - d_val);
                    break;
                case 4: { // Minkowski (p=3)
                    float diff = fabsf(q_val - d_val);
                    acc_primary += diff * diff * diff;
                    break;
                }
                case 5: { // Chebyshev (L_inf)
                    float diff = fabsf(q_val - d_val);
                    if (diff > acc_primary) acc_primary = diff;
                    break;
                }
                case 6: // Thresholded Hamming
                    if (fabsf(q_val - d_val) > 1e-6f) {
                        acc_primary += 1.0f;
                    }
                    break;
                case 7: { // Standardized Mahalanobis
                    float diff = q_val - d_val;
                    acc_primary += diff * diff;
                    break;
                }
                case 8: { // Weighted Jaccard (sum_min and sum_max)
                    float a = fabsf(q_val);
                    float b = fabsf(d_val);
                    acc_primary += fminf(a, b);
                    acc_secondary += fmaxf(a, b);
                    break;
                }
                case 9: { // Hellinger
                    float diff = sqrtf(fabsf(q_val)) - sqrtf(fabsf(d_val));
                    acc_primary += diff * diff;
                    break;
                }
            }
        }

        __syncthreads();
    }

    if (row < Q && col < N) {
        float result = 0.0f;

        switch (metric) {
            case 0: { // L2 Squared
                float q_norm = query_norms[row];
                float d_norm = data_norms[col];
                float dist = q_norm + d_norm - 2.0f * acc_primary;
                result = (dist < 0.0f) ? 0.0f : dist;
                break;
            }
            case 1: // Dot Product
                result = acc_primary;
                break;
            case 2: { // Cosine Similarity
                float q_norm = query_norms[row];
                float d_norm = data_norms[col];
                float denom = sqrtf(q_norm) * sqrtf(d_norm);
                result = (denom > 1e-9f) ? (acc_primary / denom) : 0.0f;
                break;
            }
            case 3: // Manhattan
                result = acc_primary;
                break;
            case 4: // Minkowski (p=3)
                result = cbrtf(acc_primary);
                break;
            case 5: // Chebyshev
                result = acc_primary;
                break;
            case 6: // Hamming
                result = acc_primary;
                break;
            case 7: // Mahalanobis
                result = sqrtf(acc_primary);
                break;
            case 8: // Weighted Jaccard
                result = (acc_secondary <= 1e-9f) ? 0.0f : (1.0f - (acc_primary / acc_secondary));
                break;
            case 9: // Hellinger
                result = sqrtf(0.5f * acc_primary);
                break;
        }

        dist_matrix[row * N + col] = result;
    }
}
