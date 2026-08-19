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

    int col = blockIdx.x * TILE_DIM + threadIdx.x; 

    // Accumulators for tiled evaluation (1x4 register tiling)
    float acc_primary[4] = {0.0f};
    float acc_secondary[4] = {0.0f};

    int num_tiles = (D + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        // Load 32x32 s_query using 32x8 threads (each thread loads 4 elements)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int ty = threadIdx.y + i * 8;
            int global_row = blockIdx.y * TILE_DIM + ty;
            int d_q = t * TILE_DIM + threadIdx.x;
            if (global_row < Q && d_q < D) {
                s_query[ty][threadIdx.x] = queries[global_row * D + d_q];
            } else {
                s_query[ty][threadIdx.x] = 0.0f;
            }
        }

        // Load 32x32 s_data using 32x8 threads (SoA layout)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int ty = threadIdx.y + i * 8;
            int d_n = t * TILE_DIM + ty;
            if (col < N && d_n < D) {
                s_data[ty][threadIdx.x] = dataset[d_n * N + col];
            } else {
                s_data[ty][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();

        int valid_d = D - t * TILE_DIM;
        int limit = (valid_d < TILE_DIM) ? valid_d : TILE_DIM;

        for (int k = 0; k < limit; ++k) {
            float d_val = s_data[k][threadIdx.x]; // Broadcast from shared memory

            switch (metric) {
                case 0: // L2 (GEMM dot product)
                case 1: // Dot Product
                case 2: // Cosine Similarity
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        acc_primary[i] += q_val * d_val;
                    }
                    break;
                case 3: // Manhattan (L1)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        acc_primary[i] += fabsf(q_val - d_val);
                    }
                    break;
                case 4: // Minkowski (p=3)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        float diff = fabsf(q_val - d_val);
                        acc_primary[i] += diff * diff * diff;
                    }
                    break;
                case 5: // Chebyshev (L_inf)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        float diff = fabsf(q_val - d_val);
                        if (diff > acc_primary[i]) acc_primary[i] = diff;
                    }
                    break;
                case 6: // Thresholded Hamming
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        if (fabsf(q_val - d_val) > 1e-6f) {
                            acc_primary[i] += 1.0f;
                        }
                    }
                    break;
                case 7: // Standardized Mahalanobis
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        float diff = q_val - d_val;
                        acc_primary[i] += diff * diff;
                    }
                    break;
                case 8: // Weighted Jaccard
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        float a = fabsf(q_val);
                        float b = fabsf(d_val);
                        acc_primary[i] += fminf(a, b);
                        acc_secondary[i] += fmaxf(a, b);
                    }
                    break;
                case 9: // Hellinger
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float q_val = s_query[threadIdx.y + i * 8][k];
                        float diff = sqrtf(fabsf(q_val)) - sqrtf(fabsf(d_val));
                        acc_primary[i] += diff * diff;
                    }
                    break;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int global_row = blockIdx.y * TILE_DIM + threadIdx.y + i * 8;
        
        if (global_row < Q && col < N) {
            float result = 0.0f;
            float ap = acc_primary[i];
            float as = acc_secondary[i];

            switch (metric) {
                case 0: { // L2 Squared
                    float q_norm = query_norms[global_row];
                    float d_norm = data_norms[col];
                    float dist = q_norm + d_norm - 2.0f * ap;
                    result = (dist < 0.0f) ? 0.0f : dist;
                    break;
                }
                case 1: // Dot Product
                    result = ap;
                    break;
                case 2: { // Cosine Similarity
                    float q_norm = query_norms[global_row];
                    float d_norm = data_norms[col];
                    float denom = sqrtf(q_norm) * sqrtf(d_norm);
                    result = (denom > 1e-9f) ? (ap / denom) : 0.0f;
                    break;
                }
                case 3: // Manhattan
                    result = ap;
                    break;
                case 4: // Minkowski (p=3)
                    result = cbrtf(ap);
                    break;
                case 5: // Chebyshev
                    result = ap;
                    break;
                case 6: // Hamming
                    result = ap;
                    break;
                case 7: // Mahalanobis
                    result = sqrtf(ap);
                    break;
                case 8: // Weighted Jaccard
                    result = (as <= 1e-9f) ? 0.0f : (1.0f - (ap / as));
                    break;
                case 9: // Hellinger
                    result = sqrtf(0.5f * ap);
                    break;
            }

            dist_matrix[global_row * N + col] = result;
        }
    }
}

#define MAX_K 128

// Top-K Selection Kernel (Warp-Level Parallel, 32 threads per query)
extern "C" __global__ void knn_topk_select(
    const float* __restrict__ dist_matrix,
    float* __restrict__ topk_distances,
    int* __restrict__ topk_indices,
    int Q, int N, int K,
    int metric
) {
    int q = blockIdx.x; // 1 block per query
    if (q >= Q) return;
    
    int lane = threadIdx.x; // 0 to 31
    if (lane >= 32) return; // Ensure only 1 warp is used
    if (K > MAX_K) K = MAX_K; // Safety limit

    // Is it an argmax metric? (Dot Product or Cosine Similarity)
    int is_argmax = (metric == 1 || metric == 2) ? 1 : 0;
    const float* row = dist_matrix + q * N;
    
    // Thread-local array for top-K
    float best_dists[MAX_K];
    int best_indices[MAX_K];
    
    for (int k = 0; k < K; ++k) {
        best_dists[k] = is_argmax ? -FLT_MAX : FLT_MAX;
        best_indices[k] = -1;
    }

    // Parallel scan of N elements (each thread scans N/32)
    for (int i = lane; i < N; i += 32) {
        float val = row[i];
        
        // Insertion sort logic
        if ((!is_argmax && val < best_dists[K - 1]) || (is_argmax && val > best_dists[K - 1])) {
            int pos = K - 1;
            while (pos > 0) {
                bool swap = (!is_argmax && val < best_dists[pos - 1]) || (is_argmax && val > best_dists[pos - 1]);
                if (swap) {
                    best_dists[pos] = best_dists[pos - 1];
                    best_indices[pos] = best_indices[pos - 1];
                    pos--;
                } else {
                    break;
                }
            }
            best_dists[pos] = val;
            best_indices[pos] = i;
        }
    }

    // Write thread-local results to shared memory
    __shared__ float s_dists[32 * MAX_K];
    __shared__ int s_indices[32 * MAX_K];
    
    for (int k = 0; k < K; ++k) {
        s_dists[lane * K + k] = best_dists[k];
        s_indices[lane * K + k] = best_indices[k];
    }
    
    __syncthreads();
    
    // Thread 0 merges the 32 arrays
    if (lane == 0) {
        float final_dists[MAX_K];
        int final_indices[MAX_K];
        for (int k = 0; k < K; ++k) {
            final_dists[k] = is_argmax ? -FLT_MAX : FLT_MAX;
            final_indices[k] = -1;
        }
        
        // Merge all 32 * K elements
        for (int i = 0; i < 32 * K; ++i) {
            float val = s_dists[i];
            int idx = s_indices[i];
            
            if (idx == -1) continue;
            
            if ((!is_argmax && val < final_dists[K - 1]) || (is_argmax && val > final_dists[K - 1])) {
                int pos = K - 1;
                while (pos > 0) {
                    bool swap = (!is_argmax && val < final_dists[pos - 1]) || (is_argmax && val > final_dists[pos - 1]);
                    if (swap) {
                        final_dists[pos] = final_dists[pos - 1];
                        final_indices[pos] = final_indices[pos - 1];
                        pos--;
                    } else {
                        break;
                    }
                }
                final_dists[pos] = val;
                final_indices[pos] = idx;
            }
        }
        
        // Write to global memory
        for (int k = 0; k < K; ++k) {
            topk_distances[q * K + k] = final_dists[k];
            topk_indices[q * K + k] = final_indices[k];
        }
    }
}
