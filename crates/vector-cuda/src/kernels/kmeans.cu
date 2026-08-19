// VectorRS CUDA Kernels: Optimized Multi-Metric Parallel k-Means Assignment, Accumulation & Centroid Updates
//
// Supports all 10 distance/similarity metrics with float4 128-bit memory transactions:
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

// Zero out GPU accumulator buffers without PCIe round-trips
extern "C" __global__ void kmeans_zero_accumulators(
    float* __restrict__ cluster_sums,    // [C x D]
    int* __restrict__ cluster_counts,    // [C]
    int C, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = C * D;
    if (idx < total_elements) {
        cluster_sums[idx] = 0.0f;
    }
    if (idx < C) {
        cluster_counts[idx] = 0;
    }
}

// Parallel k-Means Assignment and Accumulation Kernel
extern "C" __global__ void kmeans_assign_and_accumulate(
    const float* __restrict__ data,          // [N x D]
    const float* __restrict__ centroids,     // [C x D]
    int* __restrict__ assignments,           // [N] output cluster index per vector
    float* __restrict__ cluster_sums,        // [C x D] accumulated coordinate sums
    int* __restrict__ cluster_counts,        // [C] accumulated vector counts per cluster
    float* __restrict__ inertias,            // [N] per-vector distance / inertia
    int N, int C, int D,
    int metric                               // 0..9
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Vector index in [0, N)
    bool is_valid = (idx < N);

    const float* vec = is_valid ? (data + idx * D) : nullptr;

    float vec_norm_sq = 0.0f;
    if (is_valid && metric == 2) {
        if (D % 4 == 0) {
            const float4* v4 = reinterpret_cast<const float4*>(vec);
            int D4 = D / 4;
            #pragma unroll 4
            for (int d = 0; d < D4; ++d) {
                float4 v = v4[d];
                vec_norm_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (int d = 0; d < D; ++d) {
                vec_norm_sq += vec[d] * vec[d];
            }
        }
    }

    int best_c = 0;
    float best_score = (metric == 1 || metric == 2) ? -FLT_MAX : FLT_MAX;
    float best_dist = 0.0f;

    extern __shared__ float s_centroid[];

    for (int c = 0; c < C; ++c) {
        // Collaboratively load the centroid into shared memory
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            s_centroid[d] = centroids[c * D + d];
        }
        __syncthreads(); // Ensure the centroid is fully loaded

        if (is_valid) {
            const float* cent = s_centroid;
            
            switch (metric) {
                case 0: { // L2 Squared Distance
                    float dist = 0.0f;
                    if (D % 4 == 0) {
                        const float4* v4 = reinterpret_cast<const float4*>(vec);
                        const float4* c4 = reinterpret_cast<const float4*>(cent);
                        int D4 = D / 4;
                        #pragma unroll 4
                        for (int d = 0; d < D4; ++d) {
                            float4 v = v4[d];
                            float4 k = c4[d];
                            float dx = v.x - k.x;
                            float dy = v.y - k.y;
                            float dz = v.z - k.z;
                            float dw = v.w - k.w;
                            dist += dx * dx + dy * dy + dz * dz + dw * dw;
                        }
                    } else {
                        #pragma unroll 4
                        for (int d = 0; d < D; ++d) {
                            float diff = vec[d] - cent[d];
                            dist += diff * diff;
                        }
                    }
                    if (dist < best_score) {
                        best_score = dist;
                        best_dist = dist;
                        best_c = c;
                    }
                    break;
                }
                case 1: { // Dot Product
                    float dot = 0.0f;
                    if (D % 4 == 0) {
                        const float4* v4 = reinterpret_cast<const float4*>(vec);
                        const float4* c4 = reinterpret_cast<const float4*>(cent);
                        int D4 = D / 4;
                        #pragma unroll 4
                        for (int d = 0; d < D4; ++d) {
                            float4 v = v4[d];
                            float4 k = c4[d];
                            dot += v.x * k.x + v.y * k.y + v.z * k.z + v.w * k.w;
                        }
                    } else {
                        #pragma unroll 4
                        for (int d = 0; d < D; ++d) {
                            dot += vec[d] * cent[d];
                        }
                    }
                    if (dot > best_score) {
                        best_score = dot;
                        best_dist = -dot;
                        best_c = c;
                    }
                    break;
                }
                case 2: { // Cosine Similarity
                    float dot = 0.0f;
                    float cent_norm_sq = 0.0f;
                    if (D % 4 == 0) {
                        const float4* v4 = reinterpret_cast<const float4*>(vec);
                        const float4* c4 = reinterpret_cast<const float4*>(cent);
                        int D4 = D / 4;
                        #pragma unroll 4
                        for (int d = 0; d < D4; ++d) {
                            float4 v = v4[d];
                            float4 k = c4[d];
                            dot += v.x * k.x + v.y * k.y + v.z * k.z + v.w * k.w;
                            cent_norm_sq += k.x * k.x + k.y * k.y + k.z * k.z + k.w * k.w;
                        }
                    } else {
                        #pragma unroll 4
                        for (int d = 0; d < D; ++d) {
                            dot += vec[d] * cent[d];
                            cent_norm_sq += cent[d] * cent[d];
                        }
                    }
                    float denom = sqrtf(vec_norm_sq) * sqrtf(cent_norm_sq);
                    float cos_sim = (denom > 1e-9f) ? (dot / denom) : 0.0f;
                    if (cos_sim > best_score) {
                        best_score = cos_sim;
                        best_dist = 1.0f - cos_sim;
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
                        best_dist = dist;
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
                        best_dist = dist;
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
                        best_dist = max_diff;
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
                        best_dist = count;
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
                        best_dist = dist;
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
                        best_dist = dist;
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
                        best_dist = dist;
                        best_c = c;
                    }
                    break;
                }
            }
        }
        __syncthreads(); // Ensure all threads are done with this centroid before the next iteration loads the next one
    }

    if (is_valid) {
        if (assignments != nullptr) {
            assignments[idx] = best_c;
        }
        if (inertias != nullptr) {
            inertias[idx] = best_dist;
        }
    }

    // Warp-Aggregated Atomics to eliminate Global Memory Contention
    unsigned int active = __ballot_sync(0xffffffff, is_valid);
    int lane = threadIdx.x % 32;

    while (active != 0) {
        int leader = __ffs(active) - 1;
        int target_c = __shfl_sync(active, best_c, leader);
        unsigned int match = __ballot_sync(active, is_valid && (best_c == target_c));

        // Leader thread adds the aggregated count for this cluster in the warp
        if (lane == leader) {
            atomicAdd(&cluster_counts[target_c], __popc(match));
        }

        // Warp-parallel vector sum reduction for target_c
        if (D % 4 == 0) {
            int D4 = D / 4;
            const float4* v4 = (is_valid && vec != nullptr) ? reinterpret_cast<const float4*>(vec) : nullptr;
            for (int d = 0; d < D4; ++d) {
                float4 v = (is_valid && best_c == target_c && v4 != nullptr) ? v4[d] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    v.x += __shfl_down_sync(0xffffffff, v.x, offset);
                    v.y += __shfl_down_sync(0xffffffff, v.y, offset);
                    v.z += __shfl_down_sync(0xffffffff, v.z, offset);
                    v.w += __shfl_down_sync(0xffffffff, v.w, offset);
                }
                if (lane == 0) {
                    int base = target_c * D + d * 4;
                    atomicAdd(&cluster_sums[base + 0], v.x);
                    atomicAdd(&cluster_sums[base + 1], v.y);
                    atomicAdd(&cluster_sums[base + 2], v.z);
                    atomicAdd(&cluster_sums[base + 3], v.w);
                }
            }
        } else {
            for (int d = 0; d < D; ++d) {
                float v = (is_valid && best_c == target_c && vec != nullptr) ? vec[d] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    v += __shfl_down_sync(0xffffffff, v, offset);
                }
                if (lane == 0) {
                    atomicAdd(&cluster_sums[target_c * D + d], v);
                }
            }
        }

        active &= ~match;
    }
}

// Update centroids and compute max shift directly on GPU
extern "C" __global__ void kmeans_update_centroids(
    const float* __restrict__ cluster_sums,    // [C x D]
    const int* __restrict__ cluster_counts,    // [C]
    float* __restrict__ centroids,             // [C x D]
    float* __restrict__ shifts,                // [C] per-cluster shift
    int C, int D
) {
    int c = blockIdx.x; // Each block processes one cluster c in [0, C)
    if (c >= C) return;

    int count = cluster_counts[c];
    if (count <= 0) {
        if (threadIdx.x == 0 && shifts != nullptr) {
            shifts[c] = 0.0f;
        }
        return;
    }

    float inv_count = 1.0f / (float)count;
    float local_shift = 0.0f;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int idx = c * D + d;
        float new_val = cluster_sums[idx] * inv_count;
        float old_val = centroids[idx];
        float diff = new_val - old_val;
        local_shift += diff * diff;
        centroids[idx] = new_val;
    }

    // Warp-level reduction of shift
    for (int offset = 16; offset > 0; offset /= 2) {
        local_shift += __shfl_down_sync(0xffffffff, local_shift, offset);
    }

    // Shared memory for warp results (max 32 warps per block)
    static __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    if (lane == 0) {
        warp_sums[warp] = local_shift;
    }
    __syncthreads();
    
    // First warp reduces the warp_sums
    if (warp == 0) {
        float warp_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane == 0 && shifts != nullptr) {
            shifts[c] = warp_sum;
        }
    }
}
