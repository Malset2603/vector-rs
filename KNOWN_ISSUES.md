# Known Issues and Technical Debt

This document tracks significant architectural trade-offs, limitations, and technical debt within the `vector-rs` codebase. These items represent design choices or features requiring dedicated architectural refactoring in future development phases.

## 1. Incomplete Mahalanobis Distance Implementation
**Description:** The CUDA kernel computes $L_2$ squared distance (`diff * diff`) for `DistanceMetric::Mahalanobis`, operating under the assumption of a standardized unit covariance matrix ($\Sigma = I$). True Mahalanobis distance requires an inverse covariance matrix ($\Sigma^{-1}$) parameter.
**Rationale & Trade-off:** Injecting a full covariance matrix ($D \times D$) into the universal `knn.cu` or `kmeans.cu` kernel would excessively bloat the kernel argument space, consuming valuable GPU registers and penalizing throughput for all other 9 distance metrics.
**Future Resolution:** 
- Develop a dedicated kernel (e.g., `knn_mahalanobis.cu`) specifically designed to load and apply precision matrices without degrading universal distance evaluation performance.

## 2. Dense Distance Matrix VRAM Footprint for Large Query Batches
**Description:** In `knn.cu`, the pairwise distance calculation generates an intermediate dense matrix of size $Q \times N$ in VRAM before executing the Top-K selection pass. For massive batch queries (e.g., $Q = 10,000, N = 1,000,000$), this requires substantial GPU memory allocation ($O(Q \times N)$).
**Rationale & Trade-off:** Separating distance calculation and Top-K selection into a two-pass pipeline avoids excessive kernel complexity while allowing independent optimization (SoA memory coalescing on pass 1, warp-level reduction on pass 2).
**Future Resolution:**
- Implement dynamic CPU-side query batch chunking to bound maximum VRAM allocation.
- Develop a fused GEMM + Top-K kernel that evaluates and reduces distances in on-chip SRAM/registers without materializing the full distance matrix.

## 3. Global FP32 Precision and Tensor Core Incompatibility
**Description:** The entire `vector-rs` ecosystem (SIMD vectors, HNSW graph structures, and MPI communication buffers) natively uses single-precision floating point (`f32`). CUDA Tensor Cores (`wmma`) strictly require half-precision (`f16` / `bf16`) inputs.
**Rationale & Trade-off:** Maintaining uniform `f32` precision ensures numerical consistency and cross-platform compatibility across CPU SIMD (AVX2/AVX-512/NEON) and GPU layers without introducing runtime type-conversion overhead.
**Future Resolution:**
- Introduce a modular precision layer (using the `half` crate) to support native FP16 vector representations, unlocking hardware Tensor Core acceleration where slight precision trade-offs are acceptable.
