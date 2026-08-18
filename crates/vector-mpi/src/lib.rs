//! # vector-mpi
//!
//! MPI-based distributed k-Means index trainer for the VectorRS engine.
//!
//! This crate provides three execution modes for training IVF coarse centroids:
//!
//! - **Local mode** (default, no feature flags): Reads all shard files, concatenates them,
//!   and trains centroids on a single process. This serves as a baseline for performance
//!   comparison.
//!
//! - **MPI distributed mode** (`--features mpi`): Each MPI rank loads its own shard file
//!   and collaborates via `MPI_Bcast` and `MPI_Allreduce` collective operations to compute
//!   globally consistent centroids across all ranks.
//!
//! - **CUDA-Aware MPI mode** (`--features cuda`): Uses GPU acceleration via `vector-cuda`
//!   for local shard computation, combined with MPI-style synchronization for cross-rank
//!   centroid consistency. Supports single-GPU (`CudaKMeansEngine`) and multi-GPU
//!   (`DistributedKMeansEngine` DDP) within each rank.
//!
//! Both modes produce identical centroid output files that can be loaded by VectorRS
//! Worker nodes for instant index startup.
//!
//! ## Feature Flags
//!
//! - `mpi` — Enables real MPI communication via the [`mpi`](https://crates.io/crates/mpi)
//!   crate (`rsmpi`). Requires a system MPI SDK (MS-MPI on Windows, OpenMPI on Linux).
//! - `cuda` — Enables CUDA-Aware GPU-accelerated k-Means training via the
//!   [`vector-cuda`] crate. Requires NVIDIA CUDA Toolkit.

pub mod distributed_kmeans;

pub use distributed_kmeans::fit_local;

#[cfg(feature = "mpi")]
pub use distributed_kmeans::fit_distributed;

#[cfg(feature = "cuda")]
pub use distributed_kmeans::fit_cuda_aware;
