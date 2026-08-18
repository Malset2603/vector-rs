//! # vector-cuda
//!
//! GPU CUDA accelerated Exact K-NN vector search and k-Means clustering for VectorRS.
//!
//! This crate provides:
//! - **Exact K-NN GEMM search** — GPU-parallel batch distance matrix computation
//!   and top-K candidate reduction.
//! - **Parallel k-Means clustering** — Hardware-accelerated centroid assignment and
//!   atomic coordinate accumulation for rapid index training.
//! - **Device memory abstractions** — `DeviceBuffer` and `CudaDeviceContext` for seamless
//!   Host-to-Device (H2D) and Device-to-Host (D2H) data transfers.

pub mod ddp;
pub mod device;
pub mod error;
pub mod kernels;
pub mod kmeans;
pub mod knn;

pub use ddp::{
    CollectiveOps, DistributedKMeansEngine, DistributedKnnEngine, GpuShardMode, MultiGpuContext,
};
pub use device::{CudaDeviceContext, DeviceBuffer};
pub use error::CudaError;
pub use kmeans::{CudaKMeansEngine, CudaKMeansResult};
pub use knn::CudaKnnEngine;
