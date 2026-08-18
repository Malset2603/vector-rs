//! Distributed Data Parallel (DDP) Multi-GPU Engine for VectorRS.
//!
//! Provides multi-GPU sharded/replicated Exact K-NN search, distributed k-Means clustering,
//! and NCCL-style collective tensor synchronization.

pub mod collective;
pub mod distributed_kmeans;
pub mod distributed_knn;

pub use collective::CollectiveOps;
pub use distributed_kmeans::DistributedKMeansEngine;
pub use distributed_knn::{DistributedKnnEngine, GpuShardMode};

use crate::device::CudaDeviceContext;
use crate::error::CudaError;

/// Context managing a cluster of $G$ GPU hardware devices.
#[derive(Debug, Clone)]
pub struct MultiGpuContext {
    num_gpus: usize,
    devices: Vec<CudaDeviceContext>,
}

impl MultiGpuContext {
    /// Creates a new `MultiGpuContext` representing `num_gpus` device ranks, strictly validating hardware availability.
    ///
    /// Returns `Err(CudaError::InsufficientDevices)` if `num_gpus` exceeds the number of physical hardware GPUs available.
    pub fn try_new(num_gpus: usize) -> Result<Self, CudaError> {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        let available = CudaDeviceContext::device_count();
        if num_gpus > available {
            return Err(CudaError::InsufficientDevices {
                requested: num_gpus,
                available,
            });
        }

        let mut devices = Vec::with_capacity(num_gpus);
        for ordinal in 0..num_gpus {
            devices.push(CudaDeviceContext::with_ordinal_strict(ordinal)?);
        }

        Ok(Self { num_gpus, devices })
    }

    /// Creates a `MultiGpuContext` using software emulation (for CPU testing and simulation).
    pub fn emulator(num_gpus: usize) -> Self {
        assert!(num_gpus >= 1, "num_gpus must be at least 1");
        let devices = (0..num_gpus)
            .map(CudaDeviceContext::software_emulator)
            .collect();
        Self { num_gpus, devices }
    }

    /// Creates a new `MultiGpuContext` representing `num_gpus` device ranks.
    ///
    /// # Panics
    /// Panics if the requested number of GPUs exceeds available physical hardware devices.
    pub fn new(num_gpus: usize) -> Self {
        Self::try_new(num_gpus).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Returns the number of active GPU devices in the context.
    #[inline]
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }

    /// Returns a slice of individual device contexts.
    #[inline]
    pub fn devices(&self) -> &[CudaDeviceContext] {
        &self.devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_context_emulator() {
        let ctx = MultiGpuContext::emulator(8);
        assert_eq!(ctx.num_gpus(), 8);
        assert_eq!(ctx.devices().len(), 8);
    }

    #[test]
    fn test_multi_gpu_context_emulator_single_gpu() {
        let ctx = MultiGpuContext::emulator(1);
        assert_eq!(ctx.num_gpus(), 1);
        assert_eq!(ctx.devices().len(), 1);
        assert_eq!(ctx.devices()[0].device_id(), 0);
    }

    #[test]
    fn test_multi_gpu_context_clone_and_debug() {
        let ctx = MultiGpuContext::emulator(2);
        let cloned = ctx.clone();
        assert_eq!(cloned.num_gpus(), 2);
        assert_eq!(cloned.devices().len(), 2);

        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("MultiGpuContext"));
    }

    #[test]
    fn test_insufficient_hardware_error() {
        let available = CudaDeviceContext::device_count();
        let requested = available + 100;
        let res = MultiGpuContext::try_new(requested);
        assert!(res.is_err());
        match res.unwrap_err() {
            CudaError::InsufficientDevices {
                requested: r,
                available: a,
            } => {
                assert_eq!(r, requested);
                assert_eq!(a, available);
            }
            other => panic!("Expected InsufficientDevices error, got: {:?}", other),
        }
    }
}
