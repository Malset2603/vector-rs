//! Error types for CUDA acceleration and DDP multi-GPU operations.

use thiserror::Error;

/// Error variants for CUDA hardware initialization and multi-GPU DDP orchestration.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// Insufficient hardware GPUs available compared to the requested count.
    #[error(
        "Insufficient hardware GPUs: requested {requested} device(s), but only {available} physical GPU(s) found"
    )]
    InsufficientDevices { requested: usize, available: usize },

    /// The specified GPU device ordinal was not found or could not be accessed.
    #[error("CUDA device with ordinal {ordinal} was not found or could not be initialized")]
    DeviceNotFound { ordinal: usize },

    /// CUDA driver or runtime error.
    #[error("CUDA driver error: {0}")]
    DriverError(String),

    /// No CUDA-capable hardware devices found on the system.
    #[error("No CUDA-capable GPU hardware devices found on this system")]
    HardwareNotAvailable,

    /// Invalid argument or configuration.
    #[error("Invalid CUDA configuration: {0}")]
    InvalidConfig(String),
}
