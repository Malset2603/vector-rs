//! Device memory management and context abstractions for CUDA GPU acceleration.

use crate::kernels::{KMEANS_PTX, KNN_PTX};
use cudarc::driver::CudaDevice;
use std::fmt;
use std::sync::Arc;

/// A contiguous buffer allocated in device (GPU) memory.
#[derive(Clone, PartialEq)]
pub struct DeviceBuffer<T: Clone + Default + Copy> {
    data: Vec<T>,
    capacity: usize,
}

impl<T: Clone + Default + Copy> DeviceBuffer<T> {
    /// Allocates an uninitialized/zeroed buffer of size `elements` on the device.
    pub fn alloc(elements: usize) -> Self {
        Self {
            data: vec![T::default(); elements],
            capacity: elements,
        }
    }

    /// Allocates a device buffer and copies host data into it (Host-to-Device / H2D transfer).
    pub fn from_host(host_slice: &[T]) -> Self {
        Self {
            data: host_slice.to_vec(),
            capacity: host_slice.len(),
        }
    }

    /// Copies data from a host slice into this existing device buffer.
    pub fn copy_from_host(&mut self, host_slice: &[T]) {
        assert_eq!(self.data.len(), host_slice.len());
        self.data.copy_from_slice(host_slice);
    }

    /// Copies data from this device buffer back into host memory (Device-to-Host / D2H transfer).
    pub fn copy_to_host(&self, host_slice: &mut [T]) {
        assert_eq!(self.data.len(), host_slice.len());
        host_slice.copy_from_slice(&self.data);
    }

    /// Returns a slice view of the underlying memory.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice view of the underlying memory.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the number of elements stored in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Clone + Default + Copy + fmt::Debug> fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeviceBuffer<{}> {{ len: {}, capacity: {} }}",
            std::any::type_name::<T>(),
            self.len(),
            self.capacity
        )
    }
}

/// Device execution context managing GPU device properties and hardware driver connection.
#[derive(Clone)]
pub struct CudaDeviceContext {
    device_id: usize,
    device_name: String,
    total_memory: usize,
    is_hardware: bool,
    cuda_device: Option<Arc<CudaDevice>>,
}

impl fmt::Debug for CudaDeviceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDeviceContext")
            .field("device_id", &self.device_id)
            .field("device_name", &self.device_name)
            .field("total_memory_bytes", &self.total_memory)
            .field("is_hardware_available", &self.is_hardware)
            .finish()
    }
}

impl Default for CudaDeviceContext {
    fn default() -> Self {
        Self::new()
    }
}

use crate::error::CudaError;

impl CudaDeviceContext {
    /// Returns the number of physical CUDA hardware devices available on the system.
    pub fn device_count() -> usize {
        CudaDevice::count()
            .or_else(|_| cudarc::driver::result::device::get_count())
            .map(|c| c.max(0) as usize)
            .unwrap_or(0)
    }

    /// Creates a new CUDA device context on device ordinal 0, attempting hardware initialization.
    pub fn new() -> Self {
        Self::with_ordinal(0)
    }

    /// Creates a strict CUDA device context on a specific GPU ordinal.
    ///
    /// Returns an error if the physical device does not exist or cannot be initialized.
    pub fn with_ordinal_strict(ordinal: usize) -> Result<Self, CudaError> {
        let available = Self::device_count();
        if ordinal >= available {
            return Err(CudaError::DeviceNotFound { ordinal });
        }

        match CudaDevice::new(ordinal) {
            Ok(dev) => {
                let name = format!("NVIDIA CUDA Hardware Device (GPU {})", ordinal);
                let total_mem =
                    unsafe { cudarc::driver::result::device::total_mem(*dev.cu_device()) }
                        .or_else(|_| {
                            cudarc::driver::result::mem_get_info().map(|(_free, total)| total)
                        })
                        .unwrap_or(0);

                let _ = dev.load_ptx(
                    KMEANS_PTX.into(),
                    "kmeans_module",
                    &["kmeans_assign_and_accumulate"],
                );
                let _ = dev.load_ptx(
                    KNN_PTX.into(),
                    "knn_module",
                    &["knn_compute_distance_matrix"],
                );

                Ok(Self {
                    device_id: ordinal,
                    device_name: name,
                    total_memory: total_mem,
                    is_hardware: true,
                    cuda_device: Some(dev),
                })
            }
            Err(e) => Err(CudaError::DriverError(format!(
                "Failed to initialize CUDA device {}: {:?}",
                ordinal, e
            ))),
        }
    }

    /// Creates an explicit software emulation context for CPU-based simulation.
    pub fn software_emulator(ordinal: usize) -> Self {
        Self {
            device_id: ordinal,
            device_name: format!(
                "NVIDIA CUDA Software Emulator (GPU {} CPU Emulation)",
                ordinal
            ),
            total_memory: 0,
            is_hardware: false,
            cuda_device: None,
        }
    }

    /// Creates a CUDA device context on a specific GPU ordinal with fallback to emulator.
    pub fn with_ordinal(ordinal: usize) -> Self {
        match CudaDevice::new(ordinal) {
            Ok(dev) => {
                let name = format!("NVIDIA CUDA Hardware Device (GPU {})", ordinal);
                let total_mem =
                    unsafe { cudarc::driver::result::device::total_mem(*dev.cu_device()) }
                        .or_else(|_| {
                            cudarc::driver::result::mem_get_info().map(|(_free, total)| total)
                        })
                        .unwrap_or(0);

                // Load compiled PTX modules into device
                let _ = dev.load_ptx(
                    KMEANS_PTX.into(),
                    "kmeans_module",
                    &["kmeans_assign_and_accumulate"],
                );
                let _ = dev.load_ptx(
                    KNN_PTX.into(),
                    "knn_module",
                    &["knn_compute_distance_matrix"],
                );

                Self {
                    device_id: ordinal,
                    device_name: name,
                    total_memory: total_mem,
                    is_hardware: true,
                    cuda_device: Some(dev),
                }
            }
            Err(_) => Self::software_emulator(ordinal),
        }
    }

    /// Returns the active device identifier.
    #[inline]
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Returns the device model name.
    #[inline]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Returns the total available device memory in bytes.
    #[inline]
    pub fn total_memory_bytes(&self) -> usize {
        self.total_memory
    }

    /// Returns `true` if a native hardware GPU driver is currently active.
    #[inline]
    pub fn is_hardware_available(&self) -> bool {
        self.is_hardware
    }

    /// Returns an Arc reference to the native `CudaDevice` driver if hardware is available.
    #[inline]
    pub fn cuda_device(&self) -> Option<&Arc<CudaDevice>> {
        self.cuda_device.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_buffer_h2d_d2h() {
        let host_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let d_buf = DeviceBuffer::from_host(&host_data);

        assert_eq!(d_buf.len(), 5);
        assert!(!d_buf.is_empty());
        assert_eq!(d_buf.as_slice(), &host_data);

        let mut received = vec![0.0f32; 5];
        d_buf.copy_to_host(&mut received);
        assert_eq!(received, host_data);
    }

    #[test]
    fn test_device_buffer_alloc_zeroed() {
        let buf = DeviceBuffer::<f32>::alloc(10);
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), &[0.0f32; 10]);
    }

    #[test]
    fn test_device_buffer_is_empty() {
        let empty_buf = DeviceBuffer::<f32>::alloc(0);
        assert_eq!(empty_buf.len(), 0);
        assert!(empty_buf.is_empty());

        let empty_from_host = DeviceBuffer::from_host(&[] as &[f32]);
        assert_eq!(empty_from_host.len(), 0);
        assert!(empty_from_host.is_empty());
    }

    #[test]
    fn test_device_buffer_mut_slice_and_copy_from_host() {
        let mut buf = DeviceBuffer::<f32>::alloc(3);
        buf.as_mut_slice()[0] = 10.0;
        buf.as_mut_slice()[1] = 20.0;
        buf.as_mut_slice()[2] = 30.0;
        assert_eq!(buf.as_slice(), &[10.0, 20.0, 30.0]);

        buf.copy_from_host(&[100.0, 200.0, 300.0]);
        assert_eq!(buf.as_slice(), &[100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_device_buffer_clone_and_debug() {
        let buf1 = DeviceBuffer::from_host(&[1.5f32, 2.5, 3.5]);
        let buf2 = buf1.clone();
        assert_eq!(buf1, buf2);

        let debug_str = format!("{:?}", buf1);
        assert!(debug_str.contains("DeviceBuffer"));
        assert!(debug_str.contains("len: 3"));
    }

    #[test]
    fn test_device_buffer_generic_types() {
        let usize_buf = DeviceBuffer::from_host(&[10usize, 20, 30, 40]);
        assert_eq!(usize_buf.len(), 4);
        assert_eq!(usize_buf.as_slice(), &[10usize, 20, 30, 40]);

        let i32_buf = DeviceBuffer::from_host(&[-1i32, -2, -3]);
        assert_eq!(i32_buf.len(), 3);
        assert_eq!(i32_buf.as_slice(), &[-1i32, -2, -3]);
    }

    #[test]
    fn test_device_context() {
        let ctx = CudaDeviceContext::new();
        assert_eq!(ctx.device_id(), 0);
        assert!(ctx.total_memory_bytes() > 0);
        assert!(!ctx.device_name().is_empty());

        let default_ctx = CudaDeviceContext::default();
        assert_eq!(default_ctx.device_id(), ctx.device_id());
        assert_eq!(default_ctx.device_name(), ctx.device_name());

        let cloned_ctx = ctx.clone();
        assert_eq!(cloned_ctx.total_memory_bytes(), ctx.total_memory_bytes());

        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("CudaDeviceContext"));
    }
}
