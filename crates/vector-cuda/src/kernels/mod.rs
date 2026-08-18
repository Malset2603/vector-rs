//! Embedded CUDA PTX Kernel Strings compiled at build time.

pub const KMEANS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kmeans.ptx"));
pub const KNN_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/knn.ptx"));
