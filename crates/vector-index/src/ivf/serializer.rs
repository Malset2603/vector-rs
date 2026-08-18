//! Binary serialization and deserialization for IVF-PQ index state.
//!
//! Provides fast zero-overhead disk persistence and recovery of coarse centroids,
//! PQ codebooks, and inverted list partitions.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::config::IvfPqConfig;
use super::inverted_list::{InvertedIndex, InvertedList};
use super::pq::ProductQuantizer;
use crate::DistanceMetric;
use crate::types::{Result, VectorId, VectorIndexError};

/// Magic identifier for VectorRS IVF-PQ binary format ("IVFPQVEC").
pub const IVF_PQ_MAGIC: u64 = 0x4956_4650_5156_4543;

/// Current IVF-PQ binary format version.
pub const IVF_PQ_VERSION: u32 = 1;

/// Serializer for saving and loading IVF-PQ index structures.
pub struct IvfPqSerializer;

impl IvfPqSerializer {
    /// Serializes and writes IVF-PQ index components to a binary file at `path`.
    pub fn save_to_file<P: AsRef<Path>>(
        path: P,
        config: &IvfPqConfig,
        coarse_centroids: &[f32],
        pq: &ProductQuantizer,
        inverted_index: &InvertedIndex,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // 1. Magic & Version Header
        writer.write_all(&IVF_PQ_MAGIC.to_le_bytes())?;
        writer.write_all(&IVF_PQ_VERSION.to_le_bytes())?;

        // 2. Metric & Dimension
        let metric_code: u8 = match metric {
            DistanceMetric::L2Squared => 0,
            DistanceMetric::DotProduct => 1,
            DistanceMetric::CosineSimilarity => 2,
            DistanceMetric::Manhattan => 3,
            DistanceMetric::Minkowski => 4,
            DistanceMetric::Chebyshev => 5,
            DistanceMetric::Hamming => 6,
            DistanceMetric::Mahalanobis => 7,
            DistanceMetric::Jaccard => 8,
            DistanceMetric::Hellinger => 9,
        };
        writer.write_all(&metric_code.to_le_bytes())?;
        writer.write_all(&(dimension as u32).to_le_bytes())?;

        // 3. Config
        writer.write_all(&(config.nlist as u32).to_le_bytes())?;
        writer.write_all(&(config.nprobe as u32).to_le_bytes())?;
        writer.write_all(&(config.num_subvectors as u32).to_le_bytes())?;
        writer.write_all(&(config.sub_clusters as u32).to_le_bytes())?;
        writer.write_all(&(config.max_kmeans_iters as u32).to_le_bytes())?;
        writer.write_all(&config.kmeans_tolerance.to_le_bytes())?;

        // 4. Coarse Centroids
        let num_centroids = (coarse_centroids.len() / dimension) as u32;
        writer.write_all(&num_centroids.to_le_bytes())?;
        for &val in coarse_centroids {
            writer.write_all(&val.to_le_bytes())?;
        }

        // 5. PQ Codebooks
        let codebooks_len = pq.codebooks.len() as u32;
        writer.write_all(&codebooks_len.to_le_bytes())?;
        for &val in &pq.codebooks {
            writer.write_all(&val.to_le_bytes())?;
        }

        // 6. Inverted Index Lists
        let nlist = inverted_index.nlist() as u32;
        writer.write_all(&nlist.to_le_bytes())?;

        for list in &inverted_index.lists {
            let count = list.len() as u32;
            writer.write_all(&count.to_le_bytes())?;

            for &id in &list.ids {
                writer.write_all(&id.to_le_bytes())?;
            }

            writer.write_all(&list.codes)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Loads and deserializes IVF-PQ index components from a binary file at `path`.
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<(
        IvfPqConfig,
        Vec<f32>,
        ProductQuantizer,
        InvertedIndex,
        usize,
        DistanceMetric,
    )> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)?;
        let mut reader = BufReader::new(file);

        // 1. Magic Header Validation
        let mut magic_buf = [0u8; 8];
        reader.read_exact(&mut magic_buf)?;
        let magic = u64::from_le_bytes(magic_buf);
        if magic != IVF_PQ_MAGIC {
            return Err(VectorIndexError::InvalidHeader {
                path: path_buf,
                reason: format!("expected magic {IVF_PQ_MAGIC:#X}, got {magic:#X}"),
            });
        }

        // 2. Version Header Validation
        let mut ver_buf = [0u8; 4];
        reader.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);
        if version != IVF_PQ_VERSION {
            return Err(VectorIndexError::InvalidHeader {
                path: path_buf,
                reason: format!("unsupported version: {version}"),
            });
        }

        // 3. Metric & Dimension
        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let metric = match metric_buf[0] {
            0 => DistanceMetric::L2Squared,
            1 => DistanceMetric::DotProduct,
            2 => DistanceMetric::CosineSimilarity,
            3 => DistanceMetric::Manhattan,
            4 => DistanceMetric::Minkowski,
            5 => DistanceMetric::Chebyshev,
            6 => DistanceMetric::Hamming,
            7 => DistanceMetric::Mahalanobis,
            8 => DistanceMetric::Jaccard,
            9 => DistanceMetric::Hellinger,
            other => {
                return Err(VectorIndexError::InvalidHeader {
                    path: path_buf,
                    reason: format!("invalid metric code: {other}"),
                });
            }
        };

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let dimension = u32::from_le_bytes(u32_buf) as usize;

        // 4. Config
        reader.read_exact(&mut u32_buf)?;
        let nlist = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let nprobe = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let num_subvectors = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let sub_clusters = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let max_kmeans_iters = u32::from_le_bytes(u32_buf) as usize;

        let mut f32_buf = [0u8; 4];
        reader.read_exact(&mut f32_buf)?;
        let kmeans_tolerance = f32::from_le_bytes(f32_buf);

        let config = IvfPqConfig {
            nlist,
            nprobe,
            num_subvectors,
            sub_clusters,
            max_kmeans_iters,
            kmeans_tolerance,
            max_train_points: None,
        };

        // 5. Coarse Centroids
        reader.read_exact(&mut u32_buf)?;
        let num_centroids = u32::from_le_bytes(u32_buf) as usize;
        let mut coarse_centroids = Vec::with_capacity(num_centroids * dimension);
        for _ in 0..(num_centroids * dimension) {
            reader.read_exact(&mut f32_buf)?;
            coarse_centroids.push(f32::from_le_bytes(f32_buf));
        }

        // 6. PQ Codebooks
        reader.read_exact(&mut u32_buf)?;
        let codebooks_len = u32::from_le_bytes(u32_buf) as usize;
        let mut codebooks = Vec::with_capacity(codebooks_len);
        for _ in 0..codebooks_len {
            reader.read_exact(&mut f32_buf)?;
            codebooks.push(f32::from_le_bytes(f32_buf));
        }
        let pq = ProductQuantizer::new(dimension, num_subvectors, sub_clusters, codebooks);

        // 7. Inverted Index Lists
        reader.read_exact(&mut u32_buf)?;
        let lists_count = u32::from_le_bytes(u32_buf) as usize;
        let mut lists = Vec::with_capacity(lists_count);
        let mut total_vectors = 0;

        for _ in 0..lists_count {
            reader.read_exact(&mut u32_buf)?;
            let count = u32::from_le_bytes(u32_buf) as usize;

            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                reader.read_exact(&mut u32_buf)?;
                ids.push(u32::from_le_bytes(u32_buf) as VectorId);
            }

            let mut codes = vec![0u8; count * num_subvectors];
            reader.read_exact(&mut codes)?;

            total_vectors += count;
            lists.push(InvertedList { ids, codes });
        }

        let inverted_index = InvertedIndex {
            lists,
            num_subvectors,
            total_vectors,
        };

        Ok((
            config,
            coarse_centroids,
            pq,
            inverted_index,
            dimension,
            metric,
        ))
    }

    // ------------------------------------------------------------------
    // Centroid-only serialization (used by MPI distributed trainer)
    // ------------------------------------------------------------------

    /// Serializes coarse centroids and training metadata to a standalone binary file.
    ///
    /// This lightweight format stores only the centroid vectors (no PQ codebooks or
    /// inverted lists), enabling the MPI distributed trainer to output centroid files
    /// that can later be loaded by workers or fed into a full IVF-PQ build pipeline.
    ///
    /// # Binary Format
    ///
    /// | Offset | Size       | Field                          |
    /// |--------|------------|--------------------------------|
    /// | 0      | 8          | Magic (`CENTROID_FILE_MAGIC`)   |
    /// | 8      | 4          | Version (u32)                  |
    /// | 12     | 1          | Metric code (u8)               |
    /// | 13     | 4          | Dimension (u32)                |
    /// | 17     | 4          | K — number of centroids (u32)  |
    /// | 21     | 4          | Iterations (u32)               |
    /// | 25     | 4          | Inertia (f32)                  |
    /// | 29     | K×D×4      | Centroid data (f32 LE)         |
    pub fn save_centroid_file<P: AsRef<Path>>(
        path: P,
        centroids: &[f32],
        dimension: usize,
        k: usize,
        metric: DistanceMetric,
        iterations: usize,
        inertia: f32,
    ) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        writer.write_all(&CENTROID_FILE_MAGIC.to_le_bytes())?;
        writer.write_all(&CENTROID_FILE_VERSION.to_le_bytes())?;

        // Metric
        let metric_code: u8 = metric_to_code(metric);
        writer.write_all(&metric_code.to_le_bytes())?;

        // Dimension, K, iterations, inertia
        writer.write_all(&(dimension as u32).to_le_bytes())?;
        writer.write_all(&(k as u32).to_le_bytes())?;
        writer.write_all(&(iterations as u32).to_le_bytes())?;
        writer.write_all(&inertia.to_le_bytes())?;

        // Centroid data
        for &val in centroids {
            writer.write_all(&val.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Loads coarse centroids and training metadata from a standalone centroid file.
    pub fn load_centroid_file<P: AsRef<Path>>(path: P) -> Result<CentroidFileData> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)?;
        let mut reader = BufReader::new(file);

        // Magic
        let mut magic_buf = [0u8; 8];
        reader.read_exact(&mut magic_buf)?;
        let magic = u64::from_le_bytes(magic_buf);
        if magic != CENTROID_FILE_MAGIC {
            return Err(VectorIndexError::InvalidHeader {
                path: path_buf,
                reason: format!("expected centroid magic {CENTROID_FILE_MAGIC:#X}, got {magic:#X}"),
            });
        }

        // Version
        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let version = u32::from_le_bytes(u32_buf);
        if version != CENTROID_FILE_VERSION {
            return Err(VectorIndexError::InvalidHeader {
                path: path_buf,
                reason: format!("unsupported centroid file version: {version}"),
            });
        }

        // Metric
        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let metric = code_to_metric(metric_buf[0], &path_buf)?;

        // Dimension, K, iterations, inertia
        reader.read_exact(&mut u32_buf)?;
        let dimension = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let k = u32::from_le_bytes(u32_buf) as usize;

        reader.read_exact(&mut u32_buf)?;
        let iterations = u32::from_le_bytes(u32_buf) as usize;

        let mut f32_buf = [0u8; 4];
        reader.read_exact(&mut f32_buf)?;
        let inertia = f32::from_le_bytes(f32_buf);

        // Centroid data
        let mut centroids = Vec::with_capacity(k * dimension);
        for _ in 0..(k * dimension) {
            reader.read_exact(&mut f32_buf)?;
            centroids.push(f32::from_le_bytes(f32_buf));
        }

        Ok(CentroidFileData {
            centroids,
            dimension,
            k,
            metric,
            iterations,
            inertia,
        })
    }
}

/// Magic identifier for VectorRS standalone centroid binary files ("VRSKMEAN").
pub const CENTROID_FILE_MAGIC: u64 = 0x5652_534B_4D45_414E;

/// Current centroid file format version.
pub const CENTROID_FILE_VERSION: u32 = 1;

/// Data loaded from a standalone centroid binary file.
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidFileData {
    /// Flat buffer of $k$ centroid vectors, length $k \times D$.
    pub centroids: Vec<f32>,
    /// Dimensionality of each centroid.
    pub dimension: usize,
    /// Number of centroids ($k$).
    pub k: usize,
    /// Distance metric used during training.
    pub metric: DistanceMetric,
    /// Number of Lloyd's iterations completed.
    pub iterations: usize,
    /// Final inertia (sum of distances) at convergence.
    pub inertia: f32,
}

/// Converts a `DistanceMetric` to its wire code byte.
fn metric_to_code(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::L2Squared => 0,
        DistanceMetric::DotProduct => 1,
        DistanceMetric::CosineSimilarity => 2,
        DistanceMetric::Manhattan => 3,
        DistanceMetric::Minkowski => 4,
        DistanceMetric::Chebyshev => 5,
        DistanceMetric::Hamming => 6,
        DistanceMetric::Mahalanobis => 7,
        DistanceMetric::Jaccard => 8,
        DistanceMetric::Hellinger => 9,
    }
}

/// Converts a wire code byte back to a `DistanceMetric`.
fn code_to_metric(code: u8, path: &std::path::Path) -> Result<DistanceMetric> {
    match code {
        0 => Ok(DistanceMetric::L2Squared),
        1 => Ok(DistanceMetric::DotProduct),
        2 => Ok(DistanceMetric::CosineSimilarity),
        3 => Ok(DistanceMetric::Manhattan),
        4 => Ok(DistanceMetric::Minkowski),
        5 => Ok(DistanceMetric::Chebyshev),
        6 => Ok(DistanceMetric::Hamming),
        7 => Ok(DistanceMetric::Mahalanobis),
        8 => Ok(DistanceMetric::Jaccard),
        9 => Ok(DistanceMetric::Hellinger),
        other => Err(VectorIndexError::InvalidHeader {
            path: path.to_path_buf(),
            reason: format!("invalid metric code: {other}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_serializer_roundtrip() {
        let dimension = 4;
        let config = IvfPqConfig::new(2, 1, 2)
            .with_sub_clusters(2)
            .with_max_kmeans_iters(10);

        let coarse_centroids = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let codebooks = vec![
            // Sub 0
            0.0, 0.0, 0.5, 0.5, // Sub 1
            0.0, 0.0, 0.5, 0.5,
        ];
        let pq = ProductQuantizer::new(dimension, 2, 2, codebooks);

        let mut inv_index = InvertedIndex::new(2, 2);
        inv_index.add(0, 10, &[0, 1]);
        inv_index.add(1, 20, &[1, 0]);

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        IvfPqSerializer::save_to_file(
            path,
            &config,
            &coarse_centroids,
            &pq,
            &inv_index,
            dimension,
            DistanceMetric::L2Squared,
        )
        .unwrap();

        let (loaded_cfg, loaded_cc, loaded_pq, loaded_inv, loaded_dim, loaded_metric) =
            IvfPqSerializer::load_from_file(path).unwrap();

        assert_eq!(loaded_cfg, config);
        assert_eq!(loaded_cc, coarse_centroids);
        assert_eq!(loaded_pq, pq);
        assert_eq!(loaded_inv, inv_index);
        assert_eq!(loaded_dim, dimension);
        assert_eq!(loaded_metric, DistanceMetric::L2Squared);
    }

    #[test]
    fn test_serializer_metrics_roundtrip() {
        let dimension = 2;
        let config = IvfPqConfig::new(1, 1, 1).with_sub_clusters(1);
        let coarse_centroids = vec![1.0, 2.0];
        let codebooks = vec![1.0, 2.0];
        let pq = ProductQuantizer::new(dimension, 1, 1, codebooks);
        let mut inv_index = InvertedIndex::new(1, 1);
        inv_index.add(0, 1, &[0]);

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        for metric in [DistanceMetric::DotProduct, DistanceMetric::CosineSimilarity] {
            IvfPqSerializer::save_to_file(
                path,
                &config,
                &coarse_centroids,
                &pq,
                &inv_index,
                dimension,
                metric,
            )
            .unwrap();

            let (_, _, _, _, _, loaded_metric) = IvfPqSerializer::load_from_file(path).unwrap();
            assert_eq!(loaded_metric, metric);
        }
    }

    #[test]
    fn test_serializer_invalid_magic() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"BADMAGIC12345678").unwrap();
        let res = IvfPqSerializer::load_from_file(tmp.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_serializer_invalid_version() {
        let tmp = NamedTempFile::new().unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&IVF_PQ_MAGIC.to_le_bytes());
        data.extend_from_slice(&999u32.to_le_bytes()); // invalid version
        std::fs::write(tmp.path(), &data).unwrap();

        let res = IvfPqSerializer::load_from_file(tmp.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_serializer_invalid_metric_code() {
        let tmp = NamedTempFile::new().unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&IVF_PQ_MAGIC.to_le_bytes());
        data.extend_from_slice(&IVF_PQ_VERSION.to_le_bytes());
        data.push(99); // invalid metric
        std::fs::write(tmp.path(), &data).unwrap();

        let res = IvfPqSerializer::load_from_file(tmp.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_serializer_non_existent_file() {
        let res = IvfPqSerializer::load_from_file("non_existent_file_path_12345.bin");
        assert!(res.is_err());
    }

    #[test]
    fn test_centroid_file_roundtrip() {
        let dimension = 4;
        let k = 2;
        let centroids = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let metric = DistanceMetric::L2Squared;
        let iterations = 15;
        let inertia = 42.5;

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        IvfPqSerializer::save_centroid_file(
            path, &centroids, dimension, k, metric, iterations, inertia,
        )
        .unwrap();

        let loaded = IvfPqSerializer::load_centroid_file(path).unwrap();
        assert_eq!(loaded.centroids, centroids);
        assert_eq!(loaded.dimension, dimension);
        assert_eq!(loaded.k, k);
        assert_eq!(loaded.metric, metric);
        assert_eq!(loaded.iterations, iterations);
        assert!((loaded.inertia - inertia).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_file_all_metrics() {
        let centroids = vec![1.0, 2.0];
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path();

        for metric in [
            DistanceMetric::L2Squared,
            DistanceMetric::DotProduct,
            DistanceMetric::CosineSimilarity,
            DistanceMetric::Manhattan,
            DistanceMetric::Minkowski,
            DistanceMetric::Chebyshev,
            DistanceMetric::Hamming,
            DistanceMetric::Mahalanobis,
            DistanceMetric::Jaccard,
            DistanceMetric::Hellinger,
        ] {
            IvfPqSerializer::save_centroid_file(path, &centroids, 2, 1, metric, 1, 0.0).unwrap();
            let loaded = IvfPqSerializer::load_centroid_file(path).unwrap();
            assert_eq!(
                loaded.metric, metric,
                "Metric roundtrip failed for {:?}",
                metric
            );
        }
    }

    #[test]
    fn test_centroid_file_invalid_magic() {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"NOTCENTR12345678").unwrap();
        let res = IvfPqSerializer::load_centroid_file(tmp.path());
        assert!(res.is_err());
    }
}
