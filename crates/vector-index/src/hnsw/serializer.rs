//! Binary serialization and deserialization for HNSW graph topology.
//!
//! Allows persisting the built index graph to disk and loading it back
//! with zero-overhead binary decoding.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;

use super::config::HnswConfig;
use super::graph::{HnswGraph, NodeLinks};
use crate::types::{Result, VectorIndexError};

/// Magic identifier for VectorRS HNSW graph binary format ("HNSWVECR").
pub const HNSW_MAGIC: u64 = 0x484E_5357_5645_4352;

/// File format version.
pub const HNSW_VERSION: u32 = 1;

/// Serializer for saving and loading HNSW graph structures.
pub struct HnswSerializer;

impl HnswSerializer {
    /// Saves the given `HnswGraph` to a binary file at `path`.
    pub fn save_to_file<P: AsRef<Path>>(graph: &HnswGraph, path: P) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(&HNSW_MAGIC.to_le_bytes())?;
        writer.write_all(&HNSW_VERSION.to_le_bytes())?;

        // Write config
        writer.write_all(&(graph.config.m as u32).to_le_bytes())?;
        writer.write_all(&(graph.config.m0 as u32).to_le_bytes())?;
        writer.write_all(&(graph.config.ef_construction as u32).to_le_bytes())?;
        writer.write_all(&(graph.config.ef_search as u32).to_le_bytes())?;
        writer.write_all(&(if graph.config.use_heuristic { 1u8 } else { 0u8 }).to_le_bytes())?;

        // Write state
        let max_level = *graph.max_level.read() as u32;
        writer.write_all(&max_level.to_le_bytes())?;

        let ep = graph.entry_point.read().unwrap_or(u32::MAX);
        writer.write_all(&ep.to_le_bytes())?;

        let num_nodes = graph.nodes.len() as u64;
        writer.write_all(&num_nodes.to_le_bytes())?;

        // Write nodes
        for node in &graph.nodes {
            let max_l = node.max_level() as u32;
            writer.write_all(&max_l.to_le_bytes())?;

            for layer_lock in &node.layers {
                let neighbors = layer_lock.read();
                let count = neighbors.len() as u32;
                writer.write_all(&count.to_le_bytes())?;

                for &n_id in neighbors.iter() {
                    writer.write_all(&n_id.to_le_bytes())?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Loads an `HnswGraph` from a binary file at `path`.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<HnswGraph> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];
        let mut buf1 = [0u8; 1];

        // Check magic
        reader.read_exact(&mut buf8)?;
        let magic = u64::from_le_bytes(buf8);
        if magic != HNSW_MAGIC {
            return Err(VectorIndexError::InvalidHeader {
                path: PathBuf::from(path),
                reason: format!(
                    "invalid HNSW magic: 0x{magic:016X} (expected 0x{HNSW_MAGIC:016X})"
                ),
            });
        }

        // Check version
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != HNSW_VERSION {
            return Err(VectorIndexError::InvalidHeader {
                path: PathBuf::from(path),
                reason: format!("unsupported HNSW version: {version}"),
            });
        }

        // Read config
        reader.read_exact(&mut buf4)?;
        let m = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let m0 = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let ef_construction = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let ef_search = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf1)?;
        let use_heuristic = buf1[0] == 1;

        let config = HnswConfig::new(m, ef_construction, ef_search)
            .with_m0(m0)
            .with_heuristic(use_heuristic);

        // Read state
        reader.read_exact(&mut buf4)?;
        let max_level = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let ep_raw = u32::from_le_bytes(buf4);
        let entry_point = if ep_raw == u32::MAX {
            None
        } else {
            Some(ep_raw)
        };

        reader.read_exact(&mut buf8)?;
        let num_nodes = u64::from_le_bytes(buf8) as usize;

        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            reader.read_exact(&mut buf4)?;
            let max_l = u32::from_le_bytes(buf4) as usize;

            let mut layers = Vec::with_capacity(max_l + 1);
            for _ in 0..=max_l {
                reader.read_exact(&mut buf4)?;
                let count = u32::from_le_bytes(buf4) as usize;

                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    reader.read_exact(&mut buf4)?;
                    neighbors.push(u32::from_le_bytes(buf4));
                }
                layers.push(RwLock::new(neighbors));
            }
            nodes.push(NodeLinks { layers });
        }

        Ok(HnswGraph {
            config,
            nodes,
            entry_point: RwLock::new(entry_point),
            max_level: RwLock::new(max_level),
            num_nodes: AtomicUsize::new(num_nodes),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializer_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_hnsw.bin");

        let config = HnswConfig::new(8, 50, 30);
        let mut graph = HnswGraph::new(config.clone(), 3);
        graph.init_node(0, 1);
        graph.init_node(1, 0);
        graph.init_node(2, 0);

        *graph.entry_point.write() = Some(0);
        *graph.max_level.write() = 1;

        // Add dummy edges
        graph.nodes[0].layers[0].write().push(1);
        graph.nodes[0].layers[0].write().push(2);
        graph.nodes[1].layers[0].write().push(0);
        graph.nodes[2].layers[0].write().push(0);

        HnswSerializer::save_to_file(&graph, &path).unwrap();

        let loaded = HnswSerializer::load_from_file(&path).unwrap();
        assert_eq!(loaded.config, config);
        assert_eq!(*loaded.entry_point.read(), Some(0));
        assert_eq!(*loaded.max_level.read(), 1);
        assert_eq!(loaded.nodes.len(), 3);

        assert_eq!(*loaded.nodes[0].layers[0].read(), vec![1, 2]);
        assert_eq!(*loaded.nodes[1].layers[0].read(), vec![0]);
        assert_eq!(*loaded.nodes[2].layers[0].read(), vec![0]);
    }

    #[test]
    fn test_serializer_empty_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_hnsw.bin");

        let config = HnswConfig::default();
        let graph = HnswGraph::new(config.clone(), 0);

        HnswSerializer::save_to_file(&graph, &path).unwrap();

        let loaded = HnswSerializer::load_from_file(&path).unwrap();
        assert_eq!(loaded.config, config);
        assert_eq!(*loaded.entry_point.read(), None);
        assert_eq!(loaded.nodes.len(), 0);
    }

    #[test]
    fn test_serializer_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_magic.bin");

        // Write garbage magic
        std::fs::write(&path, [0u8; 32]).unwrap();

        let result = HnswSerializer::load_from_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("invalid HNSW magic"));
    }

    #[test]
    fn test_serializer_invalid_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_version.bin");

        let mut buf = Vec::new();
        buf.extend_from_slice(&HNSW_MAGIC.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version 999
        std::fs::write(&path, buf).unwrap();

        let result = HnswSerializer::load_from_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("unsupported HNSW version"));
    }

    #[test]
    fn test_serializer_non_existent_file() {
        let result = HnswSerializer::load_from_file("non_existent_path_12345.bin");
        assert!(result.is_err());
    }
}
