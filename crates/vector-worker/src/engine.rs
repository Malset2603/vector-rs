//! Worker search engine backend managing local shard storage and indices.

use std::path::Path;

use vector_index::DistanceMetric;
use vector_index::hnsw::{HnswConfig, HnswIndex};
use vector_index::ivf::{IvfPqConfig, IvfPqIndex};
use vector_index::storage::{HeapStorage, MmapStorage, VectorStorage};
use vector_index::types::Result;
use vector_proto::SearchResultItem;

/// Search storage backend variant for local vector search.
pub enum StorageBackend {
    /// In-memory heap storage.
    Heap(HeapStorage),
    /// Memory-mapped file storage.
    Mmap(MmapStorage),
}

impl VectorStorage for StorageBackend {
    fn dimension(&self) -> usize {
        match self {
            StorageBackend::Heap(s) => s.dimension(),
            StorageBackend::Mmap(s) => s.dimension(),
        }
    }

    fn len(&self) -> usize {
        match self {
            StorageBackend::Heap(s) => s.len(),
            StorageBackend::Mmap(s) => s.len(),
        }
    }

    fn get(&self, id: u32) -> &[f32] {
        match self {
            StorageBackend::Heap(s) => s.get(id),
            StorageBackend::Mmap(s) => s.get(id),
        }
    }

    fn as_raw_slice(&self) -> &[f32] {
        match self {
            StorageBackend::Heap(s) => s.as_raw_slice(),
            StorageBackend::Mmap(s) => s.as_raw_slice(),
        }
    }
}

/// Index backend variant supporting either HNSW or IVF-PQ index.
pub enum IndexBackend {
    /// Hierarchical Navigable Small World graph index.
    Hnsw(HnswIndex<StorageBackend>),
    /// Inverted File with Product Quantization index.
    IvfPq(IvfPqIndex<StorageBackend>),
}

/// Local shard search engine for worker nodes.
pub struct WorkerEngine {
    shard_id: u32,
    backend: IndexBackend,
}

impl WorkerEngine {
    /// Creates a new `WorkerEngine` with an HNSW index from an in-memory `HeapStorage`.
    pub fn from_heap(
        shard_id: u32,
        storage: HeapStorage,
        config: HnswConfig,
        metric: DistanceMetric,
    ) -> Self {
        let index =
            HnswIndex::build_parallel_with_config(StorageBackend::Heap(storage), config, metric);
        Self {
            shard_id,
            backend: IndexBackend::Hnsw(index),
        }
    }

    /// Creates a `WorkerEngine` with an HNSW index from an `MmapStorage` and optional graph file.
    pub fn from_mmap<P: AsRef<Path>>(
        shard_id: u32,
        mmap_path: P,
        graph_path: Option<P>,
        config: HnswConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let mmap_storage = MmapStorage::open(mmap_path)?;
        let storage = StorageBackend::Mmap(mmap_storage);

        let index = if let Some(g_path) = graph_path {
            HnswIndex::load_graph(storage, g_path, metric)?
        } else {
            HnswIndex::build_parallel_with_config(storage, config, metric)
        };

        Ok(Self {
            shard_id,
            backend: IndexBackend::Hnsw(index),
        })
    }

    /// Creates a `WorkerEngine` with an IVF-PQ index from an in-memory `HeapStorage`.
    pub fn from_heap_ivf_pq(
        shard_id: u32,
        storage: HeapStorage,
        config: IvfPqConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let index = IvfPqIndex::build_with_config(StorageBackend::Heap(storage), config, metric)?;
        Ok(Self {
            shard_id,
            backend: IndexBackend::IvfPq(index),
        })
    }

    /// Creates a `WorkerEngine` with an IVF-PQ index from an `MmapStorage` and optional index file.
    pub fn from_mmap_ivf_pq<P: AsRef<Path>>(
        shard_id: u32,
        mmap_path: P,
        ivf_path: Option<P>,
        config: IvfPqConfig,
        metric: DistanceMetric,
    ) -> Result<Self> {
        let mmap_storage = MmapStorage::open(mmap_path)?;
        let storage = StorageBackend::Mmap(mmap_storage);

        let index = if let Some(path) = ivf_path {
            IvfPqIndex::load_from_file(storage, path)?
        } else {
            IvfPqIndex::build_with_config(storage, config, metric)?
        };

        Ok(Self {
            shard_id,
            backend: IndexBackend::IvfPq(index),
        })
    }

    /// Creates a `WorkerEngine` from an already constructed `HnswIndex`.
    pub fn from_hnsw_index(shard_id: u32, index: HnswIndex<StorageBackend>) -> Self {
        Self {
            shard_id,
            backend: IndexBackend::Hnsw(index),
        }
    }

    /// Creates a `WorkerEngine` from an already constructed `IvfPqIndex`.
    pub fn from_ivf_pq_index(shard_id: u32, index: IvfPqIndex<StorageBackend>) -> Self {
        Self {
            shard_id,
            backend: IndexBackend::IvfPq(index),
        }
    }

    /// Returns the shard identifier.
    pub fn shard_id(&self) -> u32 {
        self.shard_id
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        match &self.backend {
            IndexBackend::Hnsw(idx) => idx.storage().dimension(),
            IndexBackend::IvfPq(idx) => idx.storage().dimension(),
        }
    }

    /// Returns the total number of vectors in this shard.
    pub fn num_vectors(&self) -> usize {
        match &self.backend {
            IndexBackend::Hnsw(idx) => idx.storage().len(),
            IndexBackend::IvfPq(idx) => idx.storage().len(),
        }
    }

    /// Returns the index algorithm type name ("HNSW" or "IVF-PQ").
    pub fn index_type(&self) -> &str {
        match &self.backend {
            IndexBackend::Hnsw(_) => "HNSW",
            IndexBackend::IvfPq(_) => "IVF-PQ",
        }
    }

    /// Returns the distance metric used by the index backend.
    pub fn metric(&self) -> DistanceMetric {
        match &self.backend {
            IndexBackend::Hnsw(idx) => idx.metric(),
            IndexBackend::IvfPq(idx) => idx.metric(),
        }
    }

    /// Searches for top-K nearest neighbors using the active index backend.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        search_param: usize,
    ) -> Result<Vec<SearchResultItem>> {
        let results = match &self.backend {
            IndexBackend::Hnsw(idx) => {
                let ef_search = if search_param > 0 {
                    search_param
                } else {
                    idx.config().ef_search
                };
                idx.search(query, k, ef_search)?
            }
            IndexBackend::IvfPq(idx) => {
                let nprobe = if search_param > 0 {
                    search_param
                } else {
                    idx.config().nprobe
                };
                idx.search_with_nprobe(query, k, nprobe)?
            }
        };

        let proto_items = results
            .into_iter()
            .map(|r| SearchResultItem {
                id: r.id,
                distance: r.distance,
                shard_id: self.shard_id,
            })
            .collect();

        Ok(proto_items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_engine_heap_search() {
        let mut storage = HeapStorage::new(3);
        storage.push(&[1.0, 0.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0, 0.0]).unwrap();
        storage.push(&[0.0, 0.0, 1.0]).unwrap();

        let engine =
            WorkerEngine::from_heap(1, storage, HnswConfig::default(), DistanceMetric::L2Squared);

        assert_eq!(engine.shard_id(), 1);
        assert_eq!(engine.dimension(), 3);
        assert_eq!(engine.num_vectors(), 3);
        assert_eq!(engine.index_type(), "HNSW");

        let query = [1.0, 0.1, 0.0];
        let items = engine.search(&query, 2, 50).unwrap();

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, 0);
        assert_eq!(items[0].shard_id, 1);
    }

    #[test]
    fn test_worker_engine_ivf_pq_search() {
        let mut storage = HeapStorage::new(4);
        for i in 0..20 {
            let val = (i as f32) * 0.1;
            storage
                .push(&[val, val + 0.1, val + 0.2, val + 0.3])
                .unwrap();
        }

        let config = IvfPqConfig::new(2, 2, 2).with_sub_clusters(4);
        let engine =
            WorkerEngine::from_heap_ivf_pq(2, storage, config, DistanceMetric::L2Squared).unwrap();

        assert_eq!(engine.shard_id(), 2);
        assert_eq!(engine.dimension(), 4);
        assert_eq!(engine.num_vectors(), 20);
        assert_eq!(engine.index_type(), "IVF-PQ");

        let items = engine.search(&[0.0, 0.1, 0.2, 0.3], 3, 2).unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].shard_id, 2);
    }

    #[test]
    fn test_worker_engine_dimension_mismatch() {
        let storage = HeapStorage::new(3);
        let engine =
            WorkerEngine::from_heap(1, storage, HnswConfig::default(), DistanceMetric::L2Squared);

        let query = [1.0, 2.0]; // 2D query for 3D engine
        let result = engine.search(&query, 2, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_worker_engine_mmap_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("worker_mmap.bin");

        let flat_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        MmapStorage::create_from_flat(&path, 2, &flat_data).unwrap();

        let engine = WorkerEngine::from_mmap(
            5,
            &path,
            None,
            HnswConfig::default(),
            DistanceMetric::L2Squared,
        )
        .unwrap();

        assert_eq!(engine.shard_id(), 5);
        assert_eq!(engine.dimension(), 2);
        assert_eq!(engine.num_vectors(), 3);

        let results = engine.search(&[1.0, 0.0], 1, 50).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert_eq!(results[0].shard_id, 5);
    }

    #[test]
    fn test_worker_engine_similarity_metrics() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 0.0]).unwrap();
        storage.push(&[0.0, 1.0]).unwrap();

        let engine = WorkerEngine::from_heap(
            2,
            storage,
            HnswConfig::default(),
            DistanceMetric::DotProduct,
        );

        let results = engine.search(&[1.0, 0.0], 1, 50).unwrap();
        assert_eq!(results[0].id, 0);
        assert!((results[0].distance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_storage_backend_methods() {
        let mut heap = HeapStorage::new(2);
        heap.push(&[1.0, 2.0]).unwrap();
        heap.push(&[3.0, 4.0]).unwrap();
        let backend_heap = StorageBackend::Heap(heap);

        assert_eq!(backend_heap.dimension(), 2);
        assert_eq!(backend_heap.len(), 2);
        assert_eq!(backend_heap.get(0), &[1.0, 2.0]);
        assert_eq!(backend_heap.as_raw_slice(), &[1.0, 2.0, 3.0, 4.0]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_backend_mmap.bin");
        MmapStorage::create_from_flat(&path, 2, &[5.0, 6.0]).unwrap();
        let mmap = MmapStorage::open(&path).unwrap();
        let backend_mmap = StorageBackend::Mmap(mmap);

        assert_eq!(backend_mmap.dimension(), 2);
        assert_eq!(backend_mmap.len(), 1);
        assert_eq!(backend_mmap.get(0), &[5.0, 6.0]);
        assert_eq!(backend_mmap.as_raw_slice(), &[5.0, 6.0]);
    }

    #[test]
    fn test_worker_engine_direct_constructors() {
        let mut storage = HeapStorage::new(2);
        storage.push(&[1.0, 0.0]).unwrap();

        // HNSW direct
        let hnsw_idx = HnswIndex::build_parallel_with_config(
            StorageBackend::Heap(storage.clone()),
            HnswConfig::default(),
            DistanceMetric::L2Squared,
        );
        let engine_hnsw = WorkerEngine::from_hnsw_index(7, hnsw_idx);
        assert_eq!(engine_hnsw.shard_id(), 7);
        assert_eq!(engine_hnsw.index_type(), "HNSW");

        // IVF-PQ direct
        let ivf_cfg = IvfPqConfig::new(1, 1, 1).with_sub_clusters(1);
        let ivf_idx = IvfPqIndex::build_with_config(
            StorageBackend::Heap(storage),
            ivf_cfg,
            DistanceMetric::L2Squared,
        )
        .unwrap();
        let engine_ivf = WorkerEngine::from_ivf_pq_index(8, ivf_idx);
        assert_eq!(engine_ivf.shard_id(), 8);
        assert_eq!(engine_ivf.index_type(), "IVF-PQ");
    }

    #[test]
    fn test_worker_engine_mmap_ivf_pq() {
        let dir = tempfile::tempdir().unwrap();
        let mmap_path = dir.path().join("ivf_mmap.bin");
        let ivf_path = dir.path().join("ivf_index.bin");

        let flat_data = vec![1.0, 0.0, 0.0, 1.0];
        MmapStorage::create_from_flat(&mmap_path, 2, &flat_data).unwrap();

        let config = IvfPqConfig::new(1, 1, 1).with_sub_clusters(2);
        // Build & save to ivf_path
        let storage_tmp = StorageBackend::Mmap(MmapStorage::open(&mmap_path).unwrap());
        let ivf_idx =
            IvfPqIndex::build_with_config(storage_tmp, config.clone(), DistanceMetric::L2Squared)
                .unwrap();
        ivf_idx.save_to_file(&ivf_path).unwrap();

        // Load via from_mmap_ivf_pq
        let engine = WorkerEngine::from_mmap_ivf_pq(
            12,
            &mmap_path,
            Some(&ivf_path),
            config,
            DistanceMetric::L2Squared,
        )
        .unwrap();

        assert_eq!(engine.shard_id(), 12);
        assert_eq!(engine.dimension(), 2);
        assert_eq!(engine.num_vectors(), 2);
        assert_eq!(engine.index_type(), "IVF-PQ");

        let results = engine.search(&[1.0, 0.0], 1, 0).unwrap(); // search_param = 0 -> default
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }
}
