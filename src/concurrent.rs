use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Arc;

use crate::config::HnswConfig;
use crate::metrics::DistanceMetric;
use crate::hnsw::HNSW;

/// Thread pool for concurrent HNSW operations
pub struct HNSWThreadPool<D>
where
    D: DistanceMetric + Send + Sync + 'static,
{
    inner: Arc<RwLock<HNSW<D>>>,
    write_pool: rayon::ThreadPool,
    read_pool: rayon::ThreadPool,
    max_batch_size: usize,
}

impl<D> HNSWThreadPool<D>
where
    D: DistanceMetric + Send + Sync + 'static,
{
    /// Create a new thread pool for HNSW operations
    pub fn new(config: HnswConfig, metric: D) -> Self {
        let hnsw = HNSW::new(config.clone(), metric);
        let inner = Arc::new(RwLock::new(hnsw));
        let num_threads = if config.num_threads == 0 {
            std::thread::available_parallelism().unwrap().get()
        } else {
            config.num_threads
        };
        let write_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads / 2 + 1)
            .thread_name(|i| format!("hnsw-write-{}", i))
            .build()
            .unwrap();
        let read_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("hnsw-read-{}", i))
            .build()
            .unwrap();
        Self {
            inner,
            write_pool,
            read_pool,
            max_batch_size: config.batch_size,
        }
    }

    /// Parallel batch insert
    pub fn insert_batch(&self, vectors: Vec<Vec<f32>>) -> Vec<usize> {
        if vectors.is_empty() {
            return Vec::new();
        }
        let batch_size = self.max_batch_size;
        let batches: Vec<Vec<Vec<f32>>> = vectors
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        let mut all_ids = Vec::with_capacity(vectors.len());
        for batch in batches {
            let ids = self.write_pool.install(|| {
                let mut guard = self.inner.write();
                guard.insert_batch(batch)
            });
            all_ids.extend(ids);
        }
        all_ids
    }

    /// Parallel batch search
    pub fn search_knn_batch(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> Vec<Vec<crate::node::SearchResult>> {
        if queries.is_empty() {
            return Vec::new();
        }
        self.read_pool.install(|| {
            let mut guard = self.inner.write();
            guard.search_knn_batch(&queries, k)
        })
    }

    /// Parallel similarity search with threshold
    pub fn search_similarity_batch(
        &self,
        queries: Vec<Vec<f32>>,
        threshold: f32,
        k: usize,
    ) -> Vec<Vec<crate::node::SearchResult>> {
        self.read_pool.install(|| {
            queries
                .par_iter()
                .map(|query| {
                    let guard = self.inner.read();
                    let results = guard.search_knn(query, k);
                    results
                        .into_iter()
                        .filter(|r| r.distance <= threshold)
                        .collect()
                })
                .collect()
        })
    }

    /// Get read-only reference for custom parallel operations
    pub fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&HNSW<D>) -> R + Send,
        R: Send,
    {
        self.read_pool.install(|| {
            let guard = self.inner.read();
            f(&guard)
        })
    }

    /// Execute write operation with thread pool
    pub fn with_write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut HNSW<D>) -> R + Send,
        R: Send,
    {
        self.write_pool.install(|| {
            let mut guard = self.inner.write();
            f(&mut guard)
        })
    }
}

/// Concurrent builder for HNSW index
pub struct ConcurrentHNSWBuilder<D>
where
    D: DistanceMetric + Send + Sync + 'static,
{
    config: HnswConfig,
    metric: D,
    initial_vectors: Vec<Vec<f32>>,
}

impl<D> ConcurrentHNSWBuilder<D>
where
    D: DistanceMetric + Send + Sync + 'static,
{
    /// Create a new builder
    pub fn new(config: HnswConfig, metric: D) -> Self {
        Self {
            config,
            metric,
            initial_vectors: Vec::new(),
        }
    }

    /// Add initial vectors for batch construction
    pub fn with_initial_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        self.initial_vectors = vectors;
        self
    }

    /// Build the concurrent HNSW index
    pub fn build(self) -> HNSWThreadPool<D> {
        let pool = HNSWThreadPool::new(self.config, self.metric);
        if !self.initial_vectors.is_empty() {
            pool.insert_batch(self.initial_vectors);
        }
        pool
    }
}
