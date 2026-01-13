use bit_vec::BitVec;
use hashbrown::HashSet;
use parking_lot::{Mutex, RwLock};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{max, min};
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::config::{ConcurrentStats, HnswConfig, HnswStats};
use crate::global::{
    BATCH_SIZE, DEFAULT_K_MULTIPLIER, GROWTH_INCREMENT, INITIAL_CAPACITY,
    LOCAL_VISITED_SIZE_MULTIPLIER, MAX_EF_CONSTRUCTION, MAX_NEIGHBORS_PER_LAYER,
    MAX_NEIGHBORS_SEARCH_LAYER, OPTIMIZED_SEARCH_ITERATIONS, SEARCH_DISTANCE_THRESHOLD_MULTIPLIER,
    SEARCH_LAYER_ITERATIONS,
};
use crate::metrics::DistanceMetric;
use crate::node::{HnswNode, SearchResult};
use crate::storage::FlatVecStorage;

use std::cmp::Ordering;

/// Wrapper for SearchResult that reverses ordering for use in BinaryHeap
/// This allows BinaryHeap to work as a min-heap based on distance
#[derive(Debug, Clone)]
struct ReverseSearchResult(SearchResult);

impl PartialEq for ReverseSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for ReverseSearchResult {}

impl PartialOrd for ReverseSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.distance.partial_cmp(&self.0.distance)
    }
}

impl Ord for ReverseSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .distance
            .partial_cmp(&self.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Main Core Structure
pub struct HNSW<D: DistanceMetric> {
    config: HnswConfig,
    vec_storage: FlatVecStorage,
    nodes: Vec<Option<HnswNode>>,
    entry_point: Option<usize>,
    level_max: usize,
    max_level: usize,
    metric: D,
    rng: StdRng,
    deleted_bits: BitVec,
    next_id: usize,
    concurrent_stats: ConcurrentStats,
}

impl<D: DistanceMetric> HNSW<D> {
    /// Creates a new HNSW index with the given configuration and distance metric
    ///
    /// # Arguments
    /// * `config` - Configuration parameters for the HNSW graph
    /// * `metric` - Distance metric to use for vector comparisons
    ///
    /// # Returns
    /// A new HNSW instance
    pub fn new(config: HnswConfig, metric: D) -> Self {
        let max_elements = config.max_elements;
        let max_level = ((max_elements as f64).ln() * config.level_multiplier) as usize;
        let initial_capacity = min(INITIAL_CAPACITY, max_elements);
        Self {
            config,
            vec_storage: FlatVecStorage::new(0, initial_capacity),
            nodes: Vec::with_capacity(initial_capacity),
            entry_point: None,
            level_max: 0,
            max_level,
            metric,
            rng: StdRng::from_entropy(),
            deleted_bits: BitVec::from_elem(max_elements, false),
            next_id: 0,
            concurrent_stats: ConcurrentStats::default(),
        }
    }

    /// Converts the HNSW into a concurrent version for multi-threaded access
    pub fn into_concurrent(self) -> ConcurrentHNSW<D>
    where
        D: 'static + Send + Sync,
    {
        ConcurrentHNSW::new(self)
    }

    /// Generates a random level for a new node using exponential distribution
    fn get_random_level(&mut self) -> usize {
        let uniform: f64 = self.rng.r#gen();
        (-uniform.ln() * self.config.level_multiplier) as usize
    }

    /// Inserts a vector into the HNSW graph
    ///
    /// # Arguments
    /// * `vector` - The vector to insert
    ///
    /// # Returns
    /// The ID of the inserted node
    ///
    /// # Panics
    /// Panics if the maximum elements limit is exceeded
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        // Initialize storage dimension if it's the first vector
        if self.vec_storage.dim() == 0 {
            let dim = vector.len();
            let initial_capacity = min(INITIAL_CAPACITY, self.config.max_elements);
            self.vec_storage = FlatVecStorage::new(dim, initial_capacity);
        }
        // Get the next node ID
        let node_id = self.next_id;
        if node_id >= self.config.max_elements {
            panic!("Exceeded maximum elements limit");
        }
        // Add vector to storage with the same ID
        let storage_id = self.vec_storage.add_vector(&vector);
        let level = min(self.get_random_level(), self.max_level);
        let mut new_node = HnswNode::new(node_id, level);
        self.next_id += 1;
        if node_id >= self.nodes.len() {
            let new_size = min(
                max(node_id + 1, self.nodes.len() * 2),
                self.config.max_elements,
            );
            self.nodes.resize_with(new_size, || None);
        }
        if node_id >= self.deleted_bits.len() {
            self.deleted_bits.grow(node_id + GROWTH_INCREMENT, false);
        }
        if self.entry_point.is_none() {
            // For the first node, just add it and set as entry point
            self.nodes[node_id] = Some(new_node);
            self.entry_point = Some(node_id);
            self.level_max = level;
            return node_id;
        }
        let ep = self.entry_point.unwrap();
        let query_vector = self.vec_storage.get_vector(node_id); // Use node_id here
        let (mut curr_node, mut curr_level) = (ep, self.level_max);
        while curr_level > level {
            let best = { self.search_layer_best(query_vector, curr_node, curr_level) };
            if let Some(best_id) = best {
                curr_node = best_id;
            }
            curr_level -= 1;
        }
        let start_layer = min(level, self.level_max);
        let mut all_selected_neighbors = Vec::new();
        for curr_layer in (0..=start_layer).rev() {
            let neighbors = {
                self.search_layer_optimized(
                    query_vector,
                    curr_node,
                    min(self.config.ef_construction, MAX_EF_CONSTRUCTION),
                    curr_layer,
                )
            };
            if let Some(neighbors) = neighbors {
                let selected_neighbors: Vec<usize> = if neighbors.len() > self.config.m {
                    let mut neighbor_ids: Vec<(usize, f32)> =
                        neighbors.iter().map(|r| (r.id, r.distance)).collect();
                    neighbor_ids.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    neighbor_ids
                        .iter()
                        .take(self.config.m)
                        .map(|(id, _)| *id)
                        .collect()
                } else {
                    neighbors.iter().map(|r| r.id).collect()
                };
                all_selected_neighbors.push((curr_layer, selected_neighbors));
            }
        }
        for (curr_layer, selected_neighbors) in all_selected_neighbors {
            for &neighbor_id in &selected_neighbors {
                self.add_connection(node_id, neighbor_id, curr_layer);
                self.add_connection(neighbor_id, node_id, curr_layer);
            }
            for &neighbor_id in &selected_neighbors {
                self.prune_connections(neighbor_id, curr_layer);
            }
        }
        self.nodes[node_id] = Some(new_node);
        if level > self.level_max {
            self.level_max = level;
            self.entry_point = Some(node_id);
        }
        node_id
    }

    /// Inserts multiple vectors in batch
    ///
    /// # Arguments
    /// * `vectors` - List of vectors to insert
    ///
    /// # Returns
    /// List of IDs for the inserted nodes
    pub fn insert_batch(&mut self, vectors: Vec<Vec<f32>>) -> Vec<usize> {
        let vectors_len = vectors.len();
        let mut ids = Vec::with_capacity(vectors_len);
        let new_next_id = self.next_id + vectors_len;
        if new_next_id > self.nodes.len() {
            let new_size = min(
                max(new_next_id, self.nodes.len() * 2),
                self.config.max_elements,
            );
            self.nodes.resize_with(new_size, || None);
            self.deleted_bits.grow(new_size, false);
        }
        for vector in vectors {
            let id = self.insert(vector);
            ids.push(id);
        }
        self.concurrent_stats.parallel_inserts += 1;
        self.concurrent_stats.avg_batch_size =
            (self.concurrent_stats.avg_batch_size * 0.9 + vectors_len as f32 * 0.1);
        ids
    }

    /// Performs greedy search in a specific layer to find the nearest node
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `entry_point` - Starting point for the search
    /// * `layer` - Layer to search in
    ///
    /// # Returns
    /// ID of the nearest node found, or the entry_point if search fails
    fn search_layer_best(&self, query: &[f32], entry_point: usize, layer: usize) -> Option<usize> {
        if entry_point >= self.nodes.len() {
            return Some(entry_point);
        }
        let mut best_id = entry_point;
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let mut best_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut visited = HashSet::new();
        visited.insert(entry_point);
        let mut current = entry_point;
        for _ in 0..SEARCH_LAYER_ITERATIONS {
            let node = match self.get_node(current) {
                Some(node) => node,
                None => break,
            };
            if layer >= node.friends.len() {
                break;
            }
            let mut improved = false;
            let mut neighbor_count = 0;
            for &neighbor_id in node.friends[layer].iter() {
                if neighbor_count >= MAX_NEIGHBORS_PER_LAYER {
                    break;
                }
                if neighbor_id >= self.nodes.len()
                    || visited.contains(&neighbor_id)
                    || self.is_deleted(neighbor_id)
                {
                    continue;
                }
                neighbor_count += 1;
                visited.insert(neighbor_id);
                let neighbor_vector = self.vec_storage.get_vector(neighbor_id);
                let dist_sq = self.metric.distance_squared_direct(query, neighbor_vector);
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_id = neighbor_id;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
            current = best_id;
        }
        Some(best_id)
    }

    /// Searches for nearest neighbors in a specific layer (optimized version with local visited array)
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `entry_point` - Starting point for the search
    /// * `ef` - Search width parameter (exploration factor)
    /// * `layer` - Layer to search in
    ///
    /// # Returns
    /// List of search results sorted by distance, or None if search fails
    fn search_layer_optimized(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Option<Vec<SearchResult>> {
        if entry_point >= self.nodes.len() || self.nodes[entry_point].is_none() {
            return None;
        }
        if ef == 0 {
            return None;
        }
        if entry_point >= 10000 {
            return self.search_layer_fallback(query, entry_point, ef, layer);
        }
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let entry_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut candidates = BinaryHeap::new();
        candidates.push(ReverseSearchResult(SearchResult::new(
            entry_point,
            entry_dist_sq,
        )));
        let visited_size = min(self.nodes.len(), 10000);
        let mut visited_local = vec![false; visited_size];
        visited_local[entry_point] = true;
        let mut results = BinaryHeap::new();
        results.push(SearchResult::new(entry_point, entry_dist_sq));
        for _ in 0..min(ef * 3, OPTIMIZED_SEARCH_ITERATIONS) {
            if let Some(candidate) = candidates.pop() {
                let current = candidate.0;
                if results.len() >= ef {
                    let worst_distance = results.peek().map_or(f32::INFINITY, |r| r.distance);
                    if current.distance > worst_distance * SEARCH_DISTANCE_THRESHOLD_MULTIPLIER {
                        continue;
                    }
                }
                let node = match self.get_node(current.id) {
                    Some(node) => node,
                    None => continue,
                };
                if layer >= node.friends.len() {
                    continue;
                }
                let mut neighbor_count = 0;
                for &neighbor_id in node.friends[layer].iter() {
                    if neighbor_count >= MAX_NEIGHBORS_SEARCH_LAYER {
                        break;
                    }
                    if neighbor_id >= self.nodes.len() || self.is_deleted(neighbor_id) {
                        continue;
                    }
                    if neighbor_id < visited_local.len() {
                        if visited_local[neighbor_id] {
                            continue;
                        }
                        visited_local[neighbor_id] = true;
                    } else {
                        continue;
                    }
                    neighbor_count += 1;
                    let neighbor_vector = self.vec_storage.get_vector(neighbor_id);
                    let dist_sq = self.metric.distance_squared_direct(query, neighbor_vector);
                    candidates.push(ReverseSearchResult(SearchResult::new(neighbor_id, dist_sq)));
                    if results.len() < ef {
                        results.push(SearchResult::new(neighbor_id, dist_sq));
                    } else if dist_sq < results.peek().unwrap().distance {
                        results.pop();
                        results.push(SearchResult::new(neighbor_id, dist_sq));
                    }
                    if candidates.len() > ef * 2 {
                        candidates.pop();
                    }
                }
            } else {
                break;
            }
        }
        let mut final_results: Vec<SearchResult> = results.into_vec();
        final_results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if final_results.len() > ef {
            final_results.truncate(ef);
        }
        Some(final_results)
    }

    /// Searches for nearest neighbors in a specific layer (fallback version with HashSet)
    /// Used when the node ID is too large for the local visited array or when local array fails
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `entry_point` - Starting point for the search
    /// * `ef` - Search width parameter (exploration factor)
    /// * `layer` - Layer to search in
    ///
    /// # Returns
    /// List of search results sorted by distance
    fn search_layer_fallback(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Option<Vec<SearchResult>> {
        if entry_point >= self.nodes.len() || self.nodes[entry_point].is_none() {
            return None;
        }
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let entry_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::with_capacity(ef * 2 + 100);
        visited.insert(entry_point);
        candidates.push(ReverseSearchResult(SearchResult::new(
            entry_point,
            entry_dist_sq,
        )));
        let mut results = vec![SearchResult::new(entry_point, entry_dist_sq)];
        for _ in 0..min(ef * 3, OPTIMIZED_SEARCH_ITERATIONS) {
            if let Some(candidate) = candidates.pop() {
                let current = candidate.0;
                if results.len() < ef {
                    if !results.iter().any(|r| r.id == current.id) {
                        results.push(current.clone());
                    }
                } else {
                    let worst_idx = results
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap())
                        .map(|(idx, _)| idx);
                    if let Some(worst_idx) = worst_idx {
                        if current.distance < results[worst_idx].distance {
                            results[worst_idx] = current.clone();
                        }
                    }
                }
                let node = match self.get_node(current.id) {
                    Some(node) => node,
                    None => continue,
                };
                if layer >= node.friends.len() {
                    continue;
                }
                for &neighbor_id in node.friends[layer].iter().take(MAX_NEIGHBORS_SEARCH_LAYER) {
                    if neighbor_id >= self.nodes.len()
                        || visited.contains(&neighbor_id)
                        || self.is_deleted(neighbor_id)
                    {
                        continue;
                    }
                    visited.insert(neighbor_id);
                    let neighbor_vector = self.vec_storage.get_vector(neighbor_id);
                    let dist_sq = self.metric.distance_squared_direct(query, neighbor_vector);
                    candidates.push(ReverseSearchResult(SearchResult::new(neighbor_id, dist_sq)));
                    if candidates.len() > ef * 2 {
                        candidates.pop();
                    }
                }
            } else {
                break;
            }
        }
        results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if results.len() > ef {
            results.truncate(ef);
        }
        Some(results)
    }

    /// Adds a connection between two nodes at a specific layer
    ///
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    /// * `layer` - Layer to add the connection in
    fn add_connection(&mut self, from: usize, to: usize, layer: usize) {
        if let Some(node) = self.nodes[from].as_mut() {
            if layer >= node.friends.len() {
                node.friends.resize(layer + 1, SmallVec::new());
            }
            if !node.friends[layer].contains(&to) {
                node.friends[layer].push(to);
            }
        }
    }

    /// Prunes connections for a node at a specific layer to maintain degree limits
    ///
    /// # Arguments
    /// * `node_id` - Node ID to prune connections for
    /// * `layer` - Layer to prune connections in
    fn prune_connections(&mut self, node_id: usize, layer: usize) {
        if let Some(node) = self.nodes[node_id].as_mut() {
            if layer >= node.friends.len() {
                return;
            }
            let current_len = node.friends[layer].len();
            let max_connections = if layer == 0 {
                self.config.m_max_0
            } else {
                self.config.m_max
            };
            if current_len > max_connections {
                use rand::seq::SliceRandom;
                use rand::thread_rng;
                let mut friends = std::mem::take(&mut node.friends[layer]);
                if friends.len() > max_connections {
                    friends.shuffle(&mut thread_rng());
                    friends.truncate(max_connections);
                }
                node.friends[layer] = friends;
            }
        }
    }

    /// Searches for k nearest neighbors to a query vector
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// List of k nearest neighbors sorted by distance
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }
        let ep = self.entry_point.unwrap();
        if ep >= self.nodes.len() {
            return Vec::new();
        }
        let mut curr_node = ep;
        let mut curr_level = self.level_max;
        while curr_level > 0 {
            if let Some(best) = self.search_layer_best(query, curr_node, curr_level) {
                curr_node = best;
            }
            curr_level -= 1;
        }
        let ef_search = max(self.config.ef_search, k * 10);
        let ef_search = max(ef_search, 100);
        let mut results = self
            .search_layer_optimized(query, curr_node, ef_search, 0)
            .unwrap_or_else(|| Vec::new());
        if results.is_empty() && ep < self.nodes.len() && !self.is_deleted(ep) {
            let ep_vector = self.vec_storage.get_vector(ep);
            let dist_sq = self.metric.distance_squared_direct(query, ep_vector);
            results.push(SearchResult::new(ep, dist_sq));
        }
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| !self.is_deleted(r.id))
            .map(|mut r| {
                r.distance = r.distance.sqrt();
                r
            })
            .collect();
        final_results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if final_results.len() < k && self.nodes.len() > 0 {
            let mut rng = rand::thread_rng();
            let mut attempts = 0;
            while final_results.len() < k && attempts < 100 {
                let random_id = rng.gen_range(0..self.nodes.len());
                if !final_results.iter().any(|r| r.id == random_id)
                    && !self.is_deleted(random_id)
                    && random_id < self.nodes.len()
                    && self.nodes[random_id].is_some()
                {
                    let random_vector = self.vec_storage.get_vector(random_id);
                    let dist_sq = self.metric.distance_squared_direct(query, random_vector);
                    final_results.push(SearchResult::new(random_id, dist_sq.sqrt()));
                }
                attempts += 1;
            }
        }
        final_results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if final_results.len() > k {
            final_results.truncate(k);
        }
        final_results
    }

    /// Searches for k nearest neighbors for multiple query vectors in parallel
    ///
    /// # Arguments
    /// * `queries` - List of query vectors
    /// * `k` - Number of nearest neighbors to return for each query
    ///
    /// # Returns
    /// List of results for each query
    pub fn search_knn_batch(&mut self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<SearchResult>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let results: Vec<Vec<SearchResult>> = queries
            .par_iter()
            .with_min_len(self.config.batch_size)
            .map(|query| self.search_knn(query, k))
            .collect();
        self.concurrent_stats.parallel_searches += queries.len();
        results
    }

    /// Marks a node as deleted and removes its connections
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to delete
    ///
    /// # Returns
    /// true if the node was successfully deleted, false otherwise
    pub fn delete(&mut self, node_id: usize) -> bool {
        if node_id >= self.nodes.len() || self.nodes[node_id].is_none() {
            return false;
        }
        if node_id >= self.deleted_bits.len() {
            self.deleted_bits.grow(node_id + 1, false);
        }
        self.deleted_bits.set(node_id, true);
        let mut was_entry_point = false;
        let mut neighbors_to_update = Vec::new();
        {
            if let Some(node) = self.nodes[node_id].as_mut() {
                node.deleted = true;
                was_entry_point = Some(node_id) == self.entry_point;

                for (layer, friends) in node.friends.iter().enumerate() {
                    for &neighbor_id in friends {
                        neighbors_to_update.push((layer, neighbor_id));
                    }
                }
            } else {
                return false;
            }
        }
        for (layer, neighbor_id) in neighbors_to_update {
            if neighbor_id < self.nodes.len() {
                if let Some(neighbor) = self.nodes[neighbor_id].as_mut() {
                    if layer < neighbor.friends.len() {
                        neighbor.friends[layer].retain(|id| *id != node_id);
                    }
                }
            }
        }
        self.vec_storage.free_slot(node_id);
        if was_entry_point {
            self.update_entry_point();
        }
        true
    }

    /// Updates the entry point after a deletion if the current entry point was deleted
    fn update_entry_point(&mut self) {
        self.entry_point = None;
        self.level_max = 0;
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(node) = node {
                if !node.deleted && node.level > self.level_max {
                    self.level_max = node.level;
                    self.entry_point = Some(i);
                }
            }
        }
    }

    /// Retrieves a node by ID
    ///
    /// # Arguments
    /// * `id` - Node ID
    ///
    /// # Returns
    /// Reference to the node if it exists, None otherwise
    fn get_node(&self, id: usize) -> Option<&HnswNode> {
        self.nodes.get(id).and_then(|n| n.as_ref())
    }

    /// Checks if a node has been marked as deleted
    ///
    /// # Arguments
    /// * `id` - Node ID to check
    ///
    /// # Returns
    /// true if the node is deleted, false otherwise
    pub fn is_deleted(&self, id: usize) -> bool {
        id < self.deleted_bits.len() && self.deleted_bits[id]
    }

    /// Returns statistics about the HNSW graph
    ///
    /// # Returns
    /// HnswStats structure containing various metrics about the graph
    pub fn stats(&self) -> HnswStats {
        let mut node_count = 0;
        let mut avg_connections = 0.0;
        let mut max_connections = 0;
        let mut total_connections = 0;
        for node in self.nodes.iter().flatten() {
            if node.deleted {
                continue;
            }
            node_count += 1;
            let connections: usize = node.friends.iter().map(|f| f.len()).sum();
            total_connections += connections;
            max_connections = max(max_connections, connections);
        }
        if node_count > 0 {
            avg_connections = total_connections as f32 / node_count as f32;
        }
        HnswStats {
            node_count,
            max_level: self.level_max,
            entry_point: self.entry_point,
            avg_connections,
            max_connections,
            deleted_count: self.deleted_bits.iter().filter(|&b| b).count(),
            vector_dim: self.vec_storage.dim(),
            storage_size: self.vec_storage.data.len(),
            concurrent_stats: self.concurrent_stats.clone(),
        }
    }

    /// Returns the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vec_storage.len()
    }

    /// Checks if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the dimensionality of vectors in the index
    pub fn dim(&self) -> usize {
        self.vec_storage.dim()
    }

    /// Saves the HNSW index to a file (not implemented)
    ///
    /// # Arguments
    /// * `path` - File path to save to
    ///
    /// # Returns
    /// Ok(()) on success, error otherwise
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        Ok(())
    }

    /// Loads an HNSW index from a file (not implemented)
    ///
    /// # Arguments
    /// * `path` - File path to load from
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    /// Loaded HNSW instance or error
    pub fn load(path: &str, metric: D) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        ))
    }
}

/// Thread-safe concurrent version of HNSW for multi-threaded access
pub struct ConcurrentHNSW<D>
where
    D: DistanceMetric + 'static + Send + Sync,
{
    inner: Arc<RwLock<HNSW<D>>>,
    write_queue: Mutex<Vec<Vec<f32>>>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl<D: DistanceMetric + 'static + Send + Sync> ConcurrentHNSW<D> {
    /// Creates a new concurrent HNSW from an existing HNSW instance
    pub fn new(hnsw: HNSW<D>) -> Self {
        let inner = Arc::new(RwLock::new(hnsw));
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let write_queue = Arc::new(Mutex::new(Vec::new()));
        let inner_clone = inner.clone();
        let stop_flag_clone = stop_flag.clone();
        let write_queue_clone = write_queue.clone();
        let worker_handle = std::thread::spawn(move || {
            Self::background_worker(inner_clone, stop_flag_clone, write_queue_clone);
        });
        Self {
            inner,
            write_queue: Mutex::new(Vec::new()),
            worker_handle: Some(worker_handle),
            stop_flag,
        }
    }

    /// Background worker thread that processes asynchronous insertions
    fn background_worker(
        inner: Arc<RwLock<HNSW<D>>>,
        stop_flag: Arc<std::sync::atomic::AtomicBool>,
        write_queue: Arc<Mutex<Vec<Vec<f32>>>>,
    ) {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        while !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let mut queue = write_queue.lock();
            let remaining = BATCH_SIZE - batch.len();
            if remaining > 0 && !queue.is_empty() {
                let take_count = std::cmp::min(remaining, queue.len());
                let taken = queue.drain(..take_count).collect::<Vec<_>>();
                batch.extend(taken);
            }
            drop(queue);
            if batch.len() >= BATCH_SIZE {
                let mut guard = inner.write();
                guard.insert_batch(std::mem::take(&mut batch));
            }
        }
        let mut queue = write_queue.lock();
        if !batch.is_empty() {
            let mut guard = inner.write();
            guard.insert_batch(batch);
        }
        if !queue.is_empty() {
            let mut guard = inner.write();
            guard.insert_batch(std::mem::take(&mut queue));
        }
    }

    /// Searches for k nearest neighbors (thread-safe)
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let guard = self.inner.read();
        guard.search_knn(query, k)
    }

    /// Searches for k nearest neighbors for multiple queries (thread-safe)
    pub fn search_knn_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<SearchResult>> {
        let mut guard = self.inner.write();
        guard.search_knn_batch(queries, k)
    }

    /// Asynchronously inserts a vector (adds to write queue for background processing)
    pub fn insert_async(&self, vector: Vec<f32>) {
        let mut queue = self.write_queue.lock();
        queue.push(vector);
    }

    /// Returns statistics about the concurrent HNSW graph
    pub fn stats(&self) -> HnswStats {
        let guard = self.inner.read();
        guard.stats()
    }

    /// Stops the background worker thread
    pub fn stop(&mut self) {
        self.stop_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

impl<D> Drop for ConcurrentHNSW<D>
where
    D: DistanceMetric + 'static + Send + Sync,
{
    fn drop(&mut self) {
        self.stop();
    }
}
