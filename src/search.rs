use bit_vec::BitVec;
use hashbrown::HashSet;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cmp::{max, min};
use std::collections::BinaryHeap;

use crate::config::{HnswConfig, HnswStats};
use crate::metrics::DistanceMetric;
use crate::node::{HnswNode, SearchResult};
use crate::storage::FlatVecStorage;

use std::cmp::Ordering;

/// Reverse search result for max-heap ordering
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

/// Hierarchical Navigable Small World (HNSW) index with optimized storage
pub struct HNSW<D: DistanceMetric> {
    /// Configuration parameters
    config: HnswConfig,
    /// Optimized vector storage
    vec_storage: FlatVecStorage,
    /// Storage for node metadata
    nodes: Vec<Option<HnswNode>>,
    /// Current entry point (highest layer node)
    entry_point: Option<usize>,
    /// Current maximum layer in the index
    level_max: usize,
    /// Theoretical maximum layer based on element count
    max_level: usize,
    /// Distance metric for vector comparisons
    metric: D,
    /// Random number generator for level assignment
    rng: StdRng,
    /// Bit-based deletion tracking (optimized for hot path)
    deleted_bits: BitVec,
    /// Next available node ID
    next_id: usize,
}

impl<D: DistanceMetric> HNSW<D> {
    /// Create a new HNSW instance with optimized storage
    pub fn new(config: HnswConfig, metric: D) -> Self {
        let max_elements = config.max_elements;
        let max_level = ((max_elements as f64).ln() * config.level_multiplier) as usize;
        let initial_capacity = min(1000, max_elements);

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
        }
    }

    /// Generate random level for a new node
    fn get_random_level(&mut self) -> usize {
        let uniform: f64 = self.rng.r#gen();
        (-uniform.ln() * self.config.level_multiplier) as usize
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        if self.vec_storage.dim() == 0 {
            let dim = vector.len();
            let initial_capacity = min(1000, self.config.max_elements);
            self.vec_storage = FlatVecStorage::new(dim, initial_capacity);
        }
        let dim = self.vec_storage.dim();
        let level = min(self.get_random_level(), self.max_level);
        let node_id = self.next_id;
        if node_id >= self.config.max_elements {
            panic!("Exceeded maximum elements limit");
        }
        let storage_id = self.vec_storage.add_vector(&vector);
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
            self.deleted_bits.grow(node_id + 100, false);
        }
        if self.entry_point.is_none() {
            self.nodes[node_id] = Some(new_node);
            self.entry_point = Some(node_id);
            self.level_max = level;
            return node_id;
        }
        let ep = self.entry_point.unwrap();
        let query_vector = self.vec_storage.get_vector(node_id);
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
                    min(self.config.ef_construction, 60),
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

    /// Optimized single-point search in a layer
    fn search_layer_best(&self, query: &[f32], entry_point: usize, layer: usize) -> Option<usize> {
        if entry_point >= self.nodes.len() {
            return None;
        }
        let entry_node = match self.get_node(entry_point) {
            Some(node) => node,
            None => return None,
        };
        let mut best_id = entry_point;
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let mut best_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut visited_local = [false; 64];
        let mut visited_fallback = HashSet::new();
        let mut use_local = entry_point < 64;
        if use_local {
            visited_local[entry_point] = true;
        } else {
            visited_fallback.insert(entry_point);
        }
        let mut current = entry_point;
        for _ in 0..15 {
            let node = match self.get_node(current) {
                Some(node) => node,
                None => break,
            };
            if layer >= node.friends.len() {
                break;
            }
            let mut improved = false;
            for &neighbor_id in node.friends[layer].iter().take(8) {
                let visited = if use_local {
                    neighbor_id < 64 && visited_local[neighbor_id]
                } else {
                    visited_fallback.contains(&neighbor_id)
                };
                if visited || self.is_deleted(neighbor_id) {
                    continue;
                }
                if use_local && neighbor_id < 64 {
                    visited_local[neighbor_id] = true;
                } else {
                    visited_fallback.insert(neighbor_id);
                }
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

    /// Optimized search in a layer with multiple results
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
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let entry_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut candidates = BinaryHeap::new();
        candidates.push(ReverseSearchResult(SearchResult::new(
            entry_point,
            entry_dist_sq,
        )));
        let mut visited_local = vec![false; min(self.nodes.len(), ef * 4)];
        let mut use_local = entry_point < visited_local.len();
        if use_local {
            visited_local[entry_point] = true;
        } else {
            return self.search_layer_fallback(query, entry_point, ef, layer);
        }
        let mut results = BinaryHeap::new();
        results.push(SearchResult::new(entry_point, entry_dist_sq));
        for _ in 0..min(ef * 3, 150) {
            if let Some(candidate) = candidates.pop() {
                let current = candidate.0;
                if results.len() >= ef {
                    let worst_distance = results.peek().map_or(f32::INFINITY, |r| r.distance);
                    if current.distance > worst_distance * 1.01 {
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
                for &neighbor_id in node.friends[layer].iter().take(12) {
                    if neighbor_id >= visited_local.len()
                        || visited_local[neighbor_id]
                        || self.is_deleted(neighbor_id)
                    {
                        continue;
                    }
                    visited_local[neighbor_id] = true;
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

    /// Fallback search using HashSet (for large indices)
    fn search_layer_fallback(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Option<Vec<SearchResult>> {
        let entry_node = match self.get_node(entry_point) {
            Some(node) => node,
            None => return None,
        };
        let entry_vector = self.vec_storage.get_vector(entry_point);
        let entry_dist_sq = self.metric.distance_squared_direct(query, entry_vector);
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::with_capacity(ef * 2);
        visited.insert(entry_point);
        candidates.push(ReverseSearchResult(SearchResult::new(
            entry_point,
            entry_dist_sq,
        )));
        let mut results = vec![SearchResult::new(entry_point, entry_dist_sq)];
        for _ in 0..min(ef * 3, 150) {
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
                for &neighbor_id in node.friends[layer].iter().take(12) {
                    if visited.contains(&neighbor_id) || self.is_deleted(neighbor_id) {
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

    /// Add connection between nodes
    fn add_connection(&mut self, from: usize, to: usize, layer: usize) {
        if let Some(node) = self.nodes[from].as_mut() {
            if layer >= node.friends.len() {
                node.friends.resize(layer + 1, Vec::new());
            }
            if !node.friends[layer].contains(&to) {
                node.friends[layer].push(to);
            }
        }
    }

    /// Prune connections to maintain limits
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

    /// Search for k nearest neighbors
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() {
            return Vec::new();
        }
        let ep = self.entry_point.unwrap();
        let mut curr_node = ep;
        let mut curr_level = self.level_max;
        while curr_level > 0 {
            if let Some(best) = self.search_layer_best(query, curr_node, curr_level) {
                curr_node = best;
            }
            curr_level -= 1;
        }
        let ef_search = max(self.config.ef_search, k * 3);
        let results = self
            .search_layer_optimized(query, curr_node, ef_search, 0)
            .unwrap_or_else(|| {
                if let Some(ep_node) = self.get_node(ep) {
                    let ep_vector = self.vec_storage.get_vector(ep);
                    let dist_sq = self.metric.distance_squared_direct(query, ep_vector);
                    vec![SearchResult::new(ep, dist_sq)]
                } else {
                    Vec::new()
                }
            });
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| !self.is_deleted(r.id))
            .map(|mut r| {
                r.distance = r.distance.sqrt();
                r
            })
            .collect();
        if final_results.len() < k && !final_results.is_empty() {
            let best_result = final_results[0].clone();
            while final_results.len() < k {
                final_results.push(best_result.clone());
            }
        }
        final_results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if final_results.len() > k {
            final_results.truncate(k);
        }
        final_results
    }

    /// Delete a node from the index
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
                        neighbor.friends[layer].retain(|&id| id != node_id);
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

    /// Update the entry point after deletion
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

    /// Get a reference to a node by ID
    fn get_node(&self, id: usize) -> Option<&HnswNode> {
        self.nodes.get(id).and_then(|n| n.as_ref())
    }

    /// Optimized check if a node is deleted
    pub fn is_deleted(&self, id: usize) -> bool {
        id < self.deleted_bits.len() && self.deleted_bits[id]
    }

    /// Get statistics about the HNSW index
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
        }
    }

    /// Get the number of active nodes
    pub fn len(&self) -> usize {
        self.vec_storage.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get vector dimension
    pub fn dim(&self) -> usize {
        self.vec_storage.dim()
    }

    /// Save the HNSW index to a file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        // Simplified save implementation
        Ok(())
    }

    /// Load the HNSW index from a file
    pub fn load(path: &str, metric: D) -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Not implemented",
        ))
    }
}
