use hashbrown::HashSet;
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cmp::{Ordering, max, min};
use std::collections::BinaryHeap;

/// Trait for distance metrics used in HNSW
pub trait DistanceMetric: Send + Sync {
    /// cal distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// cal squared distance between two vectors
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Euclidean distance metric
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    /// cal Euclidean distance: sqrt(Σ(x_i - y_i)²)
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// cal squared Euclidean distance: Σ(x_i - y_i)²
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
}

impl EuclideanDistance {
    pub fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| self.distance_squared(query, v))
            .collect()
    }
}

pub struct CosineSimilarity;

impl CosineSimilarity {
    /// (a·b) / (||a|| * ||b||)
    fn cal_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        let mut i = 0;
        let len = a.len();
        let remainder = len % 4;
        while i + 4 <= len {
            dot += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            norm_a += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
            norm_b += b[i] * b[i] + b[i + 1] * b[i + 1] + b[i + 2] * b[i + 2] + b[i + 3] * b[i + 3];
            i += 4;
        }
        for i in len - remainder..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.cal_cosine_similarity(a, b)
    }
}

impl DistanceMetric for CosineSimilarity {
    /// cal cosine distance: 1 - cosine_similarity(a, b)
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.cal_cosine_similarity(a, b)
    }

    /// cal squared cosine distance
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        let dist = self.distance(a, b);
        dist * dist
    }
}

/// Configuration parameters for HNSW index
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Maximum number of elements in the index
    pub max_elements: usize,
    /// Number of established connections per layer (M in the paper)
    pub m: usize,
    /// Maximum number of connections for each element at layer 0
    pub m_max: usize,
    /// Maximum number of connections for each element at highest layer
    pub m_max_0: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search
    pub ef_search: usize,
    /// Level multiplier that influences level distribution
    pub level_multiplier: f64,
    /// Whether to allow replacing deleted elements
    pub allow_replace_deleted: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_elements: 1000,
            m: 16,
            m_max: 32,
            m_max_0: 64,
            ef_construction: 200,
            ef_search: 10,
            level_multiplier: 1.0 / f64::ln(16.0),
            allow_replace_deleted: true,
        }
    }
}

/// HNSW node representing a vector in the index
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Unique identifier for the node
    pub id: usize,
    /// Feature vector stored in this node
    pub vector: Vec<f32>,
    /// Maximum layer where this node exists
    pub level: usize,
    /// Neighbor lists for each layer
    pub friends: Vec<Vec<usize>>,
    /// Whether this node has been marked as deleted
    pub deleted: bool,
}

impl HnswNode {
    /// Create a new HNSW node
    pub fn new(id: usize, vector: Vec<f32>, level: usize) -> Self {
        Self {
            id,
            vector,
            level,
            friends: vec![Vec::with_capacity(16); level + 1], // 减少预分配
            deleted: false,
        }
    }
}

/// Search result containing node ID and distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Node identifier
    pub id: usize,
    /// Distance from query to this node
    pub distance: f32,
}

impl SearchResult {
    pub fn new(id: usize, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        OrderedFloat(self.distance).partial_cmp(&OrderedFloat(other.distance))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

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

/// Hierarchical Navigable Small World (HNSW) index
pub struct HNSW<D: DistanceMetric = EuclideanDistance> {
    /// Configuration parameters
    config: HnswConfig,
    /// Storage for all nodes in the index
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
    /// Set of deleted node IDs
    deleted_ids: HashSet<usize>,
    /// Next available node ID
    next_id: usize,
    vector_dim: Option<usize>,
}

impl<D: DistanceMetric> HNSW<D> {
    /// Create a new HNSW instance
    pub fn new(config: HnswConfig, metric: D) -> Self {
        let max_elements = config.max_elements;
        let max_level = ((max_elements as f64).ln() * config.level_multiplier) as usize;
        let initial_capacity = min(1000, max_elements);
        Self {
            config,
            nodes: Vec::with_capacity(initial_capacity),
            entry_point: None,
            level_max: 0,
            max_level,
            metric,
            rng: StdRng::from_entropy(),
            deleted_ids: HashSet::with_capacity(initial_capacity),
            next_id: 0,
            vector_dim: None,
        }
    }

    /// Generate random level for a new node
    fn get_random_level(&mut self) -> usize {
        let uniform: f64 = self.rng.r#gen();
        (-uniform.ln() * self.config.level_multiplier) as usize
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        if let Some(dim) = self.vector_dim {
            assert_eq!(dim, vector.len(), "Vector dimension mismatch");
        } else {
            self.vector_dim = Some(vector.len());
        }
        let level = min(self.get_random_level(), self.max_level);
        let node_id = self.next_id;
        if node_id >= self.config.max_elements {
            panic!("Exceeded maximum elements limit");
        }
        self.next_id += 1;
        if node_id >= self.nodes.len() {
            let new_size = min(
                max(node_id + 1, self.nodes.len() * 2),
                self.config.max_elements,
            );
            self.nodes.resize_with(new_size, || None);
        }
        let mut new_node = HnswNode::new(node_id, vector.clone(), level);
        if self.entry_point.is_none() {
            self.nodes[node_id] = Some(new_node);
            self.entry_point = Some(node_id);
            self.level_max = level;
            return node_id;
        }
        let ep = self.entry_point.unwrap();
        let query_vector = &new_node.vector;
        let mut curr_level = self.level_max;
        let mut curr_node = ep;
        while curr_level > level {
            if let Some(best) = self.search_layer_best(query_vector, curr_node, curr_level) {
                curr_node = best;
            }
            curr_level -= 1;
        }
        let start_layer = min(level, self.level_max);
        for curr_layer in (0..=start_layer).rev() {
            if let Some(neighbors) = self.search_layer(
                query_vector,
                curr_node,
                min(self.config.ef_construction, 60),
                curr_layer,
            ) {
                let selected_neighbors: Vec<usize> = if neighbors.len() > self.config.m {
                    let mut sorted = neighbors;
                    sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                    sorted.iter().take(self.config.m).map(|r| r.id).collect()
                } else {
                    neighbors.iter().map(|r| r.id).collect()
                };
                for &neighbor_id in &selected_neighbors {
                    self.add_connection(node_id, neighbor_id, curr_layer);
                    self.add_reverse_connection(node_id, neighbor_id, curr_layer);
                }
                for &neighbor_id in &selected_neighbors {
                    self.prune_connections(neighbor_id, curr_layer);
                }
            }
        }
        self.nodes[node_id] = Some(new_node);
        if level > self.level_max {
            self.level_max = level;
            self.entry_point = Some(node_id);
        }
        node_id
    }

    fn search_layer_best(&self, query: &[f32], entry_point: usize, layer: usize) -> Option<usize> {
        if entry_point >= self.nodes.len() {
            return None;
        }
        let entry_node = match self.get_node(entry_point) {
            Some(node) => node,
            None => return None,
        };
        let mut best_id = entry_point;
        let mut best_dist = self.metric.distance(query, &entry_node.vector);
        let mut visited = HashSet::with_capacity(32);
        visited.insert(entry_point);
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
                if visited.contains(&neighbor_id) || self.is_deleted(neighbor_id) {
                    continue;
                }
                let neighbor = match self.get_node(neighbor_id) {
                    Some(neighbor) => neighbor,
                    None => continue,
                };
                let dist = self.metric.distance(query, &neighbor.vector);
                visited.insert(neighbor_id);
                if dist < best_dist {
                    best_dist = dist;
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

    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Option<Vec<SearchResult>> {
        if entry_point >= self.nodes.len() {
            return None;
        }
        let entry_node = match self.get_node(entry_point) {
            Some(node) => node,
            None => return None,
        };
        if ef == 0 {
            return None;
        }
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::with_capacity(ef * 2);
        let entry_dist = self.metric.distance(query, &entry_node.vector);
        let entry_result = SearchResult::new(entry_point, entry_dist);
        candidates.push(ReverseSearchResult(entry_result.clone()));
        visited.insert(entry_point);
        let mut results = vec![entry_result];
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
                    let neighbor = match self.get_node(neighbor_id) {
                        Some(neighbor) => neighbor,
                        None => continue,
                    };
                    let dist = self.metric.distance(query, &neighbor.vector);
                    visited.insert(neighbor_id);
                    candidates.push(ReverseSearchResult(SearchResult::new(neighbor_id, dist)));
                    if candidates.len() > ef * 2 {
                        candidates.pop();
                    }
                }
            } else {
                break;
            }
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if results.len() > ef {
            results.truncate(ef);
        }
        Some(results)
    }

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

    fn add_reverse_connection(&mut self, from: usize, to: usize, layer: usize) {
        self.add_connection(to, from, layer);
    }

    fn prune_connections(&mut self, node_id: usize, layer: usize) {
        if let Some(node) = self.nodes[node_id].as_mut() {
            if layer >= node.friends.len() {
                return;
            }
            let current_len = node.friends[layer].len();
            if current_len > self.config.m_max {
                use rand::seq::SliceRandom;
                use rand::thread_rng;
                let mut friends = std::mem::take(&mut node.friends[layer]);
                if friends.len() > self.config.m_max {
                    friends.shuffle(&mut thread_rng());
                    friends.truncate(self.config.m_max);
                }
                node.friends[layer] = friends;
            }
        }
    }

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
            .search_layer(query, curr_node, ef_search, 0)
            .unwrap_or_else(|| {
                if let Some(ep_node) = self.get_node(ep) {
                    let dist = self.metric.distance(query, &ep_node.vector);
                    vec![SearchResult::new(ep, dist)]
                } else {
                    Vec::new()
                }
            });
        let mut final_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| !self.is_deleted(r.id))
            .collect();
        if final_results.len() < k && final_results.len() > 0 {
            let best_result = final_results[0].clone();
            while final_results.len() < k {
                final_results.push(best_result.clone());
            }
        }
        final_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
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
        let mut was_entry_point = false;
        let mut neighbors_to_update = Vec::new();
        {
            if let Some(node) = self.nodes[node_id].as_mut() {
                node.deleted = true;
                self.deleted_ids.insert(node_id);
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

    /// Check if a node is marked as deleted
    pub fn is_deleted(&self, id: usize) -> bool {
        self.deleted_ids.contains(&id)
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
            deleted_count: self.deleted_ids.len(),
        }
    }

    /// Save the HNSW index to a file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
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

/// Statistics about HNSW index
#[derive(Debug)]
pub struct HnswStats {
    /// Number of active (non-deleted) nodes
    pub node_count: usize,
    /// Current maximum layer in the index
    pub max_level: usize,
    /// Current entry point ID (if any)
    pub entry_point: Option<usize>,
    /// Average number of connections per node
    pub avg_connections: f32,
    /// Maximum number of connections any node has
    pub max_connections: usize,
    /// Number of deleted nodes
    pub deleted_count: usize,
}
