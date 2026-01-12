use ordered_float::OrderedFloat;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cmp::{max, min};
use std::collections::HashSet;

/// Trait for distance metrics used in HNSW
pub trait DistanceMetric: Send + Sync {
    /// Calculate distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Calculate squared distance between two vectors
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Euclidean distance metric
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    /// Calculate Euclidean distance: sqrt(Σ(x_i - y_i)²)
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Calculate squared Euclidean distance: Σ(x_i - y_i)²
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
}

/// Cosine similarity metric (converted to distance)
pub struct CosineSimilarity;

impl DistanceMetric for CosineSimilarity {
    /// Calculate cosine distance: 1 - cosine_similarity(a, b)
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.compute_cosine_similarity(a, b)
    }

    /// Calculate squared cosine distance
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        let dist = self.distance(a, b);
        dist * dist
    }
}

impl CosineSimilarity {
    /// Compute cosine similarity: (a·b) / (||a|| * ||b||)
    fn compute_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }

    /// Public method to get cosine similarity
    ///
    /// # Example
    /// ```
    /// let metric = CosineSimilarity;
    /// let sim = metric.similarity(&[1.0, 0.0], &[1.0, 0.0]); // Returns 1.0
    /// ```
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.compute_cosine_similarity(a, b)
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
    /// Create default HNSW configuration
    ///
    /// # Default Values
    /// - max_elements: 1000
    /// - m: 16
    /// - m_max: 32
    /// - m_max_0: 64
    /// - ef_construction: 200
    /// - ef_search: 10
    /// - level_multiplier: 1.0 / ln(16.0)
    /// - allow_replace_deleted: true
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
    ///
    /// # Arguments
    /// * `id` - Unique node identifier
    /// * `vector` - Feature vector to store
    /// * `level` - Maximum layer for this node
    pub fn new(id: usize, vector: Vec<f32>, level: usize) -> Self {
        Self {
            id,
            vector,
            level,
            friends: vec![vec![]; level + 1],
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

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        OrderedFloat(self.distance).partial_cmp(&OrderedFloat(other.distance))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

/// Hierarchical Navigable Small World (HNSW) index
///
/// # Type Parameters
/// * `D` - Distance metric type (default: EuclideanDistance)
///
/// # Example
/// ```
/// use hnsw::*;
///
/// let config = HnswConfig::default();
/// let mut hnsw = HNSW::new(config, EuclideanDistance);
///
/// // Insert vectors
/// let id1 = hnsw.insert(vec![1.0, 2.0, 3.0]);
/// let id2 = hnsw.insert(vec![2.0, 3.0, 4.0]);
///
/// // Search for nearest neighbors
/// let results = hnsw.search_knn(&[2.1, 3.1, 4.1], 1);
/// ```
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
}

impl<D: DistanceMetric> HNSW<D> {
    /// Create a new HNSW instance with given configuration and distance metric
    ///
    /// # Arguments
    /// * `config` - HNSW configuration parameters
    /// * `metric` - Distance metric implementation
    ///
    /// # Example
    /// ```
    /// use hnsw::*;
    ///
    /// let config = HnswConfig {
    ///     max_elements: 1000,
    ///     m: 16,
    ///     ..Default::default()
    /// };
    /// let hnsw = HNSW::new(config, EuclideanDistance);
    /// ```
    pub fn new(config: HnswConfig, metric: D) -> Self {
        let max_level = ((config.max_elements as f64).ln() * config.level_multiplier) as usize;
        Self {
            config,
            nodes: Vec::with_capacity(100),
            entry_point: None,
            level_max: 0,
            max_level,
            metric,
            rng: StdRng::from_entropy(),
            deleted_ids: HashSet::new(),
            next_id: 0,
        }
    }

    /// Generate random level for a new node using exponential distribution
    fn get_random_level(&mut self) -> usize {
        let uniform: f64 = self.rng.r#gen();
        (-uniform.ln() * self.config.level_multiplier) as usize
    }

    /// Insert a vector into the HNSW index
    ///
    /// # Arguments
    /// * `vector` - Feature vector to insert
    ///
    /// # Returns
    /// Unique identifier for the inserted node
    ///
    /// # Panics
    /// Panics if the maximum element limit is exceeded
    ///
    /// # Example
    /// ```
    /// let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance);
    /// let node_id = hnsw.insert(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        let level = min(self.get_random_level(), self.max_level);
        let node_id = self.next_id;
        self.next_id += 1;
        // Expand node storage if needed
        while self.nodes.len() <= node_id {
            self.nodes.push(None);
        }
        let new_node = HnswNode::new(node_id, vector.clone(), level);
        if self.nodes.len() > self.config.max_elements {
            panic!("Exceeded maximum elements limit");
        }
        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.nodes[node_id] = Some(new_node);
            self.entry_point = Some(node_id);
            self.level_max = level;
            return node_id;
        }
        // Find nearest entry point by searching from top layers down
        let ep = self.entry_point.unwrap();
        let mut curr_level = self.level_max;
        let mut curr_node = ep;
        while curr_level > level {
            if let Some(results) = self.search_layer(&vector, curr_node, 1, curr_level) {
                if let Some(best) = results.last() {
                    curr_node = best.id;
                }
            }
            curr_level -= 1;
        }
        // Insert node at each layer from bottom to top
        for curr_layer in (0..=min(level, self.level_max)).rev() {
            if let Some(neighbors) =
                self.search_layer(&vector, curr_node, self.config.ef_construction, curr_layer)
            {
                let selected_neighbors =
                    self.select_neighbors(&vector, &neighbors, self.config.m, curr_layer);
                // Create bidirectional connections
                for &neighbor_id in &selected_neighbors {
                    self.add_connection(node_id, neighbor_id, curr_layer);
                    self.add_reverse_connection(node_id, neighbor_id, curr_layer);
                }
                // Prune connections to maintain maximum limits
                let neighbors_to_prune = selected_neighbors.clone();
                for neighbor_id in neighbors_to_prune {
                    self.prune_connections(neighbor_id, curr_layer);
                }
            }
        }
        // Store the new node
        self.nodes[node_id] = Some(new_node);
        // Update entry point if new node has higher level
        if level > self.level_max {
            self.level_max = level;
            self.entry_point = Some(node_id);
        }
        node_id
    }

    /// Search for nearest neighbors in a specific layer
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `entry_point` - Starting node for search
    /// * `ef` - Size of dynamic candidate list
    /// * `layer` - Layer to search in
    ///
    /// # Returns
    /// Vector of search results sorted by distance, or None if entry point is invalid
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Option<Vec<SearchResult>> {
        if entry_point >= self.nodes.len() || self.nodes[entry_point].is_none() {
            return None;
        }
        let mut visited = HashSet::new();
        let mut candidates = Vec::new();
        let mut result = Vec::new();
        let entry_node = match self.get_node(entry_point) {
            Some(node) => node,
            None => return None,
        };
        let entry_dist = self.metric.distance(query, &entry_node.vector);
        candidates.push(SearchResult {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);
        while !candidates.is_empty() {
            // Sort candidates by distance (closest first)
            candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            let current = candidates.remove(0);
            result.push(current.clone());

            if result.len() >= ef {
                break;
            }
            let node = match self.get_node(current.id) {
                Some(node) => node,
                None => continue, // if the node does not exist, skip this step.
            };
            if layer < node.friends.len() {
                for &neighbor_id in &node.friends[layer] {
                    if visited.contains(&neighbor_id) || self.is_deleted(neighbor_id) {
                        continue;
                    }
                    let neighbor = match self.get_node(neighbor_id) {
                        Some(neighbor) => neighbor,
                        None => continue,
                    };
                    let dist = self.metric.distance(query, &neighbor.vector);
                    visited.insert(neighbor_id);
                    candidates.push(SearchResult {
                        id: neighbor_id,
                        distance: dist,
                    });
                }
            }
        }
        // Sort results and truncate to ef size
        result.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result.truncate(ef);
        Some(result)
    }

    /// Select neighbors from candidate list
    ///
    /// # Arguments
    /// * `query` - Query vector (unused in current implementation)
    /// * `candidates` - Candidate search results
    /// * `m` - Number of neighbors to select
    /// * `_layer` - Layer number (unused)
    ///
    /// # Returns
    /// Vector of selected neighbor IDs
    fn select_neighbors(
        &self,
        _query: &[f32],
        candidates: &[SearchResult],
        m: usize,
        _layer: usize,
    ) -> Vec<usize> {
        let mut selected = Vec::new();
        let mut sorted_candidates = candidates.to_vec();
        // Sort candidates by distance and select top m
        sorted_candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        for candidate in sorted_candidates.iter().take(m) {
            selected.push(candidate.id);
        }
        selected
    }

    /// Add a directed connection between two nodes at a specific layer
    ///
    /// # Arguments
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    /// * `layer` - Layer to add connection in
    fn add_connection(&mut self, from: usize, to: usize, layer: usize) {
        if let Some(node) = self.nodes[from].as_mut() {
            if layer >= node.friends.len() {
                node.friends.resize(layer + 1, vec![]);
            }
            if !node.friends[layer].contains(&to) {
                node.friends[layer].push(to);
            }
        }
    }

    /// Add a reverse connection (makes connection bidirectional)
    ///
    /// # Arguments
    /// * `from` - Original source node ID
    /// * `to` - Original target node ID
    /// * `layer` - Layer to add connection in
    fn add_reverse_connection(&mut self, from: usize, to: usize, layer: usize) {
        self.add_connection(to, from, layer);
    }

    /// Prune connections to maintain maximum connection limit
    ///
    /// # Arguments
    /// * `node_id` - Node ID whose connections need pruning
    /// * `layer` - Layer to prune connections in
    fn prune_connections(&mut self, node_id: usize, layer: usize) {
        let (vector, friends) = {
            match self.get_node(node_id) {
                Some(node) => (node.vector.clone(), node.friends[layer].clone()),
                None => return,
            }
        };
        if friends.len() <= self.config.m_max {
            return;
        }
        // Calculate distances to all friends
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for &id in &friends {
            if let Some(neighbor) = self.get_node(id) {
                let dist = self.metric.distance(&vector, &neighbor.vector);
                distances.push((id, dist));
            }
        }
        // Sort by distance and keep only closest m_max friends
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let pruned_friends: Vec<usize> = distances
            .iter()
            .take(self.config.m_max)
            .map(|&(id, _)| id)
            .collect();

        // Update node with pruned connections
        if let Some(node) = self.nodes[node_id].as_mut() {
            if layer < node.friends.len() {
                node.friends[layer] = pruned_friends;
            }
        }
    }

    /// Search for k-nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of search results sorted by distance
    ///
    /// # Example
    /// ```
    /// let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance);
    /// hnsw.insert(vec![1.0, 2.0, 3.0]);
    /// hnsw.insert(vec![2.0, 3.0, 4.0]);
    ///
    /// let results = hnsw.search_knn(&[2.1, 3.1, 4.1], 1);
    /// ```
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() {
            return Vec::new();
        }
        let ep = self.entry_point.unwrap();
        let mut curr_node = ep;
        let mut curr_level = self.level_max;
        // Navigate from top layers down to find good starting point
        while curr_level > 0 {
            if let Some(candidates) = self.search_layer(query, curr_node, 1, curr_level) {
                if !candidates.is_empty() {
                    curr_node = candidates[0].id;
                }
            }
            curr_level -= 1;
        }
        // Search in bottom layer (layer 0)
        let results = self
            .search_layer(query, curr_node, max(self.config.ef_search, k), 0)
            .unwrap_or_default();
        results
            .into_iter()
            .filter(|r| !self.is_deleted(r.id))
            .take(k)
            .collect()
    }

    /// Delete a node from the index
    ///
    /// # Arguments
    /// * `node_id` - ID of node to delete
    ///
    /// # Returns
    /// true if node was successfully deleted, false otherwise
    ///
    /// # Example
    /// ```
    /// let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance);
    /// let id = hnsw.insert(vec![1.0, 2.0, 3.0]);
    /// let deleted = hnsw.delete(id); // Returns true
    /// ```
    pub fn delete(&mut self, node_id: usize) -> bool {
        if node_id >= self.nodes.len() || self.nodes[node_id].is_none() {
            return false;
        }
        let mut was_entry_point = false;
        let mut neighbors_to_update = Vec::new();
        {
            if let Some(node) = self.nodes[node_id].as_mut() {
                // Mark node as deleted
                node.deleted = true;
                self.deleted_ids.insert(node_id);
                was_entry_point = Some(node_id) == self.entry_point;
                // Collect all neighbor connections that need updating
                for layer in 0..node.friends.len() {
                    for &neighbor_id in &node.friends[layer] {
                        neighbors_to_update.push((layer, neighbor_id));
                    }
                }
            } else {
                return false;
            }
        }
        // Remove this node from all neighbor connections
        for (layer, neighbor_id) in neighbors_to_update {
            if neighbor_id < self.nodes.len() {
                if let Some(neighbor) = self.nodes[neighbor_id].as_mut() {
                    if layer < neighbor.friends.len() {
                        neighbor.friends[layer].retain(|&id| id != node_id);
                    }
                }
            }
        }
        // Update entry point if this was the entry point
        if was_entry_point {
            self.update_entry_point();
        }
        true
    }

    /// Update the entry point after deletion
    fn update_entry_point(&mut self) {
        self.entry_point = self
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.as_ref().map_or(false, |n| !n.deleted))
            .map(|(i, _)| i);
        if let Some(ep) = self.entry_point {
            self.level_max = self.get_node(ep).unwrap().level;
        } else {
            self.level_max = 0;
        }
    }

    /// Get a reference to a node by ID
    ///
    /// # Arguments
    /// * `id` - Node ID
    ///
    /// # Returns
    /// Reference to node if it exists and is not deleted, None otherwise
    fn get_node(&self, id: usize) -> Option<&HnswNode> {
        self.nodes.get(id).and_then(|n| n.as_ref())
    }

    /// Check if a node is marked as deleted
    ///
    /// # Arguments
    /// * `id` - Node ID
    ///
    /// # Returns
    /// true if node is deleted, false otherwise
    pub fn is_deleted(&self, id: usize) -> bool {
        self.deleted_ids.contains(&id)
    }

    /// Get statistics about the HNSW index
    ///
    /// # Returns
    /// Statistics structure containing various metrics
    ///
    /// # Example
    /// ```
    /// let hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance);
    /// let stats = hnsw.stats();
    /// println!("Node count: {}", stats.node_count);
    /// ```
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
    ///
    /// # Arguments
    /// * `path` - File path to save to
    ///
    /// # Returns
    /// io::Result indicating success or failure
    ///
    /// # Note
    /// Currently not implemented - returns Ok(())
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        Ok(())
    }

    /// Load the HNSW index from a file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    /// Loaded HNSW instance or error
    ///
    /// # Note
    /// Currently not implemented - returns error
    pub fn load(path: &str, metric: D) -> std::io::Result<Self> {
        // TODO: Implement deserialization
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
