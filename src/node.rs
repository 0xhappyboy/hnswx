use ordered_float::OrderedFloat;
use smallvec::{SmallVec};
use std::cmp::Ordering;

/// HNSW node representing a vector in the index
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Unique identifier for the node (same as vector storage ID)
    pub id: usize,
    /// Maximum layer where this node exists
    pub level: usize,
    /// Neighbor lists for each layer (smallvec for optimization)
    pub friends: Vec<SmallVec<[usize; 16]>>,
    /// Whether this node has been marked as deleted
    pub deleted: bool,
    /// Cached level for fast access
    pub cached_level: u8,
}

impl HnswNode {
    /// Create a new HNSW node
    pub fn new(id: usize, level: usize) -> Self {
        let level = level.min(255);
        let mut friends = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            friends.push(SmallVec::with_capacity(16));
        }
        Self {
            id,
            level,
            friends,
            deleted: false,
            cached_level: level as u8,
        }
    }

    /// Get neighbors at a specific layer
    pub fn get_friends(&self, layer: usize) -> &[usize] {
        if layer < self.friends.len() {
            &self.friends[layer]
        } else {
            &[]
        }
    }

    /// Add a neighbor at a specific layer
    pub fn add_friend(&mut self, layer: usize, friend_id: usize) {
        if layer >= self.friends.len() {
            self.friends.resize(layer + 1, SmallVec::new());
            self.level = self.friends.len() - 1;
            self.cached_level = self.level as u8;
        }
        if !self.friends[layer].contains(&friend_id) {
            self.friends[layer].push(friend_id);
        }
    }

    /// Remove a neighbor at a specific layer
    pub fn remove_friend(&mut self, layer: usize, friend_id: usize) {
        if layer < self.friends.len() {
            if let Some(pos) = self.friends[layer].iter().position(|&id| id == friend_id) {
                self.friends[layer].remove(pos);
            }
        }
    }
}

/// Search result containing node ID and distance
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Node identifier
    pub id: usize,
    /// Distance from query to this node (squared distance during search)
    pub distance: f32,
    /// Cached vector for faster access
    pub cached_vector: Option<SmallVec<[f32; 64]>>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: usize, distance: f32) -> Self {
        Self {
            id,
            distance,
            cached_vector: None,
        }
    }

    /// Create a search result with cached vector
    pub fn with_vector(id: usize, distance: f32, vector: &[f32]) -> Self {
        let mut cached = SmallVec::with_capacity(vector.len());
        cached.extend_from_slice(vector);

        Self {
            id,
            distance,
            cached_vector: Some(cached),
        }
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
