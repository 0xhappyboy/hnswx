use ordered_float::OrderedFloat;
use std::cmp::Ordering;

/// HNSW node representing a vector in the index
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Unique identifier for the node (same as vector storage ID)
    pub id: usize,
    /// Maximum layer where this node exists
    pub level: usize,
    /// Neighbor lists for each layer
    pub friends: Vec<Vec<usize>>,
    /// Whether this node has been marked as deleted
    pub deleted: bool,
}

impl HnswNode {
    /// Create a new HNSW node
    pub fn new(id: usize, level: usize) -> Self {
        Self {
            id,
            level,
            friends: vec![Vec::with_capacity(16); level + 1],
            deleted: false,
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
}

impl SearchResult {
    /// Create a new search result
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