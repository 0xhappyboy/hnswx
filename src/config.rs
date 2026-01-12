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
    /// Vector dimension
    pub vector_dim: usize,
    /// Total storage size in floats
    pub storage_size: usize,
}
