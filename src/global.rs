/// Default batch size for asynchronous operations
pub(crate) const BATCH_SIZE: usize = 100;
/// Maximum number of iterations for greedy search in search_layer_best
pub(crate) const SEARCH_LAYER_ITERATIONS: usize = 15;
/// Maximum number of iterations for optimized search algorithms
pub(crate) const OPTIMIZED_SEARCH_ITERATIONS: usize = 150;
/// Maximum value for ef_construction parameter during insertion
pub(crate) const MAX_EF_CONSTRUCTION: usize = 60;
/// Maximum neighbors to check per layer during greedy search
pub(crate) const MAX_NEIGHBORS_PER_LAYER: usize = 8;
/// Maximum neighbors to check per layer during optimized search
pub(crate) const MAX_NEIGHBORS_SEARCH_LAYER: usize = 12;
/// Initial capacity for node storage when HNSW is created
pub(crate) const INITIAL_CAPACITY: usize = 1000;
/// Increment for growing bit vectors when resizing
pub(crate) const GROWTH_INCREMENT: usize = 100;
/// Multiplier for determining visited array size in optimized search
pub(crate) const LOCAL_VISITED_SIZE_MULTIPLIER: usize = 4;
/// Threshold multiplier for early termination in search
pub(crate) const SEARCH_DISTANCE_THRESHOLD_MULTIPLIER: f32 = 1.01;
/// Multiplier for determining ef_search based on requested k
pub(crate) const DEFAULT_K_MULTIPLIER: usize = 3;
