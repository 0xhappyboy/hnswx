#[derive(Debug, Clone)]
pub struct FlatVecStorage {
    /// All vectors stored in a flat array
    pub(crate) data: Vec<f32>,
    /// Vector dimension
    dim: usize,
    /// Number of valid vectors
    size: usize,
    /// Free slots (can be reused)
    free_slots: Vec<usize>,
}

impl FlatVecStorage {
    /// Create a new vector storage
    pub fn new(dim: usize, initial_capacity: usize) -> Self {
        Self {
            data: vec![0.0; dim * initial_capacity],
            dim,
            size: 0,
            free_slots: Vec::new(),
        }
    }

    /// Add a vector and return its ID
    pub fn add_vector(&mut self, vector: &[f32]) -> usize {
        let id = if let Some(id) = self.free_slots.pop() {
            let offset = id * self.dim;
            if offset + self.dim > self.data.len() {
                self.data.resize(offset + self.dim, 0.0);
            }
            self.data[offset..offset + self.dim].copy_from_slice(vector);
            id
        } else {
            let id = self.size;
            let required_len = (id + 1) * self.dim;
            if required_len > self.data.len() {
                self.data.resize(required_len, 0.0);
            }
            let offset = id * self.dim;
            self.data[offset..offset + self.dim].copy_from_slice(vector);
            self.size += 1;
            id
        };
        id
    }

    /// Get vector reference by ID
    pub fn get_vector(&self, id: usize) -> &[f32] {
        let offset = id * self.dim;
        &self.data[offset..offset + self.dim]
    }

    /// Mark slot as free
    pub fn free_slot(&mut self, id: usize) {
        if id >= self.size {
            return;
        }
        self.free_slots.push(id);
        let offset = id * self.dim;
        if offset + self.dim <= self.data.len() {
            for i in 0..self.dim {
                self.data[offset + i] = 0.0;
            }
        }
    }

    /// Get number of active vectors
    pub fn len(&self) -> usize {
        self.size - self.free_slots.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get storage capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity() / self.dim
    }

    /// Get vector dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}
