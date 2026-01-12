pub trait DistanceMetric: Send + Sync {
    /// Calculate distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    /// Calculate squared distance between two vectors
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32;
    /// Direct squared distance without sqrt (for comparisons)
    fn distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32;
    /// Batch distance calculation for SIMD optimization
    fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32>;
}

/// Euclidean distance metric with optimized implementations
pub struct EuclideanDistance;

impl EuclideanDistance {
    pub fn new() -> Self {
        Self
    }
}

impl DistanceMetric for EuclideanDistance {
    /// Optimized: Only compute sqrt when absolutely necessary
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_squared_direct(a, b).sqrt()
    }

    /// Optimized squared Euclidean distance
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_squared_direct(a, b)
    }

    /// Direct squared distance computation with loop unrolling
    fn distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut sum = 0.0f32;
        let mut i = 0;
        let chunks = len / 4;
        let remainder = len % 4;
        for _ in 0..chunks {
            let diff0 = a[i] - b[i];
            let diff1 = a[i + 1] - b[i + 1];
            let diff2 = a[i + 2] - b[i + 2];
            let diff3 = a[i + 3] - b[i + 3];
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            i += 4;
        }
        for j in 0..remainder {
            let diff = a[i + j] - b[i + j];
            sum += diff * diff;
        }
        sum
    }

    /// Batch computation
    fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        let mut results = Vec::with_capacity(vectors.len());
        for &vector in vectors {
            results.push(self.distance_squared_direct(query, vector));
        }
        results
    }
}

/// Cosine similarity distance metric
pub struct CosineSimilarity;

impl CosineSimilarity {
    pub fn new() -> Self {
        Self
    }

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
}

impl DistanceMetric for CosineSimilarity {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.cal_cosine_similarity(a, b)
    }

    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        let dist = self.distance(a, b);
        dist * dist
    }

    fn distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32 {
        let dist = self.distance(a, b);
        dist * dist
    }

    fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| {
                let dist = self.distance(query, v);
                dist * dist
            })
            .collect()
    }
}