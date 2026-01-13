use smallvec::SmallVec;

pub trait DistanceMetric: Send + Sync {
    /// Calculate distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    /// Calculate squared distance between two vectors
    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32;
    /// Direct squared distance without sqrt (for comparisons)
    fn distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32;
    /// Batch distance calculation for SIMD optimization
    fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32>;
    /// SIMD optimized batch distance calculation
    fn simd_batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]])
    -> SmallVec<[f32; 64]>;
}

/// Euclidean distance metric with SIMD optimizations
pub struct EuclideanDistance {
    use_simd: bool,
}

impl EuclideanDistance {
    pub fn new() -> Self {
        Self {
            use_simd: cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        }
    }

    /// SIMD-optimized squared Euclidean distance
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn simd_distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let len = a.len();
        let mut sum = 0.0f32;
        #[cfg(target_feature = "avx512f")]
        {
            let mut i = 0;
            let chunk_size = 16;
            let chunks = len / chunk_size;
            for _ in 0..chunks {
                unsafe {
                    let va = _mm512_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                    let diff = _mm512_sub_ps(va, vb);
                    let squared = _mm512_mul_ps(diff, diff);
                    sum += _mm512_reduce_add_ps(squared);
                }
                i += chunk_size;
            }
            for j in i..len {
                let diff = a[j] - b[j];
                sum += diff * diff;
            }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            let mut i = 0;
            let chunk_size = 8;
            let chunks = len / chunk_size;
            for _ in 0..chunks {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    let diff = _mm256_sub_ps(va, vb);
                    let squared = _mm256_mul_ps(diff, diff);
                    let low = _mm256_extractf128_ps(squared, 0);
                    let high = _mm256_extractf128_ps(squared, 1);
                    let sum128 = _mm_add_ps(low, high);
                    let shuf = _mm_shuffle_ps(sum128, sum128, 0b01001110);
                    let sums = _mm_add_ps(sum128, shuf);
                    let shuf2 = _mm_shuffle_ps(sums, sums, 0b10110001);
                    let final_sum = _mm_add_ps(sums, shuf2);
                    sum += _mm_cvtss_f32(final_sum);
                }
                i += chunk_size;
            }
            for j in i..len {
                let diff = a[j] - b[j];
                sum += diff * diff;
            }
        }
        #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
        {
            let mut i = 0;
            let chunk_size = 4;
            let chunks = len / chunk_size;
            for _ in 0..chunks {
                unsafe {
                    let va = _mm_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm_loadu_ps(b.as_ptr().add(i));
                    let diff = _mm_sub_ps(va, vb);
                    let squared = _mm_mul_ps(diff, diff);
                    let shuf = _mm_shuffle_ps(squared, squared, 0b01001110);
                    let sums = _mm_add_ps(squared, shuf);
                    let shuf2 = _mm_shuffle_ps(sums, sums, 0b10110001);
                    let final_sum = _mm_add_ps(sums, shuf2);

                    sum += _mm_cvtss_f32(final_sum);
                }
                i += chunk_size;
            }
            for j in i..len {
                let diff = a[j] - b[j];
                sum += diff * diff;
            }
        }
        #[cfg(not(any(
            target_feature = "sse",
            target_feature = "avx2",
            target_feature = "avx512f"
        )))]
        {
            sum = self.distance_squared_direct_fallback(a, b);
        }
        sum
    }

    /// Fallback implementation without SIMD
    fn distance_squared_direct_fallback(&self, a: &[f32], b: &[f32]) -> f32 {
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
}

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_squared_direct(a, b).sqrt()
    }

    fn distance_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_squared_direct(a, b)
    }

    fn distance_squared_direct(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.use_simd {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.simd_distance_squared_direct(a, b);
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                return self.distance_squared_direct_fallback(a, b);
            }
        }
        self.distance_squared_direct_fallback(a, b)
    }

    fn batch_distance_squared(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        if vectors.len() <= 8 {
            // Small batch: use simple loop
            let mut results = Vec::with_capacity(vectors.len());
            for &vector in vectors {
                results.push(self.distance_squared_direct(query, vector));
            }
            results
        } else {
            // Larger batch: use SIMD version
            self.simd_batch_distance_squared(query, vectors).to_vec()
        }
    }

    fn simd_batch_distance_squared(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
    ) -> SmallVec<[f32; 64]> {
        let mut results = SmallVec::with_capacity(vectors.len());
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.use_simd && vectors.len() >= 4 {
            // SIMD-optimized batch processing
            let batch_size = 4;
            let batches = vectors.len() / batch_size;
            for batch_idx in 0..batches {
                let start = batch_idx * batch_size;
                let batch_vectors = &vectors[start..start + batch_size];
                let len = query.len();
                let mut sums = [0.0f32; 4];
                let mut i = 0;
                let chunks = len / 4;
                for _ in 0..chunks {
                    let q0 = query[i];
                    let q1 = query[i + 1];
                    let q2 = query[i + 2];
                    let q3 = query[i + 3];
                    for (vec_idx, &vector) in batch_vectors.iter().enumerate() {
                        let diff0 = q0 - vector[i];
                        let diff1 = q1 - vector[i + 1];
                        let diff2 = q2 - vector[i + 2];
                        let diff3 = q3 - vector[i + 3];
                        sums[vec_idx] +=
                            diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                    }
                    i += 4;
                }
                for j in i..len {
                    let q = query[j];
                    for (vec_idx, &vector) in batch_vectors.iter().enumerate() {
                        let diff = q - vector[j];
                        sums[vec_idx] += diff * diff;
                    }
                }
                results.extend_from_slice(&sums);
            }
            let remaining_start = batches * batch_size;
            for &vector in &vectors[remaining_start..] {
                results.push(self.distance_squared_direct(query, vector));
            }
        } else {
            for &vector in vectors {
                results.push(self.distance_squared_direct(query, vector));
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            for &vector in vectors {
                results.push(self.distance_squared_direct(query, vector));
            }
        }
        results
    }
}

/// Cosine similarity distance metric with SIMD optimizations
pub struct CosineSimilarity {
    use_simd: bool,
}

impl CosineSimilarity {
    pub fn new() -> Self {
        Self {
            use_simd: cfg!(any(target_arch = "x86", target_arch = "x86_64")),
        }
    }

    /// SIMD-optimized cosine similarity
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn simd_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        #[cfg(target_feature = "avx2")]
        {
            let mut i = 0;
            let chunk_size = 8;
            let chunks = len / chunk_size;
            let mut dot_vec = _mm256_setzero_ps();
            let mut norm_a_vec = _mm256_setzero_ps();
            let mut norm_b_vec = _mm256_setzero_ps();
            for _ in 0..chunks {
                unsafe {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    let mul = _mm256_mul_ps(va, vb);
                    dot_vec = _mm256_add_ps(dot_vec, mul);
                    let a_squared = _mm256_mul_ps(va, va);
                    norm_a_vec = _mm256_add_ps(norm_a_vec, a_squared);
                    let b_squared = _mm256_mul_ps(vb, vb);
                    norm_b_vec = _mm256_add_ps(norm_b_vec, b_squared);
                }
                i += chunk_size;
            }
            unsafe {
                dot += horizontal_sum_avx(dot_vec);
                norm_a += horizontal_sum_avx(norm_a_vec);
                norm_b += horizontal_sum_avx(norm_b_vec);
            }
            for j in i..len {
                dot += a[j] * b[j];
                norm_a += a[j] * a[j];
                norm_b += b[j] * b[j];
            }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            dot = self.cosine_similarity_fallback(a, b, &mut norm_a, &mut norm_b);
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }

    fn cosine_similarity_fallback(
        &self,
        a: &[f32],
        b: &[f32],
        norm_a: &mut f32,
        norm_b: &mut f32,
    ) -> f32 {
        let mut dot = 0.0f32;
        *norm_a = 0.0;
        *norm_b = 0.0;
        let mut i = 0;
        let len = a.len();
        let chunks = len / 4;
        for _ in 0..chunks {
            dot += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            *norm_a +=
                a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
            *norm_b +=
                b[i] * b[i] + b[i + 1] * b[i + 1] + b[i + 2] * b[i + 2] + b[i + 3] * b[i + 3];
            i += 4;
        }
        for i in i..len {
            dot += a[i] * b[i];
            *norm_a += a[i] * a[i];
            *norm_b += b[i] * b[i];
        }
        dot
    }
}

#[cfg(target_feature = "avx2")]
unsafe fn horizontal_sum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let low = _mm256_extractf128_ps(v, 0);
    let high = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0b01001110);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0b10110001);
    let final_sum = _mm_add_ps(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

impl DistanceMetric for CosineSimilarity {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - self.cosine_similarity(a, b)
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

    fn simd_batch_distance_squared(
        &self,
        query: &[f32],
        vectors: &[&[f32]],
    ) -> SmallVec<[f32; 64]> {
        let mut results = SmallVec::with_capacity(vectors.len());
        if self.use_simd && vectors.len() >= 4 {
            // Batch process 4 vectors at a time
            let batch_size = 4;
            let batches = vectors.len() / batch_size;
            for batch_idx in 0..batches {
                let start = batch_idx * batch_size;
                let batch_vectors = &vectors[start..start + batch_size];
                let mut query_norm = 0.0;
                for &q in query {
                    query_norm += q * q;
                }
                let query_norm_sqrt = query_norm.sqrt();
                for &vector in batch_vectors {
                    let mut dot = 0.0;
                    let mut vec_norm = 0.0;
                    for (&q, &v) in query.iter().zip(vector.iter()) {
                        dot += q * v;
                        vec_norm += v * v;
                    }
                    let similarity = if query_norm_sqrt == 0.0 || vec_norm == 0.0 {
                        0.0
                    } else {
                        dot / (query_norm_sqrt * vec_norm.sqrt())
                    };
                    let dist = 1.0 - similarity;
                    results.push(dist * dist);
                }
            }
            let remaining_start = batches * batch_size;
            for &vector in &vectors[remaining_start..] {
                let dist = self.distance(query, vector);
                results.push(dist * dist);
            }
        } else {
            for &vector in vectors {
                let dist = self.distance(query, vector);
                results.push(dist * dist);
            }
        }
        results
    }
}

impl CosineSimilarity {
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.use_simd {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                return self.simd_cosine_similarity(a, b);
            }
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        self.cosine_similarity_fallback(a, b, &mut norm_a, &mut norm_b);
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        }
    }
}
