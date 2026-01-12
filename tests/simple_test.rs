use hnswx;

#[cfg(test)]
mod simple_test {
    use hnswx::{EuclideanDistance, HNSW, HnswConfig};

    #[test]
    fn test_hnsw_basic_operations() {
        let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance);
        let id = hnsw.insert(vec![1.0, 2.0, 3.0]);
        let results = hnsw.search_knn(&[1.1, 2.1, 3.1], 1);
        let deleted = hnsw.delete(id);
        assert_eq!(results[0].id, id);
    }
}
