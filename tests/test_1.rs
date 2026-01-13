use hnswx;

#[cfg(test)]
mod test_1 {

    use hnswx::{CosineSimilarity, EuclideanDistance, HNSW, HnswConfig};

    #[test]
    fn test_hnsw_basic() {
        let config = HnswConfig {
            max_elements: 100,
            m: 16,
            m_max: 32,
            m_max_0: 64,
            ef_construction: 50,
            ef_search: 10,
            ..Default::default()
        };
        let mut hnsw = HNSW::new(config, EuclideanDistance::new());
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![4.0, 5.0, 6.0],
            vec![5.0, 6.0, 7.0],
        ];
        let mut ids = Vec::new();
        for vector in vectors {
            ids.push(hnsw.insert(vector));
        }
        let query = vec![2.1, 3.1, 4.1];
        let results = hnsw.search_knn(&query, 3);
        assert_eq!(results.len(), 3);
        let stats = hnsw.stats();
        assert_eq!(stats.node_count, 5);
        hnsw.delete(ids[0]);
        let results_after_delete = hnsw.search_knn(&query, 3);
        assert!(results_after_delete.iter().all(|r| r.id != ids[0]));
    }

    #[test]
    fn test_cosine_similarity() {
        let config = HnswConfig::default();
        let mut hnsw = HNSW::new(config, CosineSimilarity::new());
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
        ];
        for vector in vectors {
            hnsw.insert(vector);
        }
        let query = vec![1.0, 0.0, 0.0];
        let results = hnsw.search_knn(&query, 2);
        assert!(!results.is_empty());
    }
}
