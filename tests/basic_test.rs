use hnswx;

#[cfg(test)]
mod basic_test {
    use super::*;
    use hnswx::{EuclideanDistance, HNSW, HnswConfig};
    use rand::Rng;
    use std::time::Instant;

    #[test]
    fn test_hnsw_basic_operations() {
        let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance::new());
        println!("Inserting data");
        let id1 = hnsw.insert(vec![1.0, 2.0, 3.0]);
        let id2 = hnsw.insert(vec![2.0, 3.0, 4.0]);
        let id3 = hnsw.insert(vec![3.0, 4.0, 5.0]);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        println!("Testing search functionality");
        let results = hnsw.search_knn(&[2.1, 3.1, 4.1], 2);
        assert_eq!(results.len(), 2);
        println!("Search results: {:?}", results);
        println!("Testing delete functionality");
        let deleted = hnsw.delete(id2);
        assert!(deleted);
        let results_after = hnsw.search_knn(&[2.1, 3.1, 4.1], 2);
        assert_eq!(results_after.len(), 2);
        for result in &results_after {
            assert_ne!(result.id, id2);
        }
        let stats = hnsw.stats();
        println!("Statistics: {:?}", stats);
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.deleted_count, 1);
    }

    #[test]
    fn test_hnsw_small_performance() {
        let config = HnswConfig {
            max_elements: 10_000,
            ef_construction: 100,
            ef_search: 20,
            ..Default::default()
        };
        let mut hnsw = HNSW::new(config, EuclideanDistance::new());
        let mut rng = rand::thread_rng();
        let dim = 8;
        let num_vectors = 1_000;
        println!("Inserting {} {}D vectors", num_vectors, dim);
        let insert_start = Instant::now();
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            hnsw.insert(vector);
            if i % 200 == 0 && i > 0 {
                println!("Inserted {} vectors", i);
            }
        }
        let insert_duration = insert_start.elapsed();
        println!("Insert completed, time: {:?}", insert_duration);
        println!("Testing query performance");
        let num_queries = 10;
        let mut total_query_time = 0;
        for i in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            let query_start = Instant::now();
            let results = hnsw.search_knn(&query, 5);
            let query_duration = query_start.elapsed().as_micros();
            total_query_time += query_duration;
            println!(
                "Query {}: {} results, time {}μs",
                i,
                results.len(),
                query_duration
            );
        }
        let avg_query_time = total_query_time as f64 / num_queries as f64;
        println!("Average query time: {:.2}μs", avg_query_time);
        let stats = hnsw.stats();
        println!("\nIndex statistics:");
        println!("Node count: {}", stats.node_count);
        println!("Max level: {}", stats.max_level);
        println!("Average connections: {:.2}", stats.avg_connections);
        println!("Max connections: {}", stats.max_connections);
        assert_eq!(stats.node_count, num_vectors);
        assert!(avg_query_time < 10000.0, "Query performance is reasonable");
    }

    #[test]
    fn test_hnsw_edge_cases() {
        println!("Testing edge cases");
        let mut hnsw = HNSW::new(HnswConfig::default(), EuclideanDistance::new());
        let results = hnsw.search_knn(&[1.0, 2.0, 3.0], 5);
        assert!(
            results.is_empty(),
            "Empty index should return empty results"
        );
        let id = hnsw.insert(vec![1.0, 2.0, 3.0]);
        let results = hnsw.search_knn(&[1.1, 2.1, 3.1], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
        let results = hnsw.search_knn(&[1.1, 2.1, 3.1], 10);
        assert_eq!(results.len(), 1);
        let deleted = hnsw.delete(999);
        assert!(!deleted, "Deleting non-existent node should return false");
        let deleted1 = hnsw.delete(id);
        assert!(deleted1, "First delete should succeed");
        let deleted2 = hnsw.delete(id);
        assert!(!deleted2, "Repeated delete should return false");
    }
}
