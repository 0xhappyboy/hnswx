#[cfg(test)]
mod querying_100k_data {
    use super::*;
    use hnswx::{EuclideanDistance, HNSW, HnswConfig};
    use rand::Rng;
    use std::time::Instant;

    #[test]
    fn test_hnsw_100k_vectors() {
        println!("100k vectors performance test");
        let config = HnswConfig {
            max_elements: 100_000,
            m: 16,
            m_max: 32,
            m_max_0: 64,
            ef_construction: 200,
            ef_search: 50,
            ..Default::default()
        };
        let mut hnsw = HNSW::new(config, EuclideanDistance::new());
        let mut rng = rand::thread_rng();
        let dim = 128;
        let num_vectors = 100_000;
        println!("Starting to insert {} {}D vectors", num_vectors, dim);
        let insert_start = Instant::now();
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            hnsw.insert(vector);
            if i % 10_000 == 0 && i > 0 {
                println!(
                    "  Inserted {} vectors ({:.1}%)",
                    i,
                    (i as f32 / num_vectors as f32) * 100.0
                );
                let stats = hnsw.stats();
                println!(
                    "    Current stats: nodes={}, max_level={}, avg_connections={:.2}",
                    stats.node_count, stats.max_level, stats.avg_connections
                );
            }
        }
        let insert_duration = insert_start.elapsed();
        println!("Insert completed, total time: {:?}", insert_duration);
        println!(
            "Average insertion time per vector: {:?}",
            insert_duration / num_vectors as u32
        );
        let final_stats = hnsw.stats();
        println!("\nFinal index statistics:");
        println!("Node count: {}", final_stats.node_count);
        println!("Max level: {}", final_stats.max_level);
        println!("Average connections: {:.2}", final_stats.avg_connections);
        println!("Max connections: {}", final_stats.max_connections);
        println!("Entry point: {:?}", final_stats.entry_point);
        println!("\nStarting query performance test");
        let num_queries = 100;
        let mut total_query_time = 0;
        let mut min_query_time = u128::MAX;
        let mut max_query_time = 0;
        let test_queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();
        for (i, query) in test_queries.iter().enumerate() {
            let query_start = Instant::now();
            let results = hnsw.search_knn(query, 10);
            let query_duration = query_start.elapsed().as_micros();
            total_query_time += query_duration;
            min_query_time = min_query_time.min(query_duration);
            max_query_time = max_query_time.max(query_duration);
            if i % 20 == 0 {
                println!(
                    "  Query {}: {} results, time {}μs",
                    i,
                    results.len(),
                    query_duration
                );
            }
            assert_eq!(results.len(), 10, "Should find 10 nearest neighbors");
            for j in 1..results.len() {
                assert!(
                    results[j - 1].distance <= results[j].distance,
                    "Results should be sorted by distance"
                );
            }
        }
        let avg_query_time = total_query_time as f64 / num_queries as f64;
        let qps = 1_000_000.0 / avg_query_time;
        println!("\nQuery performance summary (100 queries average):");
        println!("Average query time: {:.2}μs", avg_query_time);
        println!("Minimum query time: {}μs", min_query_time);
        println!("Maximum query time: {}μs", max_query_time);
        println!("QPS (queries per second): {:.2}", qps);
        println!("\nTesting query performance with different k values:");
        let test_k_values = [1, 5, 10, 20, 50];
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        for &k in &test_k_values {
            let start = Instant::now();
            let results = hnsw.search_knn(&query, k);
            let duration = start.elapsed().as_micros();
            println!("  k={}: {} results, time {}μs", k, results.len(), duration);
        }
        println!("\nTesting delete performance");
        let delete_count = 1_000;
        let delete_start = Instant::now();
        for i in 0..delete_count {
            if hnsw.delete(i) {
                if i > 0 && i % 200 == 0 {
                    println!("  Deleted {} nodes", i);
                }
            }
        }
        let delete_duration = delete_start.elapsed();
        println!(
            "Deleted {} nodes, time: {:?}",
            delete_count, delete_duration
        );
        println!(
            "Average delete time per node: {:?}",
            delete_duration / delete_count as u32
        );
        let after_delete_stats = hnsw.stats();
        println!("\nStatistics after deletion:");
        println!("Active node count: {}", after_delete_stats.node_count);
        println!("Deleted node count: {}", after_delete_stats.deleted_count);
        println!(
            "Average connections: {:.2}",
            after_delete_stats.avg_connections
        );
        println!("\nVerifying queries after deletion");
        let query_after: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let start = Instant::now();
        let results_after = hnsw.search_knn(&query_after, 10);
        let duration_after = start.elapsed().as_micros();
        println!(
            "Query result: {} neighbors, time {}μs",
            results_after.len(),
            duration_after
        );
        assert_eq!(
            results_after.len(),
            10,
            "Should still find 10 nearest neighbors after deletion"
        );
        for result in &results_after {
            assert!(
                result.id >= delete_count || !hnsw.is_deleted(result.id),
                "Should not return deleted nodes"
            );
        }
        assert_eq!(
            final_stats.node_count, num_vectors,
            "Should insert correct number of nodes"
        );
        assert!(
            after_delete_stats.node_count == num_vectors - delete_count,
            "Node count should decrease after deletion"
        );
        assert!(
            avg_query_time < 5000.0,
            "Query performance should be reasonable"
        );
        assert!(
            qps > 100.0,
            "Should process at least 100 queries per second"
        );
        println!(
            "Total test time estimate: {:?}",
            Instant::now().duration_since(insert_start)
        );
    }

    #[test]
    fn test_hnsw_10k_quick() {
        println!("10k vectors quick test");
        let config = HnswConfig {
            max_elements: 10_000,
            m: 8,
            m_max: 16,
            ef_construction: 50,
            ef_search: 20,
            ..Default::default()
        };
        let mut hnsw = HNSW::new(config, EuclideanDistance::new());
        let mut rng = rand::thread_rng();
        let dim = 16;
        let num_vectors = 10_000;
        println!("Inserting {} vectors", num_vectors);
        let start = Instant::now();
        for i in 0..num_vectors {
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            hnsw.insert(vector);
        }
        let insert_time = start.elapsed();
        println!("Insert time: {:?}", insert_time);
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let query_start = Instant::now();
        let results = hnsw.search_knn(&query, 5);
        let query_time = query_start.elapsed();
        println!(
            "Query time: {:?}, found {} results",
            query_time,
            results.len()
        );
        let stats = hnsw.stats();
        println!(
            "Statistics: node_count={}, avg_connections={:.2}",
            stats.node_count, stats.avg_connections
        );
        assert_eq!(results.len(), 5);
        assert_eq!(stats.node_count, num_vectors);
    }

    #[test]
    fn test_hnsw_different_configs() {
        println!("Testing different configuration parameters");
        let test_configs = vec![
            (
                "Sparse connections",
                HnswConfig {
                    max_elements: 10_000,
                    m: 4,
                    m_max: 8,
                    ef_construction: 20,
                    ef_search: 10,
                    ..Default::default()
                },
            ),
            (
                "Medium connections",
                HnswConfig {
                    max_elements: 10_000,
                    m: 16,
                    m_max: 32,
                    ef_construction: 100,
                    ef_search: 30,
                    ..Default::default()
                },
            ),
            (
                "Dense connections",
                HnswConfig {
                    max_elements: 10_000,
                    m: 32,
                    m_max: 64,
                    ef_construction: 200,
                    ef_search: 50,
                    ..Default::default()
                },
            ),
        ];
        for (name, config) in test_configs {
            println!("\nTesting configuration: {}", name);
            let mut hnsw = HNSW::new(config, EuclideanDistance::new());
            let mut rng = rand::thread_rng();
            let dim = 16;
            let num_vectors = 2_000;
            let insert_start = Instant::now();
            for _ in 0..num_vectors {
                let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
                hnsw.insert(vector);
            }
            let insert_time = insert_start.elapsed();
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            let query_start = Instant::now();
            let results = hnsw.search_knn(&query, 5);
            let query_time = query_start.elapsed();
            let stats = hnsw.stats();
            println!("  Insert time: {:?}", insert_time);
            println!("  Query time: {:?}", query_time);
            println!(
                "  Node count: {}, avg_connections: {:.2}, max_level: {}",
                stats.node_count, stats.avg_connections, stats.max_level
            );
            println!("  Query results: {} found", results.len());
        }
    }
}
