use hnswx;

#[cfg(test)]
mod querying_1million_data {
    use super::*;
    use hnswx::{EuclideanDistance, HNSW, HnswConfig};
    use rand::Rng;
    use std::time::Instant;

    #[test]
    fn test_hnsw_1_million_vectors() {
        println!("=== HNSW 1 Million Vectors Performance Test ===");
        let config = HnswConfig {
            max_elements: 1_000_000,
            m: 16,
            m_max: 32,
            m_max_0: 64,
            ef_construction: 200,
            ef_search: 100,
            level_multiplier: 1.0 / f64::ln(32.0),
            ..Default::default()
        };
        let mut hnsw = HNSW::new(config.clone(), EuclideanDistance::new());
        let mut rng = rand::thread_rng();
        let dim = 32; // 32-dimensional vectors
        let num_vectors = 1_000_000;
        println!("Dimension: {}D", dim);
        println!("Number of vectors: {}", num_vectors);
        println!(
            "Configuration: m={}, ef_construction={}, ef_search={}",
            config.m, config.ef_construction, config.ef_search
        );
        println!("\n--- Phase 1: Inserting 1 million vectors ---");
        let insert_start = Instant::now();
        let mut last_report_time = Instant::now();
        for i in 0..num_vectors {
            // Generate random vector
            let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            hnsw.insert(vector);
            if i % 50_000 == 0 && i > 0 {
                let elapsed = last_report_time.elapsed();
                if elapsed.as_secs() >= 60 || i % 100_000 == 0 {
                    println!(
                        "  Inserted: {} ({:.1}%)",
                        i,
                        (i as f32 / num_vectors as f32) * 100.0
                    );
                    if i % 200_000 == 0 {
                        let stats = hnsw.stats();
                        println!(
                            "    Current stats: nodes={}, max_level={}, avg_connections={:.2}",
                            stats.node_count, stats.max_level, stats.avg_connections
                        );
                    }
                    last_report_time = Instant::now();
                }
            }
        }
        let insert_duration = insert_start.elapsed();
        println!("Insert completed, total time: {:?}", insert_duration);
        println!(
            "Average insertion time per vector: {:?}",
            insert_duration / num_vectors as u32
        );
        let initial_stats = hnsw.stats();
        println!("\nIndex statistics after insertion:");
        println!("  Total nodes: {}", initial_stats.node_count);
        println!("  Maximum level: {}", initial_stats.max_level);
        println!(
            "  Average connections: {:.2}",
            initial_stats.avg_connections
        );
        println!("  Maximum connections: {}", initial_stats.max_connections);
        println!("  Entry point ID: {:?}", initial_stats.entry_point);
        println!("  Storage size: {} floats", initial_stats.storage_size);
        println!("\n--- Phase 2: Query performance test ---");
        let num_queries = 200;
        let mut query_times = Vec::with_capacity(num_queries);
        let test_queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();
        println!("Executing {} query tests...", num_queries);
        for (i, query) in test_queries.iter().enumerate() {
            let query_start = Instant::now();
            let results = hnsw.search_knn(query, 10);
            let query_duration = query_start.elapsed().as_micros();
            query_times.push(query_duration);
            assert_eq!(results.len(), 10, "Should find 10 nearest neighbors");
            for j in 1..results.len() {
                assert!(
                    results[j - 1].distance <= results[j].distance,
                    "Results should be sorted by distance"
                );
            }
            if i % 40 == 0 && i > 0 {
                let avg_so_far: f64 =
                    query_times.iter().map(|&t| t as f64).sum::<f64>() / query_times.len() as f64;
                println!("  Completed queries: {} (average {:.2}μs)", i, avg_so_far);
            }
        }
        let total_query_time: u128 = query_times.iter().sum();
        let avg_query_time = total_query_time as f64 / num_queries as f64;
        let min_query_time = *query_times.iter().min().unwrap();
        let max_query_time = *query_times.iter().max().unwrap();
        let qps = 1_000_000.0 / avg_query_time;
        let mut sorted_times = query_times.clone();
        sorted_times.sort_unstable();
        let p50 = sorted_times[num_queries / 2];
        let p90 = sorted_times[(num_queries * 9) / 10];
        let p95 = sorted_times[(num_queries * 19) / 20];
        println!(
            "\nQuery performance statistics ({} queries average):",
            num_queries
        );
        println!("  Average query time: {:.2}μs", avg_query_time);
        println!("  Minimum query time: {}μs", min_query_time);
        println!("  Maximum query time: {}μs", max_query_time);
        println!("  Median (P50): {}μs", p50);
        println!("  P90: {}μs", p90);
        println!("  P95: {}μs", p95);
        println!("  QPS (queries per second): {:.2}", qps);
        println!("\n--- Phase 3: Query performance with different k values ---");
        let test_k_values = [1, 5, 10, 20, 50, 100];
        let test_query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        for &k in &test_k_values {
            // Warm-up
            hnsw.search_knn(&test_query, k);
            let mut times = Vec::new();
            for _ in 0..10 {
                let start = Instant::now();
                let results = hnsw.search_knn(&test_query, k);
                let duration = start.elapsed().as_micros();
                times.push(duration);
                assert_eq!(results.len(), k, "Should find {} nearest neighbors", k);
            }
            let avg_time: f64 = times.iter().map(|&t| t as f64).sum::<f64>() / times.len() as f64;
            println!("  k={:3}: Average time {:.2}μs, results {}", k, avg_time, k);
        }
        println!("\n--- Phase 4: Delete operation test ---");
        let delete_batch_1 = 100_000; // Delete first 100k nodes
        let delete_batch_2 = 50_000; // Delete another 50k nodes
        println!("Batch delete 1: Delete first {} nodes", delete_batch_1);
        let delete_start_1 = Instant::now();
        let mut deleted_count_1 = 0;
        for i in 0..delete_batch_1 {
            if hnsw.delete(i) {
                deleted_count_1 += 1;
            }
            if i > 0 && i % 20_000 == 0 {
                println!("  Deleted: {} nodes", i);
            }
        }
        let delete_duration_1 = delete_start_1.elapsed();
        println!("  Delete completed, time: {:?}", delete_duration_1);
        println!("  Successfully deleted: {} nodes", deleted_count_1);
        println!(
            "  Average delete time: {:?}",
            delete_duration_1 / delete_batch_1 as u32
        );
        let after_delete_1_stats = hnsw.stats();
        println!("\n  Statistics after first delete:");
        println!("    Active nodes: {}", after_delete_1_stats.node_count);
        println!("    Deleted nodes: {}", after_delete_1_stats.deleted_count);
        println!(
            "    Average connections: {:.2}",
            after_delete_1_stats.avg_connections
        );
        println!("\n--- Phase 5: Query test after deletion ---");
        let delete_query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let query_after_delete_start = Instant::now();
        let results_after_delete = hnsw.search_knn(&delete_query, 10);
        let query_after_delete_time = query_after_delete_start.elapsed().as_micros();
        println!(
            "  Query after deletion: {} nearest neighbors, time {}μs",
            results_after_delete.len(),
            query_after_delete_time
        );
        for result in &results_after_delete {
            assert!(
                result.id >= delete_batch_1 || !hnsw.is_deleted(result.id),
                "Query results should not contain deleted nodes (ID: {})",
                result.id
            );
        }
        println!("\n--- Phase 6: Second delete operation ---");
        println!(
            "Batch delete 2: Delete nodes with ID {}~{}",
            delete_batch_1,
            delete_batch_1 + delete_batch_2
        );
        let delete_start_2 = Instant::now();
        let mut deleted_count_2 = 0;
        for i in delete_batch_1..(delete_batch_1 + delete_batch_2) {
            if hnsw.delete(i) {
                deleted_count_2 += 1;
            }

            if i > delete_batch_1 && (i - delete_batch_1) % 10_000 == 0 {
                println!("  Deleted: {} nodes", i - delete_batch_1);
            }
        }
        let delete_duration_2 = delete_start_2.elapsed();
        println!("  Delete completed, time: {:?}", delete_duration_2);
        println!("  Successfully deleted: {} nodes", deleted_count_2);
        println!(
            "  Average delete time: {:?}",
            delete_duration_2 / delete_batch_2 as u32
        );
        let final_stats = hnsw.stats();
        println!("\n  Final statistics:");
        println!("    Active nodes: {}", final_stats.node_count);
        println!("    Deleted nodes: {}", final_stats.deleted_count);
        println!("    Total inserted nodes: {}", num_vectors);
        println!(
            "    Remaining node percentage: {:.1}%",
            (final_stats.node_count as f32 / num_vectors as f32) * 100.0
        );
        println!("\n--- Phase 7: Final query test ---");
        let final_query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let mut final_query_times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            let results = hnsw.search_knn(&final_query, 10);
            final_query_times.push(start.elapsed().as_micros());
            assert_eq!(
                results.len(),
                10,
                "Final query should find 10 nearest neighbors"
            );
        }
        let avg_final_query_time: f64 = final_query_times.iter().map(|&t| t as f64).sum::<f64>()
            / final_query_times.len() as f64;
        println!(
            "  Final query performance (20 queries average): {:.2}μs",
            avg_final_query_time
        );
        println!("\n--- Test assertion verification ---");
        assert_eq!(
            initial_stats.node_count, num_vectors,
            "Should insert correct number of nodes"
        );
        assert_eq!(
            final_stats.node_count,
            num_vectors - deleted_count_1 - deleted_count_2,
            "Node count should decrease correctly after deletion"
        );
        assert!(
            avg_query_time < 10000.0, // Within 10 milliseconds
            "Query performance should be reasonable (average: {:.2}μs)",
            avg_query_time
        );
        assert!(
            qps > 50.0, // At least 50 queries per second
            "Should process at least 50 queries per second (actual: {:.2} QPS)",
            qps
        );
        let total_test_time = insert_start.elapsed();
        println!("\n=== Test completed ===");
        println!("Total test time: {:?}", total_test_time);
        println!(
            "Memory usage estimate: ~{:.1} MB",
            (final_stats.storage_size * 4) as f64 / 1024.0 / 1024.0
        );
        println!("\nPerformance summary:");
        println!(
            "  Insertion speed: {:.1} vectors/sec",
            num_vectors as f64 / insert_duration.as_secs_f64()
        );
        println!("  Query speed: {:.1} queries/sec", qps);
        println!(
            "  Delete speed: {:.1} deletes/sec",
            (deleted_count_1 + deleted_count_2) as f64
                / (delete_duration_1 + delete_duration_2).as_secs_f64()
        );
    }

    #[test]
    fn test_hnsw_1m_scalability() {
        println!("\n=== HNSW 1 Million Vectors Scalability Test ===");
        let scalability_configs = vec![
            (
                "Conservative configuration (memory friendly)",
                HnswConfig {
                    max_elements: 1_000_000,
                    m: 8,
                    m_max: 16,
                    m_max_0: 32,
                    ef_construction: 100,
                    ef_search: 50,
                    ..Default::default()
                },
            ),
            (
                "Balanced configuration (recommended)",
                HnswConfig {
                    max_elements: 1_000_000,
                    m: 16,
                    m_max: 32,
                    m_max_0: 64,
                    ef_construction: 200,
                    ef_search: 100,
                    ..Default::default()
                },
            ),
            (
                "High-performance configuration (query optimized)",
                HnswConfig {
                    max_elements: 1_000_000,
                    m: 24,
                    m_max: 48,
                    m_max_0: 96,
                    ef_construction: 400,
                    ef_search: 200,
                    ..Default::default()
                },
            ),
        ];
        for (config_name, config) in scalability_configs.iter().take(1) {
            println!("\nTesting configuration: {}", config_name);
            println!(
                "Parameters: m={}, m_max={}, ef_construction={}, ef_search={}",
                config.m, config.m_max, config.ef_construction, config.ef_search
            );
            let mut hnsw = HNSW::new(config.clone(), EuclideanDistance::new());
            let mut rng = rand::thread_rng();
            let dim = 32;
            let test_vectors = 100_000; // Test with 100k vectors
            println!("Inserting {} test vectors...", test_vectors);
            let insert_start = Instant::now();
            for i in 0..test_vectors {
                let vector: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
                hnsw.insert(vector);
            }
            let insert_time = insert_start.elapsed();
            println!("Insert completed: {:?}", insert_time);
            println!(
                "Insertion speed: {:.1} vectors/sec",
                test_vectors as f64 / insert_time.as_secs_f64()
            );
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
            let mut query_times = Vec::new();
            for _ in 0..20 {
                let start = Instant::now();
                hnsw.search_knn(&query, 10);
                query_times.push(start.elapsed().as_micros());
            }
            let avg_query: f64 =
                query_times.iter().map(|&t| t as f64).sum::<f64>() / query_times.len() as f64;
            println!(
                "Query performance: average {:.2}μs, {:.1} QPS",
                avg_query,
                1_000_000.0 / avg_query
            );
            let stats = hnsw.stats();
            println!(
                "Memory usage: {:.1} MB",
                (stats.storage_size * 4) as f64 / 1024.0 / 1024.0
            );
            println!("Average connections: {:.2}", stats.avg_connections);
        }
    }
}
