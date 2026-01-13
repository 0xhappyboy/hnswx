#[cfg(test)]
mod recall_tests {
    use super::*;
    use hashbrown::HashSet;
    use hnswx::{DistanceMetric, EuclideanDistance, HNSW, HnswConfig, SearchResult};
    use rand::Rng;
    use rayon::prelude::*;
    use std::time::Instant;

    fn brute_force_search(
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        metric: &EuclideanDistance,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| SearchResult::new(id, metric.distance_squared(query, vec).sqrt()))
            .collect();

        results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        results
    }

    fn calculate_recall(
        hnsw_results: &[SearchResult],
        ground_truth: &[SearchResult],
        k: usize,
    ) -> f32 {
        let hnsw_ids: HashSet<usize> = hnsw_results.iter().take(k).map(|r| r.id).collect();
        let ground_truth_ids: HashSet<usize> = ground_truth.iter().take(k).map(|r| r.id).collect();

        let intersection: HashSet<_> = hnsw_ids.intersection(&ground_truth_ids).collect();
        intersection.len() as f32 / k.min(ground_truth.len()) as f32
    }

    #[test]
    fn test_recall_with_different_ef() {
        println!("=== Recall test with different ef_search parameters ===");

        let mut rng = rand::thread_rng();
        let dim = 32;
        let num_vectors = 10_000;
        let num_queries = 100;
        let k = 10;

        println!("Generating {} {}D test vectors...", num_vectors, dim);
        let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        println!("Generating {} query vectors...", num_queries);
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        println!("Calculating ground truth (brute-force search)...");
        let metric = EuclideanDistance::new();
        let ground_truth: Vec<Vec<SearchResult>> = queries
            .par_iter()
            .map(|query| brute_force_search(&test_vectors, query, k, &metric))
            .collect();

        let ef_values = [5, 10, 20, 40, 80, 160, 320];
        let mut results = Vec::new();

        for &ef_search in &ef_values {
            println!("\nTesting ef_search = {}", ef_search);

            let config = HnswConfig {
                max_elements: num_vectors,
                m: 16,
                m_max: 32,
                m_max_0: 64,
                ef_construction: 200,
                ef_search,
                ..Default::default()
            };

            let mut hnsw = HNSW::new(config, EuclideanDistance::new());

            println!("  Inserting vectors into HNSW...");
            let insert_start = Instant::now();
            for vector in &test_vectors {
                hnsw.insert(vector.clone());
            }
            let insert_time = insert_start.elapsed();
            println!("  Insert time: {:?}", insert_time);

            println!("  Executing queries and calculating recall...");
            let query_start = Instant::now();
            let mut total_recall = 0.0;
            let mut query_times = Vec::new();

            for (i, query) in queries.iter().enumerate() {
                let start = Instant::now();
                let hnsw_results = hnsw.search_knn(query, k);
                let query_time = start.elapsed().as_micros();
                query_times.push(query_time);

                let recall = calculate_recall(&hnsw_results, &ground_truth[i], k);
                total_recall += recall;
            }

            let avg_recall = total_recall / num_queries as f32;
            let avg_query_time = query_times.iter().sum::<u128>() as f64 / query_times.len() as f64;

            results.push((ef_search, avg_recall, avg_query_time, insert_time));

            println!("  Average recall: {:.4}", avg_recall);
            println!("  Average query time: {:.2}μs", avg_query_time);
            println!("  QPS: {:.2}", 1_000_000.0 / avg_query_time);
        }

        println!("\n=== Recall test results summary ===");
        println!("Data: {} {}D vectors", num_vectors, dim);
        println!("Queries: {} times, k={}", num_queries, k);
        println!("\n| ef_search | Recall | Avg Query Time(μs) | QPS | Insert Time |");
        println!("|-----------|--------|-----------------|-----|----------|");

        for (ef, recall, query_time, insert_time) in results {
            println!(
                "| {:9} | {:.4}  | {:15.2} | {:4.0} | {:8?} |",
                ef,
                recall,
                query_time,
                1_000_000.0 / query_time,
                insert_time
            );
        }
    }

    #[test]
    fn test_recall_with_different_dataset_sizes() {
        println!("=== Recall test with different dataset sizes ===");

        let mut rng = rand::thread_rng();
        let dim = 32;
        let k = 10;
        let ef_search = 50;
        let num_queries = 50;

        let dataset_sizes = [1_000, 5_000, 10_000, 20_000, 30_000];
        let mut results = Vec::new();

        for &num_vectors in &dataset_sizes {
            println!("\nTesting dataset size: {}", num_vectors);

            println!("  Generating {} {}D test vectors...", num_vectors, dim);
            let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();

            let queries: Vec<Vec<f32>> = (0..num_queries)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();

            println!("  Calculating ground truth...");
            let metric = EuclideanDistance::new();
            let ground_truth: Vec<Vec<SearchResult>> = queries
                .par_iter()
                .map(|query| brute_force_search(&test_vectors, query, k, &metric))
                .collect();

            let config = HnswConfig {
                max_elements: num_vectors,
                m: 16,
                m_max: 32,
                m_max_0: 64,
                ef_construction: 200,
                ef_search,
                ..Default::default()
            };

            let mut hnsw = HNSW::new(config, EuclideanDistance::new());

            println!("  Inserting vectors into HNSW...");
            let insert_start = Instant::now();
            hnsw.insert_batch(test_vectors.clone());
            let insert_time = insert_start.elapsed();

            println!("  Executing queries and calculating recall...");
            let mut total_recall = 0.0;
            let mut query_times = Vec::new();

            for (i, query) in queries.iter().enumerate() {
                let start = Instant::now();
                let hnsw_results = hnsw.search_knn(query, k);
                let query_time = start.elapsed().as_micros();
                query_times.push(query_time);

                let recall = calculate_recall(&hnsw_results, &ground_truth[i], k);
                total_recall += recall;
            }

            let avg_recall = total_recall / num_queries as f32;
            let avg_query_time = query_times.iter().sum::<u128>() as f64 / query_times.len() as f64;

            let stats = hnsw.stats();
            results.push((
                num_vectors,
                avg_recall,
                avg_query_time,
                insert_time,
                stats.avg_connections,
            ));

            println!("  Average recall: {:.4}", avg_recall);
            println!("  Average query time: {:.2}μs", avg_query_time);
            println!("  Average connections: {:.2}", stats.avg_connections);
        }

        println!("\n=== Dataset size effect on recall results summary ===");
        println!("ef_search = {}, k = {}, dimension = {}", ef_search, k, dim);
        println!(
            "\n| Data Size | Recall | Avg Query Time(μs) | QPS | Insert Time | Avg Connections |"
        );
        println!("|--------|--------|-----------------|-----|----------|------------|");

        for (size, recall, query_time, insert_time, avg_conn) in results {
            println!(
                "| {:6} | {:.4}  | {:15.2} | {:4.0} | {:8?} | {:10.2} |",
                size,
                recall,
                query_time,
                1_000_000.0 / query_time,
                insert_time,
                avg_conn
            );
        }
    }

    #[test]
    fn test_recall_with_different_dimensions() {
        println!("=== Recall test with different dimensions ===");

        let mut rng = rand::thread_rng();
        let num_vectors = 10_000;
        let num_queries = 50;
        let k = 10;
        let ef_search = 50;

        let dimensions = [16, 32, 64, 128, 256];
        let mut results = Vec::new();

        for &dim in &dimensions {
            println!("\nTesting dimension: {}D", dim);

            println!("  Generating {} {}D test vectors...", num_vectors, dim);
            let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();

            let queries: Vec<Vec<f32>> = (0..num_queries)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();

            println!("  Calculating ground truth...");
            let metric = EuclideanDistance::new();
            let ground_truth: Vec<Vec<SearchResult>> = queries
                .par_iter()
                .map(|query| brute_force_search(&test_vectors, query, k, &metric))
                .collect();

            let config = HnswConfig {
                max_elements: num_vectors,
                m: 16,
                m_max: 32,
                m_max_0: 64,
                ef_construction: 200,
                ef_search,
                ..Default::default()
            };

            let mut hnsw = HNSW::new(config, EuclideanDistance::new());

            println!("  Inserting vectors into HNSW...");
            let insert_start = Instant::now();
            hnsw.insert_batch(test_vectors.clone());
            let insert_time = insert_start.elapsed();

            println!("  Executing queries and calculating recall...");
            let mut total_recall = 0.0;
            let mut query_times = Vec::new();

            for (i, query) in queries.iter().enumerate() {
                let start = Instant::now();
                let hnsw_results = hnsw.search_knn(query, k);
                let query_time = start.elapsed().as_micros();
                query_times.push(query_time);

                let recall = calculate_recall(&hnsw_results, &ground_truth[i], k);
                total_recall += recall;
            }

            let avg_recall = total_recall / num_queries as f32;
            let avg_query_time = query_times.iter().sum::<u128>() as f64 / query_times.len() as f64;

            let stats = hnsw.stats();
            results.push((
                dim,
                avg_recall,
                avg_query_time,
                insert_time,
                stats.avg_connections,
            ));

            println!("  Average recall: {:.4}", avg_recall);
            println!("  Average query time: {:.2}μs", avg_query_time);
            println!("  Average connections: {:.2}", stats.avg_connections);
        }

        println!("\n=== Dimension effect on recall results summary ===");
        println!(
            "Data size = {}, ef_search = {}, k = {}",
            num_vectors, ef_search, k
        );
        println!(
            "\n| Dimension | Recall | Avg Query Time(μs) | QPS | Insert Time | Avg Connections |"
        );
        println!("|------|--------|-----------------|-----|----------|------------|");

        for (dim, recall, query_time, insert_time, avg_conn) in results {
            println!(
                "| {:4} | {:.4}  | {:15.2} | {:4.0} | {:8?} | {:10.2} |",
                dim,
                recall,
                query_time,
                1_000_000.0 / query_time,
                insert_time,
                avg_conn
            );
        }
    }

    #[test]
    fn test_comprehensive_recall_analysis() {
        println!("=== Comprehensive recall test and precision-speed trade-off analysis ===");

        let mut rng = rand::thread_rng();
        let dim = 64;
        let num_vectors = 50_000;
        let num_queries = 100;
        let k_values = [1, 5, 10, 20, 50];

        println!("Generating {} {}D test vectors...", num_vectors, dim);
        let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        println!("Generating {} query vectors...", num_queries);
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        println!("Calculating ground truth (brute-force search)...");
        let metric = EuclideanDistance::new();
        let mut ground_truth_by_k = Vec::new();

        for &k in &k_values {
            println!("  Calculating ground truth for k={}...", k);
            let ground_truth: Vec<Vec<SearchResult>> = queries
                .par_iter()
                .map(|query| brute_force_search(&test_vectors, query, k, &metric))
                .collect();
            ground_truth_by_k.push((k, ground_truth));
        }

        let configs = vec![
            (
                "Fast-LowPrecision",
                HnswConfig {
                    max_elements: num_vectors,
                    m: 8,
                    m_max: 16,
                    m_max_0: 32,
                    ef_construction: 100,
                    ef_search: 20,
                    ..Default::default()
                },
            ),
            (
                "Balanced",
                HnswConfig {
                    max_elements: num_vectors,
                    m: 16,
                    m_max: 32,
                    m_max_0: 64,
                    ef_construction: 200,
                    ef_search: 50,
                    ..Default::default()
                },
            ),
            (
                "HighPrecision-Slow",
                HnswConfig {
                    max_elements: num_vectors,
                    m: 24,
                    m_max: 48,
                    m_max_0: 96,
                    ef_construction: 400,
                    ef_search: 100,
                    ..Default::default()
                },
            ),
        ];

        let mut all_results = Vec::new();

        for (config_name, config) in configs {
            println!("\nTesting configuration: {}", config_name);

            let mut hnsw = HNSW::new(config, EuclideanDistance::new());

            let insert_start = Instant::now();
            hnsw.insert_batch(test_vectors.clone());
            let insert_time = insert_start.elapsed();

            let stats = hnsw.stats();
            println!("  Insert time: {:?}", insert_time);
            println!("  Average connections: {:.2}", stats.avg_connections);
            println!("  Maximum level: {}", stats.max_level);

            let mut config_results = Vec::new();

            for &k in &k_values {
                let (_, ground_truth) = ground_truth_by_k
                    .iter()
                    .find(|(k_val, _)| *k_val == k)
                    .unwrap();

                let mut total_recall = 0.0;
                let mut query_times = Vec::new();

                for (i, query) in queries.iter().enumerate() {
                    let start = Instant::now();
                    let hnsw_results = hnsw.search_knn(query, k);
                    let query_time = start.elapsed().as_micros();
                    query_times.push(query_time);

                    let recall = calculate_recall(&hnsw_results, &ground_truth[i], k);
                    total_recall += recall;
                }

                let avg_recall = total_recall / num_queries as f32;
                let avg_query_time =
                    query_times.iter().sum::<u128>() as f64 / query_times.len() as f64;

                config_results.push((k, avg_recall, avg_query_time));
                println!(
                    "  k={}: Recall={:.4}, Query time={:.2}μs",
                    k, avg_recall, avg_query_time
                );
            }

            all_results.push((config_name, config_results, insert_time, stats));
        }

        println!("\n=== Comprehensive recall analysis results ===");
        println!("Data: {} {}D vectors", num_vectors, dim);
        println!("Queries: {} times", num_queries);

        for (config_name, config_results, insert_time, stats) in &all_results {
            println!("\nConfiguration: {}", config_name);
            println!(
                "Insert time: {:?}, Average connections: {:.2}, Maximum level: {}",
                insert_time, stats.avg_connections, stats.max_level
            );

            println!("| k | Recall | Query Time(μs) | QPS |");
            println!("|---|---|---|---|");

            for &(k, recall, query_time) in config_results {
                println!(
                    "| {} | {:.4} | {:.2} | {:.0} |",
                    k,
                    recall,
                    query_time,
                    1_000_000.0 / query_time
                );
            }
        }

        println!("\n=== Precision-speed trade-off recommendations ===");
        println!(
            "1. For applications requiring high recall (>0.95), recommend 'HighPrecision-Slow' configuration"
        );
        println!(
            "2. For applications requiring fast response (<100μs), recommend 'Fast-LowPrecision' configuration"
        );
        println!("3. For most balanced scenarios, recommend 'Balanced' configuration");
        println!(
            "4. ef_search has the greatest impact on recall, adjust according to recall requirements"
        );
    }

    #[test]
    fn test_recall_after_deletion() {
        println!("=== Recall test after deletion operations ===");

        let mut rng = rand::thread_rng();
        let dim = 32;
        let num_vectors = 10_000;
        let delete_count = 1_000;
        let num_queries = 50;
        let k = 10;
        let ef_search = 50;

        println!("Generating {} {}D test vectors...", num_vectors, dim);
        let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        let config = HnswConfig {
            max_elements: num_vectors,
            m: 16,
            m_max: 32,
            m_max_0: 64,
            ef_construction: 200,
            ef_search,
            ..Default::default()
        };

        let mut hnsw = HNSW::new(config, EuclideanDistance::new());

        println!("Inserting vectors into HNSW...");
        hnsw.insert_batch(test_vectors.clone());

        println!("\nTesting recall before deletion...");
        let metric = EuclideanDistance::new();
        let ground_truth_before: Vec<Vec<SearchResult>> = queries
            .par_iter()
            .map(|query| brute_force_search(&test_vectors, query, k, &metric))
            .collect();

        let mut recall_before = 0.0;
        for (i, query) in queries.iter().enumerate() {
            let hnsw_results = hnsw.search_knn(query, k);
            let recall = calculate_recall(&hnsw_results, &ground_truth_before[i], k);
            recall_before += recall;
        }
        recall_before /= num_queries as f32;

        println!("\nDeleting {} nodes...", delete_count);
        for i in 0..delete_count {
            hnsw.delete(i);
        }

        println!("Recalculating ground truth (excluding deleted nodes)...");
        let active_vectors: Vec<Vec<f32>> = test_vectors[delete_count..].to_vec();
        let ground_truth_after: Vec<Vec<SearchResult>> = queries
            .par_iter()
            .map(|query| {
                brute_force_search(&active_vectors, query, k, &metric)
                    .into_iter()
                    .map(|mut r| {
                        r.id += delete_count;
                        r
                    })
                    .collect()
            })
            .collect();

        println!("\nTesting recall after deletion...");
        let mut recall_after = 0.0;
        let mut contains_deleted = 0;

        for (i, query) in queries.iter().enumerate() {
            let hnsw_results = hnsw.search_knn(query, k);

            let has_deleted = hnsw_results.iter().any(|r| r.id < delete_count);
            if has_deleted {
                contains_deleted += 1;
            }

            let recall = calculate_recall(&hnsw_results, &ground_truth_after[i], k);
            recall_after += recall;
        }
        recall_after /= num_queries as f32;

        let stats_before = hnsw.stats();
        println!("\n=== Deletion operation effect on recall results ===");
        println!("Dataset: {} {}D vectors", num_vectors, dim);
        println!(
            "Deleted: {} nodes ({:.1}%)",
            delete_count,
            delete_count as f32 / num_vectors as f32 * 100.0
        );
        println!(
            "Queries: {} times, k={}, ef_search={}",
            num_queries, k, ef_search
        );
        println!();
        println!("Recall before deletion: {:.4}", recall_before);
        println!("Recall after deletion: {:.4}", recall_after);
        println!("Recall change: {:.4}", recall_after - recall_before);
        println!(
            "Queries containing deleted nodes: {} ({:.1}%)",
            contains_deleted,
            contains_deleted as f32 / num_queries as f32 * 100.0
        );
        println!();
        println!("Statistics after deletion:");
        println!("  Active node count: {}", stats_before.node_count);
        println!("  Deleted node count: {}", stats_before.deleted_count);
        println!("  Average connections: {:.2}", stats_before.avg_connections);
        println!("  Maximum level: {}", stats_before.max_level);
        assert!(
            recall_after > 0.8,
            "Recall should remain high after deletion"
        );
        assert!(
            contains_deleted < num_queries / 10,
            "Should rarely return deleted nodes"
        );
    }
}
