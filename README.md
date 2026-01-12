<h1 align="center">
    HNSWX
</h1>
<h4 align="center">
A Rust implementation of the Hierarchical Navigable Small World (HNSW) algorithm. HNSW is an efficient approximate nearest neighbor search algorithm, particularly suitable for high-dimensional vector retrieval.
</h4>
<p align="center">
  <a href="https://github.com/0xhappyboy/hnswx/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
    <a href="https://crates.io/crates/hnswx">
<img src="https://img.shields.io/badge/crates-hnswx-20B2AA.svg?style=flat&labelColor=0F1F2D&color=FFD700&logo=rust&logoColor=FFD700">
</a>
</p>
<p align="center">
<a href="./README_zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

# Installation

```
cargo add hnswx
```

# Test

## The test inserts 1 million 32-dimensional vectors into the HNSW index and performs query and delete operations.

```
cargo test test_hnsw_1_million_vectors -- --nocapture
```

### result

```
=== HNSW 1 Million Vectors Performance Test ===
Dimension: 32D
Number of vectors: 1000000
Configuration: m=16, ef_construction=200, ef_search=100

--- Phase 1: Inserting 1 million vectors ---
test massive_scale_test::test_hnsw_1_million_vectors has been running for over 60 seconds
Inserted: 100000 (10.0%)
Inserted: 200000 (20.0%)
Current stats: nodes=200001, max_level=3, avg_connections=15.34
Inserted: 300000 (30.0%)
Inserted: 400000 (40.0%)
Current stats: nodes=400001, max_level=3, avg_connections=15.33
Inserted: 500000 (50.0%)
Inserted: 600000 (60.0%)
Current stats: nodes=600001, max_level=3, avg_connections=15.32
Inserted: 700000 (70.0%)
Inserted: 800000 (80.0%)
Current stats: nodes=800001, max_level=3, avg_connections=15.32
Inserted: 900000 (90.0%)
Insertion completed, total time: 950.1687562s
Average insertion time per vector: 950.168µs

Index statistics after insertion:
Total nodes: 1000000
Maximum level: 3
Average connections: 15.32
Maximum connections: 152
Entry point ID: Some(12931)
Storage size: 32000000 floats

--- Phase 2: Query performance test ---
Executing 200 query tests...
Completed queries: 40 (average 1141.37μs)
Completed queries: 80 (average 1113.62μs)
Completed queries: 120 (average 1077.25μs)
Completed queries: 160 (average 1074.22μs)

Query performance statistics (200 queries average):
Average query time: 1068.71μs
Minimum query time: 41μs
Maximum query time: 2506μs
Median (P50): 1089μs
P90: 1553μs
P95: 1976μs
QPS (queries per second): 935.71

--- Phase 3: Query performance with different k values ---
k= 1: Average time 1348.50μs, results 1
k= 5: Average time 1151.60μs, results 5
k= 10: Average time 1126.40μs, results 10
k= 20: Average time 1433.50μs, results 20
k= 50: Average time 1331.70μs, results 50
k=100: Average time 1152.40μs, results 100

--- Phase 4: Delete operation test ---
Batch delete 1: Delete first 100000 nodes
Deleted: 20000 nodes
Deleted: 40000 nodes
Deleted: 60000 nodes
Deleted: 80000 nodes
Delete completed, time: 368.2621ms
Successfully deleted: 100000 nodes
Average delete time: 3.682µs

Statistics after first delete:
Active nodes: 900000
Deleted nodes: 100000
Average connections: 15.07

--- Phase 5: Query test after deletion ---
Query after deletion: 10 nearest neighbors, time 1612μs

--- Phase 6: Second delete operation ---
Batch delete 2: Delete nodes with ID 100000~150000
Deleted: 10000 nodes
Deleted: 20000 nodes
Deleted: 30000 nodes
Deleted: 40000 nodes
Delete completed, time: 213.4648ms
Successfully deleted: 50000 nodes
Average delete time: 4.269µs

Final statistics:
Active nodes: 850000
Deleted nodes: 150000
Total inserted nodes: 1000000
Remaining node percentage: 85.0%

--- Phase 7: Final query test ---
Final query performance (20 queries average): 1313.65μs

--- Test assertion verification ---

=== Test completed ===
Total test time: 951.266665s
Memory usage estimate: ~122.1 MB

Performance summary:
Insertion speed: 1052.4 vectors/sec
Query speed: 935.7 queries/sec
Delete speed: 257853.0 deletes/sec
```

# Quick Start

## Basic Usage

```rust
use hnswx::*;

fn main() {
    let config = HnswConfig::default();
    let mut hnsw = HNSW::new(config, EuclideanDistance);
    let id1 = hnsw.insert(vec![1.0, 2.0, 3.0, 4.0]);
    let id2 = hnsw.insert(vec![2.0, 3.0, 4.0, 5.0]);
    let id3 = hnsw.insert(vec![3.0, 4.0, 5.0, 6.0]);
    println!("Inserted node IDs: {}, {}, {}", id1, id2, id3);
    let query = vec![2.1, 3.1, 4.1, 5.1];
    let results = hnsw.search_knn(&query, 2);
    println!("Search results:");
    for result in results {
        println!("  Node ID: {}, Distance: {:.4}", result.id, result.distance);
    }
    let stats = hnsw.stats();
    println!("Statistics:");
    println!("  Node count: {}", stats.node_count);
    println!("  Max level: {}", stats.max_level);
    println!("  Average connections: {:.2}", stats.avg_connections);
    hnsw.delete(id1);
    println!("Deleted node ID: {}", id1);
    let results_after_delete = hnsw.search_knn(&query, 2);
    assert!(results_after_delete.iter().all(|r| r.id != id1));
}
```

## Using Cosine Similarity

```rust
use hnswx::*;

fn main() {
    let config = HnswConfig {
        max_elements: 1000,
        m: 16,
        ef_construction: 200,
        ef_search: 10,
        ..Default::default()
    };
    // Use cosine similarity
    let mut hnsw = HNSW::new(config, CosineSimilarity);
    // Insert vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],  // Direction 1
        vec![0.0, 1.0, 0.0, 0.0],  // Direction 2
        vec![0.0, 0.0, 1.0, 0.0],  // Direction 3
        vec![0.5, 0.5, 0.0, 0.0],  // Between direction 1 and 2
    ];
    for vector in vectors {
        hnsw.insert(vector);
    }
    // Search for similar vectors
    let query = vec![0.6, 0.4, 0.0, 0.0];
    let results = hnsw.search_knn(&query, 2);
    println!("Cosine similarity search results:");
    for result in results {
        println!("  Node ID: {}, Distance: {:.4}", result.id, result.distance);
    }
}
```

## Custom Configuration

```rust
use hnswx::*;

fn main() {
    // Custom configuration parameters
    let config = HnswConfig {
        max_elements: 10000,      // Maximum number of elements
        m: 16,                    // Number of established connections per layer
        m_max: 32,                // Maximum number of connections at layer 0
        m_max_0: 64,              // Maximum number of connections at highest layer
        ef_construction: 200,     // Size of dynamic candidate list during construction
        ef_search: 10,            // Size of dynamic candidate list during search
        level_multiplier: 1.0 / 16.0_f64.ln(), // Level multiplier
        allow_replace_deleted: true,
    };
    let mut hnsw = HNSW::new(config, EuclideanDistance);
}
```

## Configuration Parameters

| Parameter               | Description                                        | Default  |
| ----------------------- | -------------------------------------------------- | -------- |
| `max_elements`          | Maximum capacity of the index                      | 1000     |
| `m`                     | Number of established connections per layer        | 16       |
| `m_max`                 | Maximum connections at layer 0                     | 32       |
| `m_max_0`               | Maximum connections at highest layer               | 64       |
| `ef_construction`       | Size of dynamic candidate list during construction | 200      |
| `ef_search`             | Size of dynamic candidate list during search       | 10       |
| `level_multiplier`      | Level multiplier affecting level distribution      | 1/ln(16) |
| `allow_replace_deleted` | Whether to allow replacing deleted elements        | true     |

## Distance Metrics

The library provides two distance metrics:

1. **EuclideanDistance** - Euclidean distance

   - Suitable for ordinary vector spaces
   - Distance formula: √Σ(xᵢ - yᵢ)²

2. **CosineSimilarity** - Cosine similarity (converted to distance)
   - Suitable for directional data
   - Distance formula: 1 - (a·b)/(||a||·||b||)

## Performance Characteristics

- **Time Complexity**:

  - Insertion: O(log n)
  - Search: O(log n)
  - Deletion: O(k), where k is number of neighbors

- **Space Complexity**: O(n × m), where n is number of nodes, m is average connections
