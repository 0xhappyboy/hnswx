<h1 align="center">
    HNSWX
</h1>
<h4 align="center">
A Rust implementation of the Hierarchical Navigable Small World (HNSW) algorithm. HNSW is an efficient approximate nearest neighbor search algorithm, particularly suitable for high-dimensional vector retrieval.
</h4>
<p align="center">
  <a href="https://github.com/0xhappyboy/hnswx/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
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
