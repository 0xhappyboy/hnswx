<h1 align="center">
    HNSWX
</h1>
<h4 align="center">
一个用Rust实现的分层可导航小世界图算法(HNSW)算法库.HNSW是一种高效的近似最近邻搜索算法,特别适用于高维向量检索.
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

# 安装

```
cargo add hnswx
```

# 测试

## 该测试将 100 万个 32 维向量插入 HNSW 索引，并执行查询和删除操作。

```
cargo test test_hnsw_1_million_vectors -- --nocapture
```

### 结果

```
=== HNSW 100万向量性能测试 ===
维度: 32D
向量数量: 1000000
配置: m=16, ef_construction=200, ef_search=100

--- 阶段1: 插入100万个向量 ---
test massive_scale_test::test_hnsw_1_million_vectors has been running for over 60 seconds
  已插入: 100000 (10.0%)
  已插入: 200000 (20.0%)
    当前统计: 节点数=200001, 最大层级=3, 平均连接数=15.34
  已插入: 300000 (30.0%)
  已插入: 400000 (40.0%)
    当前统计: 节点数=400001, 最大层级=3, 平均连接数=15.33
  已插入: 500000 (50.0%)
  已插入: 600000 (60.0%)
    当前统计: 节点数=600001, 最大层级=3, 平均连接数=15.32
  已插入: 700000 (70.0%)
  已插入: 800000 (80.0%)
    当前统计: 节点数=800001, 最大层级=3, 平均连接数=15.32
  已插入: 900000 (90.0%)
插入完成，总耗时: 950.1687562s
平均每个向量插入时间: 950.168µs

插入完成后的索引统计:
  节点总数: 1000000
  最大层级: 3
  平均连接数: 15.32
  最大连接数: 152
  入口点ID: Some(12931)
  存储大小: 32000000 浮点数

--- 阶段2: 查询性能测试 ---
执行 200 次查询测试...
  已完成查询: 40 (平均 1141.37μs)
  已完成查询: 80 (平均 1113.62μs)
  已完成查询: 120 (平均 1077.25μs)
  已完成查询: 160 (平均 1074.22μs)

查询性能统计 (200次查询平均):
  平均查询时间: 1068.71μs
  最小查询时间: 41μs
  最大查询时间: 2506μs
  中位数(P50): 1089μs
  P90: 1553μs
  P95: 1976μs
  QPS (每秒查询数): 935.71

--- 阶段3: 不同k值的查询性能 ---
  k=  1: 平均时间 1348.50μs, 结果数 1
  k=  5: 平均时间 1151.60μs, 结果数 5
  k= 10: 平均时间 1126.40μs, 结果数 10
  k= 20: 平均时间 1433.50μs, 结果数 20
  k= 50: 平均时间 1331.70μs, 结果数 50
  k=100: 平均时间 1152.40μs, 结果数 100

--- 阶段4: 删除操作测试 ---
批量删除1: 删除前100000个节点
  已删除: 20000 个节点
  已删除: 40000 个节点
  已删除: 60000 个节点
  已删除: 80000 个节点
  删除完成，耗时: 368.2621ms
  成功删除: 100000 个节点
  平均删除时间: 3.682µs

  第一次删除后统计:
    活跃节点数: 900000
    已删除节点数: 100000
    平均连接数: 15.07

--- 阶段5: 删除后查询测试 ---
  删除后查询结果: 10 个最近邻, 时间 1612μs

--- 阶段6: 第二次删除操作 ---
批量删除2: 删除ID 100000~150000 的节点
  已删除: 10000 个节点
  已删除: 20000 个节点
  已删除: 30000 个节点
  已删除: 40000 个节点
  删除完成，耗时: 213.4648ms
  成功删除: 50000 个节点
  平均删除时间: 4.269µs

  最终统计:
    活跃节点数: 850000
    已删除节点数: 150000
    总插入节点数: 1000000
    剩余节点比例: 85.0%

--- 阶段7: 最终查询测试 ---
  最终查询性能 (20次平均): 1313.65μs

--- 验证测试断言 ---

=== 测试完成 ===
总测试时间: 951.266665s
内存使用估算: ~122.1 MB

性能总结:
  插入速度: 1052.4 vectors/sec
  查询速度: 935.7 queries/sec
  删除速度: 257853.0 deletes/sec
```

# 快速开始

## 基本用法

```rust
use hnsw_rs::*;

fn main() {
// 创建默认配置
let config = HnswConfig::default();
    // 创建HNSW实例（使用欧氏距离）
    let mut hnsw = HNSW::new(config, EuclideanDistance);
    // 插入向量
    let id1 = hnsw.insert(vec![1.0, 2.0, 3.0, 4.0]);
    let id2 = hnsw.insert(vec![2.0, 3.0, 4.0, 5.0]);
    let id3 = hnsw.insert(vec![3.0, 4.0, 5.0, 6.0]);
    println!("插入的节点ID: {}, {}, {}", id1, id2, id3);
    // 搜索最近邻
    let query = vec![2.1, 3.1, 4.1, 5.1];
    let results = hnsw.search_knn(&query, 2);
    println!("搜索结果:");
    for result in results {
        println!("  节点ID: {}, 距离: {:.4}", result.id, result.distance);
    }
    // 获取统计信息
    let stats = hnsw.stats();
    println!("统计信息:");
    println!("  节点数量: {}", stats.node_count);
    println!("  最大层级: {}", stats.max_level);
    println!("  平均连接数: {:.2}", stats.avg_connections);
    // 删除节点
    hnsw.delete(id1);
    println!("删除节点ID: {}", id1);
    // 验证节点已被删除
    let results_after_delete = hnsw.search_knn(&query, 2);
    assert!(results_after_delete.iter().all(|r| r.id != id1));
}
```

## 使用余弦相似度

```rust
use hnsw_rs::*;

fn main() {
    let config = HnswConfig {
    max_elements: 1000,
    m: 16,
    ef_construction: 200,
    ef_search: 10,
    ..Default::default()
    };
    // 使用余弦相似度
    let mut hnsw = HNSW::new(config, CosineSimilarity);
    // 插入向量
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],  // 方向1
        vec![0.0, 1.0, 0.0, 0.0],  // 方向2
        vec![0.0, 0.0, 1.0, 0.0],  // 方向3
        vec![0.5, 0.5, 0.0, 0.0],  // 方向1和2之间
    ];
    for vector in vectors {
        hnsw.insert(vector);
    }
    // 搜索相似向量
    let query = vec![0.6, 0.4, 0.0, 0.0];
    let results = hnsw.search_knn(&query, 2);
    println!("余弦相似度搜索结果:");
    for result in results {
        println!("  节点ID: {}, 距离: {:.4}", result.id, result.distance);
    }
}
```

## 自定义配置

```rust
use hnsw_rs::*;

fn main() {
    // 自定义配置参数
    let config = HnswConfig {
    max_elements: 10000, // 最大元素数量
    m: 16, // 每层连接数
    m_max: 32, // 第 0 层最大连接数
    m_max_0: 64, // 最高层最大连接数
    ef_construction: 200, // 构建时候选列表大小
    ef_search: 10, // 搜索时候选列表大小
    level_multiplier: 1.0 / 16.0_f64.ln(), // 层级分布参数
    allow_replace_deleted: true,
    };
    let mut hnsw = HNSW::new(config, EuclideanDistance);
}
```

## 配置参数说明

| 参数                    | 描述                   | 默认值   |
| ----------------------- | ---------------------- | -------- |
| `max_elements`          | 索引最大容量           | 1000     |
| `m`                     | 每层建立的连接数       | 16       |
| `m_max`                 | 第 0 层最大连接数      | 32       |
| `m_max_0`               | 最高层最大连接数       | 64       |
| `ef_construction`       | 构建时动态候选列表大小 | 200      |
| `ef_search`             | 搜索时动态候选列表大小 | 10       |
| `level_multiplier`      | 层级乘数，影响层级分布 | 1/ln(16) |
| `allow_replace_deleted` | 是否允许替换已删除节点 | true     |

## 距离度量

库提供两种距离度量：

1. **EuclideanDistance** - 欧氏距离

   - 适用于普通向量空间
   - 距离公式：√Σ(xᵢ - yᵢ)²

2. **CosineSimilarity** - 余弦相似度（转换为距离）
   - 适用于方向性数据
   - 距离公式：1 - (a·b)/(||a||·||b||)

## 性能特性

- **时间复杂度**：

  - 插入：O(log n)
  - 搜索：O(log n)
  - 删除：O(k)，其中 k 为邻居数量

- **空间复杂度**：O(n × m)，其中 n 为节点数，m 为平均连接数
