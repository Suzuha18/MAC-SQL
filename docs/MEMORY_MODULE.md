# Memory Module (RAG) 使用文档

## 概述

Memory 模块是一个基于 RAG (Retrieval-Augmented Generation) 的记忆系统，用于存储和检索 SQL 生成的成功案例和失败案例。

- **成功案例 (Success Cases)**: 用于 `Decomposer` 代理，帮助它学习如何逐步完成 SQL 推理
- **失败案例 (Failure Cases)**: 用于 `Refiner` 代理，帮助它学习如何修正和纠正 SQL 错误

## 特性

### 1. 多种检索策略

模块支持三种检索策略，可以灵活切换：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| `keyword` | 基于关键词的检索 (TF-IDF风格) | 快速、无需额外依赖 | 语义理解能力有限 |
| `embedding` | 基于嵌入向量的语义检索 | 语义理解能力强 | 需要额外依赖、较慢 |
| `hybrid` | 混合检索 (关键词 + 嵌入) | 综合两者优点 | 需要额外依赖 |

### 2. 自动案例存储

- 成功执行的 SQL 会自动存储为成功案例
- SQL 执行错误会自动存储为失败案例（包含错误信息和修正后的 SQL）

### 3. 持久化存储

所有案例以 JSON 格式持久化存储，重启后可自动加载。

## 安装依赖

```bash
# 基础依赖 (已包含在 requirements.txt)
pip install -r requirements.txt

# 如果使用 embedding 或 hybrid 策略，需要额外安装：
pip install sentence-transformers numpy
```

## 配置

### 配置文件

可以通过 `memory_config.json` 文件配置记忆模块：

```json
{
  "enabled": true,
  "storage_path": "./memory_store",
  "retrieval_strategy": "keyword",
  "embedding_model": "all-MiniLM-L6-v2",
  "max_cases": 1000,
  "decomposer_top_k": 2,
  "refiner_top_k": 2,
  "min_similarity_threshold": 0.1,
  "auto_save_success": true,
  "auto_save_failure": true
}
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用记忆模块 |
| `storage_path` | string | `"./memory_store"` | 存储路径 |
| `retrieval_strategy` | string | `"keyword"` | 检索策略: `keyword`, `embedding`, `hybrid` |
| `embedding_model` | string | `"all-MiniLM-L6-v2"` | Embedding 模型名称 |
| `max_cases` | int | `1000` | 每种类型最大案例数 |
| `decomposer_top_k` | int | `2` | Decomposer 检索的案例数 |
| `refiner_top_k` | int | `2` | Refiner 检索的案例数 |
| `min_similarity_threshold` | float | `0.1` | 最小相似度阈值 |
| `auto_save_success` | bool | `true` | 自动保存成功案例 |
| `auto_save_failure` | bool | `true` | 自动保存失败案例 |

## 使用方法

### 1. 基础使用 (自动集成)

记忆模块已经自动集成到 `Decomposer` 和 `Refiner` 中，只需正常使用即可：

```python
from core.agents import Decomposer, Refiner
from core.memory import MemoryConfig, set_memory_config

# 方式1: 使用默认配置
decomposer = Decomposer(dataset_name='bird')
refiner = Refiner(data_path='./data/bird/database', dataset_name='bird')

# 方式2: 使用自定义配置
config = MemoryConfig(
    enabled=True,
    retrieval_strategy='hybrid',
    decomposer_top_k=3,
    refiner_top_k=3
)
set_memory_config(config)

decomposer = Decomposer(dataset_name='bird', memory_config=config)
refiner = Refiner(data_path='./data/bird/database', dataset_name='bird', memory_config=config)
```

### 2. 从配置文件加载

```python
from core.memory import MemoryConfig, set_memory_config

config = MemoryConfig.from_json_file('memory_config.json')
set_memory_config(config)
```

### 3. 运行时切换策略

```python
from core.memory import get_memory_store

memory_store = get_memory_store()

# 切换到 embedding 策略
memory_store.set_retrieval_strategy('embedding')

# 切换到 hybrid 策略
memory_store.set_retrieval_strategy('hybrid', embedding_model='all-MiniLM-L6-v2')
```

### 4. 手动添加案例

```python
from core.memory import get_memory_store

memory_store = get_memory_store()

# 添加成功案例
memory_store.add_success_case(
    query="What is the name of the customer with the highest total order amount?",
    evidence="Total order amount = SUM(price * quantity)",
    db_schema="# Table: customers\n[...]\n# Table: orders\n[...]",
    sql="SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY SUM(o.price * o.quantity) DESC LIMIT 1",
    reasoning_steps=[
        "Step 1: Identify relevant tables (customers, orders)",
        "Step 2: Calculate total order amount per customer",
        "Step 3: Order by total amount descending and limit to 1"
    ]
)

# 添加失败案例
memory_store.add_failure_case(
    query="List all products with price > 100",
    evidence="",
    db_schema="# Table: products\n[...]",
    sql="SELECT * FROM product WHERE price > 100",  # 错误: 表名应该是 products
    error_info="no such table: product",
    correction="SELECT * FROM products WHERE price > 100",
    correction_explanation="Table name should be 'products' not 'product'"
)
```

### 5. 查看统计信息

```python
from core.memory import get_memory_store

memory_store = get_memory_store()
stats = memory_store.get_statistics()
print(stats)
# Output:
# {
#   "success_cases_count": 150,
#   "failure_cases_count": 45,
#   "total_cases": 195,
#   "retrieval_strategy": "keyword",
#   "max_cases": 1000,
#   "storage_path": "./memory_store"
# }
```

### 6. 清除记忆

```python
from core.memory import get_memory_store

memory_store = get_memory_store()

# 清除所有
memory_store.clear_all()

# 只清除成功案例
memory_store.clear_success_cases()

# 只清除失败案例
memory_store.clear_failure_cases()
```

### 7. 禁用/启用记忆

```python
# 在代理级别禁用
decomposer.set_memory_enabled(False)
refiner.set_memory_enabled(False)

# 或者通过配置禁用
config = MemoryConfig(enabled=False)
set_memory_config(config)
```

## 文件结构

```
memory_store/
├── success_cases.json    # 成功案例存储
└── failure_cases.json    # 失败案例存储
```

## 案例数据格式

### 成功案例 (Success Case)

```json
{
  "case_id": "success_0_20231229120000",
  "case_type": "success",
  "query": "What is the gender of the youngest client?",
  "evidence": "Later birthdate refers to younger age",
  "db_schema": "# Table: client\n[...]",
  "sql": "SELECT gender FROM client ORDER BY birth_date DESC LIMIT 1",
  "reasoning_steps": [
    "Sub question 1: Find the youngest client...",
    "Sub question 2: Get the gender of that client..."
  ],
  "error_info": null,
  "correction": null,
  "correction_explanation": null,
  "metadata": {"dataset": "bird"},
  "created_at": "2023-12-29T12:00:00"
}
```

### 失败案例 (Failure Case)

```json
{
  "case_id": "failure_0_20231229120000",
  "case_type": "failure",
  "query": "List all products with price > 100",
  "evidence": "",
  "db_schema": "# Table: products\n[...]",
  "sql": "SELECT * FROM product WHERE price > 100",
  "reasoning_steps": [],
  "error_info": "no such table: product",
  "correction": "SELECT * FROM products WHERE price > 100",
  "correction_explanation": "Table name should be 'products' not 'product'",
  "metadata": {"dataset": "bird"},
  "created_at": "2023-12-29T12:00:00"
}
```

## 策略选择建议

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 快速原型开发 | `keyword` | 无需额外依赖，速度快 |
| 生产环境 | `hybrid` | 综合性能最佳 |
| 资源受限环境 | `keyword` | 内存和计算需求最低 |
| 高精度需求 | `embedding` | 语义理解最准确 |

## 注意事项

1. **Embedding 策略**需要下载预训练模型，首次使用时可能需要一些时间
2. 记忆存储会自动去重，相同的案例不会被重复存储
3. 当案例数超过 `max_cases` 时，会自动删除最旧的案例
4. 建议定期备份 `memory_store` 目录

## 扩展开发

### 自定义检索策略

可以继承 `RetrievalStrategy` 基类实现自定义策略：

```python
from core.memory import RetrievalStrategy, MemoryCase
from typing import List, Tuple

class MyCustomStrategy(RetrievalStrategy):
    def retrieve(self, query: str, cases: List[MemoryCase], top_k: int = 3) -> List[Tuple[MemoryCase, float]]:
        # 实现自定义检索逻辑
        ...
    
    def get_strategy_name(self) -> str:
        return "custom"
```

## 问题排查

### 1. 记忆不生效
- 检查 `enabled` 是否为 `true`
- 检查 `min_similarity_threshold` 是否设置过高

### 2. 检索速度慢
- 考虑切换到 `keyword` 策略
- 减少 `max_cases` 数量

### 3. 找不到相关案例
- 降低 `min_similarity_threshold`
- 增加 `decomposer_top_k` 或 `refiner_top_k`
- 尝试使用 `hybrid` 策略
