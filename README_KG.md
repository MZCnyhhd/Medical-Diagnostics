# 知识图谱功能使用指南

本项目已集成 Neo4j 知识图谱功能，用于存储和查询结构化的医学知识。

## 📋 功能特性

- **实体类型**：疾病、症状、检查、治疗、科室
- **关系类型**：疾病-症状、疾病-检查、疾病-治疗、疾病-科室
- **智能查询**：根据症状查找相关疾病、获取疾病完整信息、查找相关疾病

## 🚀 快速开始

### 1. 启动 Neo4j 数据库

使用 Docker Compose 一键启动（包含 Neo4j 服务）：

```bash
docker-compose up -d
```

Neo4j 服务将在以下端口启动：
- **HTTP 端口**：7474（Neo4j Browser，访问 http://localhost:7474）
- **Bolt 端口**：7687（应用程序连接）

默认用户名/密码：`neo4j` / `password`

### 2. 构建知识图谱

运行构建脚本，从医学知识库中抽取知识并构建图谱：

```bash
python src/scripts/build_kg.py
```

脚本会：
- 读取 `data/knowledge_base/` 下的所有 `.md` 文件
- 使用 LLM 抽取结构化知识（疾病、症状、检查、治疗、科室）
- 将知识写入 Neo4j 数据库

### 3. 查看知识图谱

访问 Neo4j Browser：http://localhost:7474

使用以下 Cypher 查询查看图谱：

```cypher
// 查看所有疾病
MATCH (d:Disease) RETURN d LIMIT 10

// 查看疾病及其症状关系
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN d.name, s.name LIMIT 20

// 查看图谱统计
MATCH (d:Disease) WITH count(d) as disease_count
MATCH (s:Symptom) WITH disease_count, count(s) as symptom_count
MATCH ()-[r]->() WITH disease_count, symptom_count, count(r) as relation_count
RETURN disease_count, symptom_count, relation_count
```

## 💻 代码使用示例

### 基本查询

```python
from src.services.kg import get_kg

kg = get_kg()

# 根据症状查找相关疾病
diseases = kg.find_diseases_by_symptoms(["多饮", "多尿"], limit=5)
for disease in diseases:
    print(f"疾病: {disease['disease_name']}, 匹配症状数: {disease['match_count']}")

# 获取疾病完整信息
disease_info = kg.get_disease_info("糖尿病")
print(f"症状: {disease_info['symptoms']}")
print(f"检查: {disease_info['examinations']}")
print(f"治疗: {disease_info['treatments']}")
print(f"科室: {disease_info['departments']}")

# 查找相关疾病
related = kg.get_related_diseases("糖尿病", limit=5)
for rel in related:
    print(f"相关疾病: {rel['disease_name']}, 共同症状数: {rel['common_symptoms']}")
```

### 结合 RAG 使用

知识图谱已集成到 RAG 服务中，可以在诊断时自动查询：

```python
from src.services.rag import retrieve_knowledge_with_kg

# 混合检索：向量检索 + 知识图谱
knowledge = retrieve_knowledge_with_kg("患者主诉多饮多尿", k=3, use_kg=True)
print(knowledge)
```

## 📊 知识图谱结构

### 实体类型

- **Disease（疾病）**：疾病名称、描述、别名
- **Symptom（症状）**：症状名称、描述
- **Examination（检查）**：检查项目名称、描述
- **Treatment（治疗）**：治疗方法名称、描述
- **Department（科室）**：科室名称

### 关系类型

- **HAS_SYMPTOM**：疾病 → 症状（属性：frequency 频率）
- **REQUIRES_EXAMINATION**：疾病 → 检查
- **TREATED_BY**：疾病 → 治疗
- **BELONGS_TO_DEPARTMENT**：疾病 → 科室

## 🔧 配置

在 `config/apikey.env` 中配置 Neo4j 连接信息：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

如果使用 Docker Compose，容器内会自动配置为：

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## 📝 注意事项

1. **首次运行**：需要先运行 `build_kg.py` 构建知识图谱
2. **LLM 依赖**：构建脚本需要使用 LLM 抽取知识，确保已配置 API Key
3. **性能优化**：大量数据时建议分批处理，避免一次性处理过多文件
4. **数据更新**：更新知识库后，需要重新运行构建脚本

## 🎯 未来扩展

- [ ] 支持更多实体类型（药物、基因、病理等）
- [ ] 支持更复杂的关系（因果关系、时序关系等）
- [ ] 图谱可视化界面
- [ ] 知识图谱推理（路径查询、推荐等）
- [ ] 与诊断流程深度集成（图谱辅助分诊、诊断验证等）


