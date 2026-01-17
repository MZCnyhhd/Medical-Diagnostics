"""
模块名称: Knowledge Graph Service (知识图谱服务)
功能描述:

    封装 Neo4j 数据库操作，提供对医疗知识图谱的 CRUD 接口。
    支持节点（疾病、症状、检查）的创建、关系的建立以及基于 Cypher 语句的复杂查询。

设计理念:

    1.  **Driver 模式**: 使用 Neo4j Driver 管理连接池，保证高并发下的性能。
    2.  **抽象层**: 屏蔽 Cypher 语法细节，提供 `add_entity`, `add_relation`, `query_subgraph` 等语义化接口。
    3.  **资源管理**: 实现 `close` 方法，确保应用退出时释放数据库连接。

线程安全性:

    - Neo4j Driver 是线程安全的，可以在多线程环境中共享。

依赖关系:

    - `neo4j`: 官方 Python 驱动。
    - `src.core.settings`: 获取 Neo4j 连接配置 (URI, User, Password)。
"""

import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
# [内部模块 | Internal Modules] =========================================================================================
from src.services.logging import log_info, log_warn, log_error

# [定义类] ##############################################################################################################
# [知识图谱管理类] ========================================================================================================
class KnowledgeGraph:
    """
    Neo4j 知识图谱管理类。
    提供图谱连接、CRUD 操作和复杂查询功能。
    """
    
    # [初始化] ============================================================================================================
    def __init__(self):
        """
        初始化 Neo4j 连接。
        从环境变量读取配置，并建立驱动连接。
        """
        # [step1] 读取配置
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        # [step2] 建立连接并测试
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            log_info(f"[KG] 成功连接到 Neo4j: {self.uri}")
        except Exception as e:
            log_warn(f"[KG] Neo4j 连接失败: {e}，知识图谱功能将不可用")
            self.driver = None
    
    # [资源释放] ==========================================================================================================
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
    
    # [内部-执行查询] =====================================================================================================
    def _execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行 Cypher 查询并返回字典列表。
        :param query: Cypher 语句
        :param parameters: 参数字典
        :return: 结果列表
        """
        # [step1] 检查驱动状态
        if not self.driver:
            return []
        
        # [step2] 执行查询
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            log_error(f"[KG] 查询执行失败: {e}")
            return []

    
    # [实体创建] ==========================================================================================================
    
    def create_disease(self, name: str, description: str = "", aliases: List[str] = None) -> bool:
        """创建疾病实体"""
        query = """
        MERGE (d:Disease {name: $name})
        SET d.description = $description,
            d.aliases = $aliases,
            d.updated_at = datetime()
        RETURN d
        """
        result = self._execute_query(query, {
            "name": name,
            "description": description,
            "aliases": aliases or []
        })
        return len(result) > 0
    
    def create_symptom(self, name: str, description: str = "") -> bool:
        """创建症状实体"""
        query = """
        MERGE (s:Symptom {name: $name})
        SET s.description = $description,
            s.updated_at = datetime()
        RETURN s
        """
        result = self._execute_query(query, {"name": name, "description": description})
        return len(result) > 0
    
    def create_examination(self, name: str, description: str = "") -> bool:
        """创建检查项目实体"""
        query = """
        MERGE (e:Examination {name: $name})
        SET e.description = $description,
            e.updated_at = datetime()
        RETURN e
        """
        result = self._execute_query(query, {"name": name, "description": description})
        return len(result) > 0
    
    def create_treatment(self, name: str, description: str = "") -> bool:
        """创建治疗方法实体"""
        query = """
        MERGE (t:Treatment {name: $name})
        SET t.description = $description,
            t.updated_at = datetime()
        RETURN t
        """
        result = self._execute_query(query, {"name": name, "description": description})
        return len(result) > 0
    
    def create_department(self, name: str) -> bool:
        """创建科室实体"""
        query = """
        MERGE (dept:Department {name: $name})
        SET dept.updated_at = datetime()
        RETURN dept
        """
        result = self._execute_query(query, {"name": name})
        return len(result) > 0
    
    # [关系创建] ==========================================================================================================
    
    def link_disease_symptom(self, disease_name: str, symptom_name: str, frequency: str = "常见") -> bool:
        """创建疾病-症状关系"""
        query = """
        MATCH (d:Disease {name: $disease_name})
        MATCH (s:Symptom {name: $symptom_name})
        MERGE (d)-[r:HAS_SYMPTOM {frequency: $frequency}]->(s)
        RETURN r
        """
        result = self._execute_query(query, {
            "disease_name": disease_name,
            "symptom_name": symptom_name,
            "frequency": frequency
        })
        return len(result) > 0
    
    def link_disease_examination(self, disease_name: str, exam_name: str) -> bool:
        """创建疾病-检查关系"""
        query = """
        MATCH (d:Disease {name: $disease_name})
        MATCH (e:Examination {name: $exam_name})
        MERGE (d)-[r:REQUIRES_EXAMINATION]->(e)
        RETURN r
        """
        result = self._execute_query(query, {
            "disease_name": disease_name,
            "exam_name": exam_name
        })
        return len(result) > 0
    
    def link_disease_treatment(self, disease_name: str, treatment_name: str) -> bool:
        """创建疾病-治疗关系"""
        query = """
        MATCH (d:Disease {name: $disease_name})
        MATCH (t:Treatment {name: $treatment_name})
        MERGE (d)-[r:TREATED_BY]->(t)
        RETURN r
        """
        result = self._execute_query(query, {
            "disease_name": disease_name,
            "treatment_name": treatment_name
        })
        return len(result) > 0
    
    def link_disease_department(self, disease_name: str, department_name: str) -> bool:
        """创建疾病-科室关系"""
        query = """
        MATCH (d:Disease {name: $disease_name})
        MATCH (dept:Department {name: $department_name})
        MERGE (d)-[r:BELONGS_TO_DEPARTMENT]->(dept)
        RETURN r
        """
        result = self._execute_query(query, {
            "disease_name": disease_name,
            "department_name": department_name
        })
        return len(result) > 0
    
    # [查询接口] ==========================================================================================================
    
    def find_diseases_by_symptoms(self, symptoms: List[str], limit: int = 5) -> List[Dict]:
        """
        根据症状查找相关疾病。
        :param symptoms: 症状列表
        :param limit: 限制数量
        :return: 疾病列表
        """
        query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, count(DISTINCT s) as match_count
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN d.name as disease_name, 
               d.description as description,
               match_count,
               [(d)-[:HAS_SYMPTOM]->(s) WHERE s.name IN $symptoms | s.name] as matched_symptoms
        """
        result = self._execute_query(query, {"symptoms": symptoms, "limit": limit})
        return result
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        获取疾病的完整信息（症状、检查、治疗、科室）。
        :param disease_name: 疾病名称
        :return: 疾病信息字典
        """
        query = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[:REQUIRES_EXAMINATION]->(e:Examination)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
        OPTIONAL MATCH (d)-[:BELONGS_TO_DEPARTMENT]->(dept:Department)
        RETURN d.name as name,
               d.description as description,
               collect(DISTINCT s.name) as symptoms,
               collect(DISTINCT e.name) as examinations,
               collect(DISTINCT t.name) as treatments,
               collect(DISTINCT dept.name) as departments
        """
        result = self._execute_query(query, {"disease_name": disease_name})
        return result[0] if result else None
    
    def get_related_diseases(self, disease_name: str, limit: int = 5) -> List[Dict]:
        """
        查找相关疾病（通过共享症状）。
        :param disease_name: 疾病名称
        :param limit: 限制数量
        :return: 相关疾病列表
        """
        query = """
        MATCH (d1:Disease {name: $disease_name})-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
        WHERE d1 <> d2
        WITH d2, count(DISTINCT s) as common_symptoms
        ORDER BY common_symptoms DESC
        LIMIT $limit
        RETURN d2.name as disease_name, common_symptoms
        """
        result = self._execute_query(query, {"disease_name": disease_name, "limit": limit})
        return result
    
    def search_entities(self, keyword: str, entity_types: List[str] = None) -> List[Dict]:
        """
        搜索实体（疾病、症状、检查、治疗）。
        :param keyword: 关键词
        :param entity_types: 实体类型过滤
        :return: 实体列表
        """
        if entity_types is None:
            entity_types = ["Disease", "Symptom", "Examination", "Treatment", "Department"]
        
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN $entity_types)
          AND (n.name CONTAINS $keyword OR any(alias IN n.aliases WHERE alias CONTAINS $keyword))
        RETURN labels(n)[0] as type, n.name as name, n.description as description
        LIMIT 20
        """
        result = self._execute_query(query, {
            "keyword": keyword,
            "entity_types": entity_types
        })
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """获取知识图谱统计信息"""
        query = """
        MATCH (d:Disease) WITH count(d) as disease_count
        MATCH (s:Symptom) WITH disease_count, count(s) as symptom_count
        MATCH (e:Examination) WITH disease_count, symptom_count, count(e) as exam_count
        MATCH (t:Treatment) WITH disease_count, symptom_count, exam_count, count(t) as treatment_count
        MATCH (dept:Department) WITH disease_count, symptom_count, exam_count, treatment_count, count(dept) as dept_count
        MATCH ()-[r]->() WITH disease_count, symptom_count, exam_count, treatment_count, dept_count, count(r) as relation_count
        RETURN disease_count, symptom_count, exam_count, treatment_count, dept_count, relation_count
        """
        result = self._execute_query(query)
        return result[0] if result else {}
    
    # [Graph RAG 增强查询接口] ==============================================================================================
    
    # TODO: 该方法目前未使用，保留以备未来扩展模糊匹配功能
    # def find_diseases_by_symptoms_fuzzy(...) 
    
    def get_disease_full_context(self, disease_name: str) -> Optional[Dict]:
        """
        获取疾病的完整上下文信息（用于 Graph RAG 检索增强）。
        :param disease_name: 疾病名称
        :return: 完整上下文
        """
        # [step1] 查询基本信息和所有关联
        query = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[r1:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[:REQUIRES_EXAMINATION]->(e:Examination)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
        OPTIONAL MATCH (d)-[:BELONGS_TO_DEPARTMENT]->(dept:Department)
        WITH d, 
             collect(DISTINCT {name: s.name, frequency: r1.frequency}) as symptoms,
             collect(DISTINCT e.name) as examinations,
             collect(DISTINCT t.name) as treatments,
             collect(DISTINCT dept.name) as departments
        RETURN d.name as name,
               d.description as description,
               d.aliases as aliases,
               symptoms,
               examinations,
               treatments,
               departments
        """
        result = self._execute_query(query, {"disease_name": disease_name})
        
        if not result:
            return None
        
        disease_info = result[0]
        
        # [step2] 过滤无效数据
        disease_info["symptoms"] = [
            s for s in disease_info.get("symptoms", []) 
            if s.get("name")
        ]
        
        # [step3] 获取相关疾病
        related = self.get_related_diseases(disease_name, limit=3)
        disease_info["related_diseases"] = [r.get("disease_name") for r in related if r.get("disease_name")]
        
        return disease_info
    
    def find_diagnostic_path(
        self, 
        symptoms: List[str], 
        target_disease: str = None,
        max_depth: int = 3
    ) -> List[Dict]:
        """
        查找从症状到疾病的诊断路径。
        :param symptoms: 症状列表
        :param target_disease: 目标疾病（可选）
        :param max_depth: 最大深度
        :return: 路径列表
        """
        if not symptoms:
            return []
        
        paths = []
        
        for symptom in symptoms[:5]:
            if target_disease:
                # [step1] 查找特定疾病路径
                query = """
                MATCH path = (s:Symptom)-[:HAS_SYMPTOM*..{max_depth}]-(d:Disease {{name: $disease_name}})
                WHERE s.name CONTAINS $symptom OR $symptom CONTAINS s.name
                RETURN s.name as symptom,
                       d.name as disease,
                       length(path) as path_length,
                       [n in nodes(path) | labels(n)[0] + ': ' + n.name] as path_nodes
                LIMIT 3
                """.format(max_depth=max_depth)
                result = self._execute_query(query, {
                    "symptom": symptom,
                    "disease_name": target_disease
                })
            else:
                # [step2] 查找所有可能路径
                query = """
                MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
                WHERE s.name CONTAINS $symptom OR $symptom CONTAINS s.name
                RETURN s.name as symptom,
                       d.name as disease,
                       1 as path_length,
                       [s.name, d.name] as path_nodes
                LIMIT 5
                """
                result = self._execute_query(query, {"symptom": symptom})
            
            for r in result:
                paths.append({
                    "symptom": r.get("symptom"),
                    "disease": r.get("disease"),
                    "path_length": r.get("path_length", 1),
                    "path_nodes": r.get("path_nodes", []),
                    "confidence": 1.0 / r.get("path_length", 1)
                })
        
        # [step3] 按置信度排序
        paths.sort(key=lambda x: x["confidence"], reverse=True)
        return paths
    
    def get_department_diseases(self, department_name: str, limit: int = 10) -> List[Dict]:
        """获取某科室的所有疾病"""
        query = """
        MATCH (d:Disease)-[:BELONGS_TO_DEPARTMENT]->(dept:Department)
        WHERE dept.name CONTAINS $department_name OR $department_name CONTAINS dept.name
        RETURN d.name as disease_name,
               d.description as description,
               dept.name as department
        LIMIT $limit
        """
        result = self._execute_query(query, {
            "department_name": department_name,
            "limit": limit
        })
        return result
    
    def get_treatment_diseases(self, treatment_name: str, limit: int = 10) -> List[Dict]:
        """获取使用某种治疗方法的所有疾病"""
        query = """
        MATCH (d:Disease)-[:TREATED_BY]->(t:Treatment)
        WHERE t.name CONTAINS $treatment_name OR $treatment_name CONTAINS t.name
        RETURN d.name as disease_name,
               d.description as description,
               t.name as treatment
        LIMIT $limit
        """
        result = self._execute_query(query, {
            "treatment_name": treatment_name,
            "limit": limit
        })
        return result


# [定义函数] ##############################################################################################################
# [全局实例-获取知识图谱] ===================================================================================================
# 全局知识图谱实例（单例模式）
_kg_instance: Optional[KnowledgeGraph] = None


def get_kg() -> KnowledgeGraph:
    """
    获取知识图谱单例实例。
    :return: KnowledgeGraph 对象
    """
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance

