"""
知识图谱服务模块：基于 Neo4j 的医学知识图谱管理

本模块提供：
1. Neo4j 数据库连接管理
2. 知识图谱实体和关系的 CRUD 操作
3. 图谱查询接口（用于辅助诊断）

知识图谱结构：
- 实体类型：疾病(Disease)、症状(Symptom)、检查(Examination)、治疗(Treatment)、科室(Department)
- 关系类型：
  * 疾病-有症状: HAS_SYMPTOM
  * 疾病-需要检查: REQUIRES_EXAMINATION
  * 疾病-治疗方法: TREATED_BY
  * 疾病-属于科室: BELONGS_TO_DEPARTMENT
  * 症状-相关疾病: RELATED_TO_DISEASE
"""

import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from src.services.logging import log_info, log_warn, log_error


class KnowledgeGraph:
    """Neo4j 知识图谱管理类"""
    
    def __init__(self):
        """
        初始化 Neo4j 连接
        
        环境变量：
        - NEO4J_URI: Neo4j 连接地址，默认 "bolt://localhost:7687"
        - NEO4J_USER: 用户名，默认 "neo4j"
        - NEO4J_PASSWORD: 密码，默认 "password"
        """
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            log_info(f"[KG] 成功连接到 Neo4j: {self.uri}")
        except Exception as e:
            log_warn(f"[KG] Neo4j 连接失败: {e}，知识图谱功能将不可用")
            self.driver = None
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
    
    def _execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行 Cypher 查询
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
        
        Returns:
            查询结果列表
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            log_error(f"[KG] 查询执行失败: {e}")
            return []
    
    # ========== 实体创建 ==========
    
    def create_disease(self, name: str, description: str = "", aliases: List[str] = None) -> bool:
        """
        创建疾病实体
        
        Args:
            name: 疾病名称
            description: 疾病描述
            aliases: 疾病别名列表
        
        Returns:
            是否创建成功
        """
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
    
    # ========== 关系创建 ==========
    
    def link_disease_symptom(self, disease_name: str, symptom_name: str, frequency: str = "常见") -> bool:
        """
        创建疾病-症状关系
        
        Args:
            disease_name: 疾病名称
            symptom_name: 症状名称
            frequency: 症状频率（常见/偶见/罕见）
        """
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
    
    # ========== 查询接口 ==========
    
    def find_diseases_by_symptoms(self, symptoms: List[str], limit: int = 5) -> List[Dict]:
        """
        根据症状查找相关疾病
        
        Args:
            symptoms: 症状列表
            limit: 返回结果数量限制
        
        Returns:
            疾病列表，包含疾病名称和匹配的症状数量
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
        获取疾病的完整信息（症状、检查、治疗、科室）
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            疾病信息字典
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
        查找相关疾病（通过共享症状）
        
        Args:
            disease_name: 疾病名称
            limit: 返回结果数量限制
        
        Returns:
            相关疾病列表
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
        搜索实体（疾病、症状、检查、治疗）
        
        Args:
            keyword: 搜索关键词
            entity_types: 实体类型列表，如 ["Disease", "Symptom"]，None 表示搜索所有类型
        
        Returns:
            匹配的实体列表
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
    
    # ========== Graph RAG 增强查询接口 ==========
    
    def find_diseases_by_symptoms_fuzzy(
        self, 
        symptoms: List[str], 
        limit: int = 5,
        min_match: int = 1
    ) -> List[Dict]:
        """
        根据症状模糊查找相关疾病（支持部分匹配）
        
        与 find_diseases_by_symptoms 不同，此方法支持症状名称的模糊匹配，
        适合处理 LLM 提取的可能不完全标准化的症状名称。
        
        Args:
            symptoms: 症状列表
            limit: 返回结果数量限制
            min_match: 最少匹配症状数量
        
        Returns:
            疾病列表，包含疾病名称、描述、匹配的症状数量等
        """
        if not symptoms:
            return []
        
        # 构建模糊匹配的 Cypher 查询
        # 使用 CONTAINS 实现部分匹配
        query = """
        UNWIND $symptoms as symptom_keyword
        MATCH (s:Symptom)
        WHERE s.name CONTAINS symptom_keyword OR symptom_keyword CONTAINS s.name
        WITH DISTINCT s
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s)
        WITH d, collect(DISTINCT s.name) as matched_symptoms, count(DISTINCT s) as match_count
        WHERE match_count >= $min_match
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN d.name as disease_name,
               d.description as description,
               match_count,
               matched_symptoms
        """
        result = self._execute_query(query, {
            "symptoms": symptoms,
            "limit": limit,
            "min_match": min_match
        })
        return result
    
    def get_disease_full_context(self, disease_name: str) -> Optional[Dict]:
        """
        获取疾病的完整上下文信息（用于 Graph RAG 检索增强）
        
        返回比 get_disease_info 更丰富的信息，包括：
        - 疾病基本信息
        - 所有关联的症状（带频率）
        - 所有关联的检查项目
        - 所有关联的治疗方法
        - 所属科室
        - 相关疾病（通过共享症状）
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            完整的疾病上下文字典
        """
        # 获取疾病基本信息和所有关联实体
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
        
        # 过滤掉空的症状（可能由于 OPTIONAL MATCH 产生）
        disease_info["symptoms"] = [
            s for s in disease_info.get("symptoms", []) 
            if s.get("name")
        ]
        
        # 获取相关疾病
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
        查找从症状到疾病的诊断路径
        
        这是一个高级图谱查询，用于解释症状与疾病之间的关联路径，
        可用于辅助诊断推理和解释生成。
        
        Args:
            symptoms: 输入症状列表
            target_disease: 目标疾病名称（可选，不指定则返回所有可能的路径）
            max_depth: 最大路径深度
        
        Returns:
            诊断路径列表，每个路径包含：
            - symptom: 起始症状
            - disease: 目标疾病
            - path: 路径描述
            - confidence: 置信度
        """
        if not symptoms:
            return []
        
        paths = []
        
        for symptom in symptoms[:5]:  # 限制症状数量避免查询过慢
            if target_disease:
                # 查找特定疾病的路径
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
                # 查找所有可能的路径
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
                    "confidence": 1.0 / r.get("path_length", 1)  # 路径越短置信度越高
                })
        
        # 按置信度排序
        paths.sort(key=lambda x: x["confidence"], reverse=True)
        return paths
    
    def get_department_diseases(self, department_name: str, limit: int = 10) -> List[Dict]:
        """
        获取某科室的所有疾病
        
        Args:
            department_name: 科室名称
            limit: 返回结果数量限制
        
        Returns:
            疾病列表
        """
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
        """
        获取使用某种治疗方法的所有疾病
        
        Args:
            treatment_name: 治疗方法名称
            limit: 返回结果数量限制
        
        Returns:
            疾病列表
        """
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


# 全局知识图谱实例（单例模式）
_kg_instance: Optional[KnowledgeGraph] = None


def get_kg() -> KnowledgeGraph:
    """获取知识图谱实例（单例）"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance

