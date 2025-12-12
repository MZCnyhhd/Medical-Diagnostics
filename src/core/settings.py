"""
统一配置管理模块
================
集中管理所有配置项，提供类型验证和默认值
"""

import os
from typing import Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """应用配置类"""
    
    # ========== LLM 配置 ==========
    llm_provider: Literal["qwen", "openai", "gemini", "local"] = "qwen"
    dashscope_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    local_model_path: Optional[str] = None
    
    # 模型参数
    qwen_model: str = "qwen-turbo"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-pro-latest"
    llm_temperature: float = 0.0
    
    # ========== 知识库配置 ==========
    # RAG 配置
    enable_rag: bool = True
    use_local_rag: bool = False
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pinecone 配置（云端 RAG）
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "medical-knowledge"
    pinecone_embedding_model: str = "llama-text-embed-v2"
    
    # Neo4j 配置（知识图谱）
    enable_neo4j: bool = False  # 默认关闭，因为未集成
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # ========== 性能配置 ==========
    max_concurrent_agents: int = 5
    agent_timeout: int = 30
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存时间（秒）
    
    # ========== 路径配置 ==========
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    knowledge_base_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        self.data_dir = self.project_root / "data"
        self.knowledge_base_dir = self.data_dir / "knowledge_base"
        self.config_dir = self.project_root / "config"
        
        # 从环境变量加载配置
        self._load_from_env()
        
        # 验证配置
        self._validate()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # LLM Provider
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        
        # API Keys
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", self.dashscope_api_key)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", self.pinecone_api_key)
        
        # Models
        self.qwen_model = os.getenv("QWEN_MODEL", self.qwen_model)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model)
        
        # Temperature
        temp = os.getenv("LLM_TEMPERATURE")
        if temp:
            try:
                self.llm_temperature = float(temp)
            except ValueError:
                pass
        
        # RAG
        self.use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"
        self.enable_rag = os.getenv("ENABLE_RAG", "true").lower() != "false"
        
        # Neo4j
        self.enable_neo4j = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        
        # Performance
        if max_agents := os.getenv("MAX_CONCURRENT_AGENTS"):
            try:
                self.max_concurrent_agents = int(max_agents)
            except ValueError:
                pass
    
    def _validate(self):
        """验证配置的合法性"""
        # 验证 LLM 配置
        if self.llm_provider == "qwen" and not self.dashscope_api_key:
            print("⚠️ 警告: 使用 Qwen 但未配置 DASHSCOPE_API_KEY")
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("使用 OpenAI 必须配置 OPENAI_API_KEY")
        
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("使用 Gemini 必须配置 GOOGLE_API_KEY")
        
        # 验证 RAG 配置
        if self.enable_rag and not self.use_local_rag and not self.pinecone_api_key:
            print("⚠️ 警告: 启用云端 RAG 但未配置 PINECONE_API_KEY，RAG 功能将不可用")
        
        # 验证路径
        if not self.knowledge_base_dir.exists():
            print(f"⚠️ 警告: 知识库目录不存在: {self.knowledge_base_dir}")
    
    def get_active_llm_config(self) -> dict:
        """获取当前激活的 LLM 配置"""
        if self.llm_provider == "qwen":
            return {
                "provider": "qwen",
                "api_key": self.dashscope_api_key,
                "model": self.qwen_model,
                "temperature": self.llm_temperature
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.llm_temperature
            }
        elif self.llm_provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": self.google_api_key,
                "model": self.gemini_model,
                "temperature": self.llm_temperature
            }
        else:
            return {
                "provider": "local",
                "model_path": self.local_model_path,
                "temperature": self.llm_temperature
            }
    
    def should_use_neo4j(self) -> bool:
        """判断是否应该使用 Neo4j"""
        return self.enable_neo4j and self.neo4j_uri
    
    def should_use_rag(self) -> bool:
        """判断是否应该使用 RAG"""
        if not self.enable_rag:
            return False
        
        if self.use_local_rag:
            return True  # 本地 RAG 总是可用
        
        return bool(self.pinecone_api_key)  # 云端 RAG 需要 API Key


# 全局单例
_settings = None

def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
