"""
模块名称: Global Settings (全局配置)
功能描述:

    管理全项目的配置参数，包括 LLM 模型参数、数据库路径、API 密钥、系统阈值等。
    基于 `pydantic-settings` 实现，支持从环境变量 (.env) 自动加载和类型验证。

设计理念:

    1.  **集中管理**: 所有配置项汇聚于此，避免硬编码散落在代码各处。
    2.  **类型安全**: 利用 Pydantic 进行强类型检查，尽早发现配置错误。
    3.  **环境隔离**: 支持通过 `.env` 文件切换开发/生产环境配置。
    4.  **默认值机制**: 提供合理的默认值，降低部署门槛。

线程安全性:

    - 配置对象通常在启动时初始化，运行时只读，因此是线程安全的。

依赖关系:

    - 第三方库: `pydantic`, `pydantic-settings`, `dotenv`.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional

# [创建全局变量] =========================================================================================================
# API Key 配置文件路径（从原 config.py 移入）
APIKEY_ENV_PATH = os.getenv("APIKEY_ENV_PATH", "config/apikey.env")


# [定义类] ##############################################################################################################
# [应用配置类] ===========================================================================================================
@dataclass
class Settings:
    """应用配置类"""
    
    # ========== LLM 配置 ==========
    llm_provider: Literal["qwen", "openai", "gemini", "baichuan", "ollama", "local"] = "qwen"
    dashscope_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    baichuan_api_key: Optional[str] = None
    local_model_path: Optional[str] = None
    
    # Ollama 配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma:latest"

    # 模型参数
    qwen_model: str = "qwen-turbo"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-1.5-pro-latest"
    baichuan_model: str = "Baichuan-M2"
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
    
    # 动态属性
    required_api_keys: list[str] = field(default_factory=lambda: ["DASHSCOPE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"])
    examples_dir: Path = field(init=False)

    def __post_init__(self):
        """初始化后处理"""
        # [step1] 初始化目录路径
        self.data_dir = self.project_root / "data"
        self.knowledge_base_dir = self.data_dir / "knowledge_base"
        self.config_dir = self.project_root / "config"
        self.examples_dir = self.data_dir / "medical_reports" / "Examples"
        
        # [step2] 从环境变量加载配置
        self._load_from_env()
        
        # [step3] 验证配置
        self._validate()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # [step1] 加载 LLM 提供商
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        
        # [step2] 加载 API Keys
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", self.dashscope_api_key)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.google_api_key = os.getenv("GOOGLE_API_KEY", self.google_api_key)
        self.baichuan_api_key = os.getenv("BAICHUAN_API_KEY", self.baichuan_api_key)
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", self.pinecone_api_key)
        
        # [step3] 加载模型名称和路径
        self.qwen_model = os.getenv("QWEN_MODEL", self.qwen_model)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model)
        self.baichuan_model = os.getenv("BAICHUAN_MODEL", self.baichuan_model)
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.local_model_path = os.getenv("LOCAL_MODEL_PATH", self.local_model_path)
        
        # [step4] 加载 Temperature
        temp = os.getenv("LLM_TEMPERATURE")
        if temp:
            try:
                self.llm_temperature = float(temp)
            except ValueError:
                pass
        
        # [step5] 加载 RAG 配置
        self.use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"
        self.enable_rag = os.getenv("ENABLE_RAG", "true").lower() != "false"
        
        # [step6] 加载 Neo4j 配置
        self.enable_neo4j = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        
        # [step7] 加载性能配置
        if max_agents := os.getenv("MAX_CONCURRENT_AGENTS"):
            try:
                self.max_concurrent_agents = int(max_agents)
            except ValueError:
                pass
    
    def _validate(self):
        """验证配置的合法性"""
        # [step1] 验证 LLM 配置
        if self.llm_provider == "qwen" and not self.dashscope_api_key:
            print("⚠️ 警告: 使用 Qwen 但未配置 DASHSCOPE_API_KEY")
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("使用 OpenAI 必须配置 OPENAI_API_KEY")
        
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("使用 Gemini 必须配置 GOOGLE_API_KEY")

        if self.llm_provider == "baichuan" and not self.baichuan_api_key:
            raise ValueError("使用 Baichuan 必须配置 BAICHUAN_API_KEY")
        
        # [step2] 验证 RAG 配置
        if self.enable_rag and not self.use_local_rag and not self.pinecone_api_key:
            print("⚠️ 警告: 启用云端 RAG 但未配置 PINECONE_API_KEY，RAG 功能将不可用")
        
        # [step3] 验证路径
        if not self.knowledge_base_dir.exists():
            print(f"⚠️ 警告: 知识库目录不存在: {self.knowledge_base_dir}")
    
    def get_active_llm_config(self) -> dict:
        """获取当前激活的 LLM 配置"""
        # [step1] 返回 Qwen 配置
        if self.llm_provider == "qwen":
            return {
                "provider": "qwen",
                "api_key": self.dashscope_api_key,
                "model": self.qwen_model,
                "temperature": self.llm_temperature
            }
        # [step2] 返回 OpenAI 配置
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.llm_temperature
            }
        # [step3] 返回 Gemini 配置
        elif self.llm_provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": self.google_api_key,
                "model": self.gemini_model,
                "temperature": self.llm_temperature
            }
        # [step4] 返回 Baichuan 配置
        elif self.llm_provider == "baichuan":
            return {
                "provider": "baichuan",
                "api_key": self.baichuan_api_key,
                "model": self.baichuan_model,
                "temperature": self.llm_temperature
            }
        # [step5] 返回 Ollama 配置
        elif self.llm_provider == "ollama":
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "temperature": self.llm_temperature
            }
        # [step6] 返回本地模型配置
        else:
            return {
                "provider": "local",
                "model": self.local_model_path,
                "temperature": self.llm_temperature
            }
    
    def should_use_neo4j(self) -> bool:
        """判断是否应该使用 Neo4j"""
        return self.enable_neo4j and bool(self.neo4j_uri)
    
    def should_use_rag(self) -> bool:
        """判断是否应该使用 RAG"""
        if not self.enable_rag:
            return False
        
        if self.use_local_rag:
            return True  # 本地 RAG 总是可用
        
        return bool(self.pinecone_api_key)  # 云端 RAG 需要 API Key

# [定义函数] ############################################################################################################
# [获取设置] ===========================================================================================================
_settings_cache = None

def get_settings() -> Settings:
    """
    获取全局配置单例。
    :return: Settings 对象
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache

# [导出单例] ==============================================================================================================
settings = get_settings()
