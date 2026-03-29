from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Callable,
    Optional,
    Dict,
    List,
    AsyncIterator,
)
from .utils import EmbeddingFunc
from .types import KnowledgeGraph
from .constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_OLLAMA_MODEL_SIZE,
    DEFAULT_OLLAMA_CREATED_AT,
    DEFAULT_OLLAMA_DIGEST,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class OllamaServerInfos:
    """Ollama 服务器模型信息配置类"""
    def __init__(self, name=None, tag=None):
        # 从环境变量或使用默认值获取模型名称
        self._lightrag_name = name or os.getenv(
            "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
        )
        # 从环境变量或使用默认值获取模型标签（版本号）
        self._lightrag_tag = tag or os.getenv(
            "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
        )
        # 模型大小
        self.LIGHTRAG_SIZE = DEFAULT_OLLAMA_MODEL_SIZE
        # 模型创建时间
        self.LIGHTRAG_CREATED_AT = DEFAULT_OLLAMA_CREATED_AT
        # 模型摘要（唯一标识）
        self.LIGHTRAG_DIGEST = DEFAULT_OLLAMA_DIGEST

    @property
    def LIGHTRAG_NAME(self):
        """获取模型名称"""
        return self._lightrag_name

    @LIGHTRAG_NAME.setter
    def LIGHTRAG_NAME(self, value):
        """设置模型名称"""
        self._lightrag_name = value

    @property
    def LIGHTRAG_TAG(self):
        """获取模型标签（版本）"""
        return self._lightrag_tag

    @LIGHTRAG_TAG.setter
    def LIGHTRAG_TAG(self, value):
        """设置模型标签（版本）"""
        self._lightrag_tag = value

    @property
    def LIGHTRAG_MODEL(self):
        """获取完整模型标识，格式为 name:tag（如 llama2:latest）"""
        return f"{self._lightrag_name}:{self._lightrag_tag}"


class TextChunkSchema(TypedDict):
    """文本分块的数据结构定义"""
    tokens: int  # 该分块包含的 token 数量
    content: str  # 分块的实际文本内容
    full_doc_id: str  # 该分块所属的源文档 ID
    chunk_order_index: int  # 分块在整个文档中的顺序位置


T = TypeVar("T")


@dataclass
class QueryParam:
    """LightRAG 查询执行的配置参数"""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    """指定检索模式：
    - "local": 基于局部上下文的信息检索（如周围实体）
    - "global": 利用全局知识进行检索
    - "hybrid": 结合局部和全局检索方法
    - "naive": 执行基础搜索而不使用高级技巧
    - "mix": 整合知识图谱和向量检索
    - "bypass": 绕过检索，直接生成响应
    """

    only_need_context: bool = False
    """如果为 True，仅返回检索的上下文，不生成响应"""

    only_need_prompt: bool = False
    """如果为 True，仅返回生成的提示词，不生成最终响应"""

    response_type: str = "Multiple Paragraphs"
    """定义响应格式。示例：'Multiple Paragraphs'（多段落）、'Single Paragraph'（单段落）、'Bullet Points'（要点列表）"""

    stream: bool = False
    """如果为 True，启用流式输出以实现实时响应"""

    top_k: int = int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    """检索的前 K 个结果数量。在 'local' 模式中代表实体数，在 'global' 模式中代表关系数"""

    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", str(DEFAULT_CHUNK_TOP_K)))
    """从向量搜索中初步检索的文本块数量，经过重新排序后保留的数量。
    如果为 None，则默认使用 top_k 的值
    """

    max_entity_tokens: int = int(
        os.getenv("MAX_ENTITY_TOKENS", str(DEFAULT_MAX_ENTITY_TOKENS))
    )
    """在统一的 token 控制系统中，为实体上下文分配的最大 token 数"""

    max_relation_tokens: int = int(
        os.getenv("MAX_RELATION_TOKENS", str(DEFAULT_MAX_RELATION_TOKENS))
    )
    """在统一的 token 控制系统中，为关系上下文分配的最大 token 数"""

    max_total_tokens: int = int(
        os.getenv("MAX_TOTAL_TOKENS", str(DEFAULT_MAX_TOTAL_TOKENS))
    )
    """整个查询上下文的最大 token 总预算（包括实体 + 关系 + 文本块 + 系统提示词）"""

    hl_keywords: list[str] = field(default_factory=list)
    """高级关键词列表，在检索中优先处理"""

    ll_keywords: list[str] = field(default_factory=list)
    """低级关键词列表，用于细化检索焦点"""

    # 历史消息仅发送给 LLM 用于上下文，不用于检索
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """存储过去的对话历史以维持上下文。
    格式：[{"role": "user/assistant", "content": "消息"}]
    """

    # TODO: 已弃用。不再在代码库中使用，所有 conversation_history 消息都发送给 LLM
    history_turns: int = int(os.getenv("HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS)))
    """在响应上下文中要考虑的完整对话轮数（用户-助手对的数量）"""

    model_func: Callable[..., object] | None = None
    """可选的 LLM 模型函数覆盖，用于这个特定查询。
    如果提供，将使用该函数而不是全局模型函数。
    这允许针对不同的查询模式使用不同的模型。
    """

    user_prompt: str | None = None
    """用户提供的查询提示词。
    为 LLM 提供额外的指令。如果提供，将被注入到提示词模板中。
    其目的是让用户自定义 LLM 生成响应的方式。
    """

    enable_rerank: bool = os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"
    """启用对检索文本块的重新排序。如果为 True 但未配置重排模型，将发出警告。
    默认为 True，在重排模型可用时启用重排。
    """

    include_references: bool = False
    """如果为 True，在响应中包含参考文献列表（针对支持的端点）。
    此参数控制 API 响应是否包含 references 字段，
    该字段包含检索内容的引用信息。
    """


@dataclass
class StorageNameSpace(ABC):
    """存储命名空间基类"""
    namespace: str  # 命名空间标识
    workspace: str  # 工作区路径
    global_config: dict[str, Any]  # 全局配置字典

    async def initialize(self):
        """初始化存储"""
        pass

    async def finalize(self):
        """完成存储操作"""
        pass

    @abstractmethod
    async def index_done_callback(self) -> None:
        """在索引完成后提交存储操作"""

    @abstractmethod
    async def drop(self) -> dict[str, str]:
        """删除存储中的所有数据并清理资源

        此抽象方法定义了删除存储实现中所有数据的合同。
        每种存储类型必须实现此方法以：
        1. 清除内存和/或外部存储中的所有数据
        2. 删除任何关联的存储文件
        3. 将存储重置为初始状态
        4. 处理任何资源的清理
        5. 如果必要，通知其他进程
        6. 此操作应立即将数据持久化到磁盘

        返回值：
            dict[str, str]: 操作状态和消息，格式如下：
                {
                    "status": str,  # "success" 或 "error"
                    "message": str  # 成功时为 "data dropped"，失败时为错误详情
                }

        实现特定细节：
        - 成功时: 返回 {"status": "success", "message": "data dropped"}
        - 失败时: 返回 {"status": "error", "message": "<错误详情>"}
        - 如果不支持: 返回 {"status": "error", "message": "unsupported"}
        """


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    """向量存储基类"""
    embedding_func: EmbeddingFunc  # 嵌入函数
    cosine_better_than_threshold: float = field(default=0.2)  # 余弦相似度阈值
    meta_fields: set[str] = field(default_factory=set)  # 元数据字段集合

    def _validate_embedding_func(self):
        """验证嵌入函数是否已提供。

        此方法应在所有向量存储实现的 __post_init__ 开始时调用。

        抛出：
            ValueError: 如果 embedding_func 为 None
        """
        if self.embedding_func is None:
            raise ValueError(
                "embedding_func is required for vector storage. "
                "Please provide a valid EmbeddingFunc instance."
            )

    def _generate_collection_suffix(self) -> str | None:
        """从嵌入函数生成集合/表后缀。

        如果嵌入函数中存在 model_name，返回后缀，否则返回 None。
        注意：嵌入函数的存在已在 __post_init__ 中验证。

        返回值：
            str | None: 后缀字符串，如 "text_embedding_3_large_3072d"，或如果 model_name 不可用则返回 None
        """
        import re

        # 检查 model_name 是否存在（model_name 在 EmbeddingFunc 中是可选的）
        model_name = getattr(self.embedding_func, "model_name", None)
        if not model_name:
            return None

        # embedding_dim 在 EmbeddingFunc 中是必需的
        embedding_dim = self.embedding_func.embedding_dim

        # 生成后缀：清理模型名称并附加维度
        safe_model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name.lower())
        return f"{safe_model_name}_{embedding_dim}d"

    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """查询向量存储并检索前 top_k 个结果。

        参数：
            query: 要搜索的查询字符串
            top_k: 要返回的顶部结果数量
            query_embedding: 可选的预计算查询嵌入。
                           如果提供，则跳过嵌入计算以获得更好的性能。
        """

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """在存储中插入或更新向量。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏
        """

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """按名称删除单个实体。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏
        """

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """删除给定实体的所有关系。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏
        """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """通过 ID 获取向量数据

        参数：
            id: 向量的唯一标识符

        返回值：
            如果找到向量数据则返回该数据，否则返回 None
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """通过 ID 列表获取多个向量数据

        参数：
            ids: 唯一标识符列表

        返回值：
            找到的向量数据对象列表
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """删除指定 ID 的向量

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            ids: 要删除的向量 ID 列表
        """

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """按 ID 获取向量，仅返回 ID 和向量数据以提高效率

        参数：
            ids: 唯一标识符列表

        返回值：
            ID 到其向量嵌入的映射字典
            格式：{id: [向量值], ...}
        """
        pass


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    """键值对存储基类"""
    embedding_func: EmbeddingFunc

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """按 ID 获取值"""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """按 ID 列表获取值"""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """返回不存在的键"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """插入或更新数据

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 更新标志以通知其他进程数据持久化是必需的
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """按 ID 从存储中删除特定记录

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 更新标志以通知其他进程数据持久化是必需的

        参数：
            ids (list[str]): 要从存储中删除的文档 ID 列表

        返回值：
            None
        """

    @abstractmethod
    async def is_empty(self) -> bool:
        """检查存储是否为空

        返回值：
            bool: 如果存储不包含任何数据则返回 True，否则返回 False
        """


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """图存储基类。所有与图中边相关的操作应该是无向的。"""

    embedding_func: EmbeddingFunc

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """检查图中是否存在节点。

        参数：
            node_id: 要检查的节点 ID

        返回值：
            如果节点存在返回 True，否则返回 False
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """检查两个节点之间是否存在边。

        参数：
            source_node_id: 源节点的 ID
            target_node_id: 目标节点的 ID

        返回值：
            如果边存在返回 True，否则返回 False
        """

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """获取节点的度数（连接的边数）。

        参数：
            node_id: 节点的 ID

        返回值：
            连接到该节点的边数
        """

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """获取边的总度数（其源节点和目标节点的度数之和）。

        参数：
            src_id: 源节点的 ID
            tgt_id: 目标节点的 ID

        返回值：
            源节点和目标节点的度数之和
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """按 ID 获取节点，仅返回节点属性。

        参数：
            node_id: 要检索的节点 ID

        返回值：
            如果找到节点，返回节点属性字典，否则返回 None
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """获取两个节点之间的边属性。

        参数：
            source_node_id: 源节点的 ID
            target_node_id: 目标节点的 ID

        返回值：
            如果找到边，返回边属性字典，否则返回 None
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """获取连接到节点的所有边。

        参数：
            source_node_id: 要获取边的节点 ID

        返回值：
            表示边的 (source_id, target_id) 元组列表，
            如果节点不存在则返回 None
        """

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """批量使用 UNWIND 获取节点

        默认实现逐个获取节点。
        重写此方法以在存储后端支持批量操作时获得更好的性能。
        """
        result = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                result[node_id] = node
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """批量使用 UNWIND 获取节点度数

        默认实现逐个获取节点度数。
        重写此方法以在存储后端支持批量操作时获得更好的性能。
        """
        result = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            result[node_id] = degree
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """批量使用 UNWIND 获取边度数，同时使用 node_degrees_batch

        默认实现逐个计算边度数。
        重写此方法以在存储后端支持批量操作时获得更好的性能。
        """
        result = {}
        for src_id, tgt_id in edge_pairs:
            degree = await self.edge_degree(src_id, tgt_id)
            result[(src_id, tgt_id)] = degree
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """批量使用 UNWIND 获取边

        默认实现逐个获取边。
        重写此方法以在存储后端支持批量操作时获得更好的性能。
        """
        result = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                result[(src_id, tgt_id)] = edge
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """批量使用 UNWIND 获取节点的边

        默认实现逐个获取节点的边。
        重写此方法以在存储后端支持批量操作时获得更好的性能。
        """
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """在图中插入新节点或更新现有节点。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            node_id: 要插入或更新的节点 ID
            node_data: 节点属性字典
        """

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """在图中插入新边或更新现有边。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            source_node_id: 源节点的 ID
            target_node_id: 目标节点的 ID
            edge_data: 边属性字典
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """从图中删除节点。

        内存存储的重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            node_id: 要删除的节点 ID
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """删除多个节点

        重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            nodes: 要删除的节点 ID 列表
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """删除多个边

        重要说明：
        1. 变更将在下一次 index_done_callback 时持久化到磁盘
        2. 在 index_done_callback 之前，只有一个进程应该更新存储，
           应使用 KG-storage-log 避免数据损坏

        参数：
            edges: 要删除的边列表，每条边都是 (source, target) 元组
        """

    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """获取图中的所有标签（实体名称）。
        不要在大图中使用此方法，而应使用 get_popular_labels 或 search_labels。

        返回值：
            图中所有节点标签的列表，按字母顺序排序
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        检索一个连通的节点子图，其中标签包含指定的 `node_label`。

        参数：
            node_label: 起始节点的标签（实体名称），* 表示所有节点
            max_depth: 子图的最大深度，默认为 3
            max_nodes: 返回的最大节点数，默认为 1000（如果可能使用 BFS)

        返回值：
            KnowledgeGraph 对象包含节点和边，
            以及 is_truncated 标志指示图是否由于 max_nodes 限制而被截断
        """

    @abstractmethod
    async def get_all_nodes(self) -> list[dict]:
        """获取图中的所有节点。

        返回值：
            所有节点的列表，其中每个节点都是其属性的字典
            （某些存储实现中边是双向的；调用者必须处理去重）
        """

    @abstractmethod
    async def get_all_edges(self) -> list[dict]:
        """获取图中的所有边。

        返回值：
            所有边的列表，其中每条边都是其属性的字典
        """

    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """按节点度数获取流行的标签（实体名称）（最连通的实体）

        参数：
            limit: 要返回的最大标签数

        返回值：
            按度数排序的标签列表（最高优先）
        """

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """使用模糊匹配搜索标签（实体名称）

        参数：
            query: 搜索查询字符串
            limit: 返回的最大结果数

        返回值：
            按相关性排序的匹配标签列表
        """


class DocStatus(str, Enum):
    """文档处理状态"""

    PENDING = "pending"  # 等待处理
    PROCESSING = "processing"  # 正在处理
    PREPROCESSED = "preprocessed"  # 预处理完成
    PROCESSED = "processed"  # 完全处理完成
    FAILED = "failed"  # 处理失败


@dataclass
class DocProcessingStatus:
    """文档处理状态数据结构"""

    content_summary: str
    """文档内容前100字符，用于预览"""
    content_length: int
    """文档总长度"""
    file_path: str
    """文档的文件路径"""
    status: DocStatus
    """当前处理状态"""
    created_at: str
    """文档创建时的 ISO 格式时间戳"""
    updated_at: str
    """文档最后更新时的 ISO 格式时间戳"""
    track_id: str | None = None
    """用于监控进度的追踪 ID"""
    chunks_count: int | None = None
    """分块后的块数，用于处理"""
    chunks_list: list[str] | None = field(default_factory=list)
    """与此文档关联的分块 ID 列表，用于删除"""
    error_msg: str | None = None
    """失败时的错误信息"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """额外元数据"""
    multimodal_processed: bool | None = field(default=None, repr=False)
    """内部字段：指示是否完成了多模态处理。在 repr() 中不显示，但可用于调试。"""

    def __post_init__(self):
        """
        基于 multimodal_processed 字段处理状态转换。

        业务规则：
        - 如果 multimodal_processed 为 False 且状态为 PROCESSED，
          则将状态更改为 PREPROCESSED
        - multimodal_processed 字段被保留（repr=False）供内部使用和调试
        """
        # 应用状态转换逻辑
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """文档状态存储的基类"""

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """获取每个状态的文档计数"""

    @abstractmethod
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """获取具有特定状态的所有文档"""

    @abstractmethod
    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """获取具有特定 track_id 的所有文档"""

    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """获取支持分页的文档

        参数：
            status_filter: 按文档状态筛选，None 表示所有状态
            page: 页码（1 开始）
            page_size: 每页的文档数（10-200）
            sort_field: 排序字段（'created_at'、'updated_at'、'id'）
            sort_direction: 排序方向（'asc' 或 'desc'）

        返回值：
            (文档列表元组、总计数) 的元组
        """

    @abstractmethod
    async def get_all_status_counts(self) -> dict[str, int]:
        """获取所有文档每个状态的计数

        返回值：
            将状态名称映射到计数的字典
        """

    @abstractmethod
    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        """按文件路径获取文档

        参数：
            file_path: 要搜索的文件路径

        返回值：
            dict[str, Any] | None: 如果找到文档则返回文档数据，否则返回 None
            返回格式与 get_by_ids 方法相同
        """


class StoragesStatus(str, Enum):
    """存储系统状态"""

    NOT_CREATED = "not_created"  # 存储未创建
    CREATED = "created"  # 存储已创建
    INITIALIZED = "initialized"  # 存储已初始化并可使用
    FINALIZED = "finalized"  # 存储已关闭


@dataclass
class DeletionResult:
    """代表一次删除操作的结果。"""

    status: Literal["success", "not_found", "fail"]  # 成功/未找到/失败
    doc_id: str  # 被删除的文档 ID
    message: str  # 操作信息
    status_code: int = 200  # HTTP 状态码
    file_path: str | None = None  # 文件路径（可选）


# Unified Query Result Data Structures for Reference List Support


@dataclass
class QueryResult:
    """
    统一的查询结果数据结构，适用于所有查询模式。

    属性：
        content: 非流式响应的文本内容
        response_iterator: 流式响应的迭代器
        raw_data: 完整的结构化数据，包括参考文献和元数据
        is_streaming: 是否为流式结果
    """

    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """
        从 raw_data 中提取参考文献列表的便捷属性。

        返回值：
            List[Dict[str, str]]: 参考文献列表，格式为：
            [{"reference_id": "1", "file_path": "/path/to/file.pdf"}, ...]
        """
        if self.raw_data:
            return self.raw_data.get("data", {}).get("references", [])
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        从 raw_data 中提取元数据的便捷属性。

        返回值：
            Dict[str, Any]: 查询元数据，包括查询模式、关键词等
        """
        if self.raw_data:
            return self.raw_data.get("metadata", {})
        return {}


@dataclass
class QueryContextResult:
    """
    统一的查询上下文结果数据结构。

    属性：
        context: LLM 上下文字符串
        raw_data: 完整的结构化数据，包括参考文献列表
    """

    context: str
    raw_data: Dict[str, Any]

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """从 raw_data 中提取参考文献列表的便捷属性。"""
        return self.raw_data.get("data", {}).get("references", [])
