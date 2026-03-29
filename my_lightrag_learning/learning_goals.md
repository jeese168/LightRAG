# LightRAG 学习目标

## 目标 1：4 种存储抽象

- 4 种基本存储类型及其职责
- 为什么分离存储（vs 单一向量数据库）
- 不深入具体实现（MongoDB/Redis/Qdrant 等）

## 目标 2：插入流程设计

- 完整流程：分块 → 实体提取 → Gleaning（多轮补充提取） → 图构建 → 嵌入 → 存储
- 设计巧思和创新点
- 不关注具体介质实现

**补充关注点：**
- 实体去重机制：如何合并不同文本块中的相同实体
- KV 对生成：LLM 为节点/边生成 key（快速匹配）和 value（描述摘要）
- 双层索引结构：vdb_entities（用于 Local）和 vdb_relationships（用于 Global）的设计
- 增量更新：新文档如何增量索引，而非全量重建

## 目标 3：检索流程和 6 种查询模式

- 6 种模式的差异：naive / local / global / hybrid / mix / bypass
- 每种模式的检索逻辑
- 设计思路和创新点
- 不关注具体介质实现

**补充关注点：**
- 双层检索范式：local（细粒度实体邻域）+ global（关系描述/高层关键词）的组合设计
- Global 信息的隐式存储：为什么不需要社区摘要也能回答宏观问题（关键：宏观信息压缩在"关系的描述"里，通过生成的高层 Keywords 命中）
- 向量-图融合检索：语义嵌入匹配 + 图结构遍历的结合方式
- Mix 模式下，它是如何平衡向量检索出的 chunks 和图检索出的 relations 的？（这是真正的难点）

## 目标 4：排序融合和重排序（Rerank）

- LightRAG 中重排序的角色
- 与传统重排序的区别（如果有）
- `enable_rerank` 参数的影响

## 目标 5：Workspace 与多知识库隔离

- **workspace 参数的作用**：实现数据隔离，支持多租户/多项目场景
- **默认值和向后兼容**：
  - 默认为空字符串 `""`（可通过环境变量 `WORKSPACE` 设置）
  - 第一个初始化的 workspace 会成为 `_default_workspace`，允许旧代码不指定 workspace 参数
- **不同存储类型的隔离实现**：
  - **文件系统**（JSON/NetworkX）：通过子目录隔离 `working_dir/workspace_name/`
  - **集合型数据库**（MongoDB/Milvus）：通过集合名前缀隔离 `workspace_entities`
  - **关系型数据库**（PostgreSQL/Neo4j）：通过 workspace 字段过滤，使用组合索引 `(workspace, id)`
  - **Qdrant**：通过 payload 中的 `workspace_id` 字段过滤查询
- **隔离粒度**：4 种存储类型（KV/Vector/Graph/DocStatus）都遵循同样的 workspace 隔离机制
- **应用场景**：
  - 多项目管理：同一套代码服务多个独立知识库
  - 多租户 SaaS：每个租户有独立的知识图谱和数据
  - A/B 测试：不同配置的实验环境互不干扰
  - 并发优化：不同 workspace 可并行操作，同一 workspace 内串行保证一致性

**补充关注点：**
- **跨 workspace 查询**：当前版本不支持跨 workspace 检索，每次查询只针对当前 workspace
- **存储成本**：不同 workspace 的数据物理隔离，不会共享索引或嵌入
- **namespace vs workspace**：namespace 是存储类型（如 `entities_vdb`），workspace 是租户/项目隔离标识
- **与其他 RAG 框架对比**：多数框架需要多实例部署实现多租户，LightRAG 通过 workspace 参数实现轻量级隔离

## 目标 6：对比分析

LightRAG（向量 + 图检索）vs 工业界主流（向量 + 关键词检索）：

- 各自的优势
- 各自的劣势
- 适用场景差异

**补充对比维度：**
- Token 效率：LightRAG vs 传统 GraphRAG（据报告称低 6000 倍，需验证）
- 索引成本：图构建需要多次 LLM 调用的代价
- 实时性：预处理 vs 即时检索的 trade-off

**LightRAG vs Microsoft GraphRAG（为什么不用社区发现）：**
- GraphRAG：Leiden 算法做层级聚类 → 为每个社区生成 Summary → 非常贵、非常慢
- LightRAG：**去掉 Leiden 社区发现**，不预先计算社区摘要
- LightRAG 的替代方案：插入时只生成 KV 对，检索时通过"生成高层关键词"隐式定位全局信息
- 这是 LightRAG Token 效率高的核心原因之一

## 目标 7：与 RAG-Anything 的承袭关系（为后续学习铺垫）

- RAG-Anything 复用了 LightRAG 的哪些模块（存储、图检索、增量更新）
- RAG-Anything 新增了什么（多模态解析层、双图构建、跨模态锚点）
- 两者的定位差异：轻量文本检索 vs 全模态文档处理

## 完成标准

能够清晰讲述：
1. 4 种存储的设计逻辑
2. 插入流程的处理步骤和创新点（双层索引、实体去重、KV 对生成）
3. 检索流程和各模式的本质区别（双层检索、Global 隐式存储）
4. 重排序的作用
5. Workspace 隔离机制和应用场景（多租户、多项目、不同存储类型的实现方式）
6. 与主流方案的对比（vs 向量+关键词、vs Microsoft GraphRAG）
7. LightRAG 和 RAG-Anything 的关系（为过渡做准备）

---

学完后转 RAG-Anything
