# LightRAG 学习检查清单

用这个清单追踪你的学习进度。每完成一项就打勾 ✅

## 第1周：核心架构 (Jan 29 - Feb 4)

### Day 1: 基础数据流 ⬜
- [X] 运行 week1_day1_basic_flow.py
- [X] 理解 4 种存储的作用
  - [X] KV_STORAGE: 存什么？
  - [X] VECTOR_STORAGE: 存什么？
  - [X] GRAPH_STORAGE: 存什么？
  - [X] DOC_STATUS_STORAGE: 存什么？
- [X] 对比不同查询模式的结果
  - [X] naive 模式特点
  - [X] local 模式特点
  - [X] global 模式特点
  - [X] hybrid 模式特点

### Day 2: lightrag.py 核心类 ⬜
- [ ] 阅读 `LightRAG.__init__()`
  - [ ] 列出 5 个重要的配置参数
- [ ] 阅读 `initialize_storages()`
  - [ ] 理解为什么必须调用
- [ ] 阅读 `ainsert()`
  - [ ] 画出插入流程图
- [ ] 阅读 `aquery()`
  - [ ] 理解如何路由到不同查询模式

### Day 3: operate.py 核心逻辑 ⬜
- [ ] 阅读 `extract_entities()`
  - [ ] LLM prompt 是什么格式？
  - [ ] 返回的实体/关系是什么结构？
- [ ] 阅读 `local_query()`
  - [ ] 如何找到相关实体？
- [ ] 阅读 `global_query()`
  - [ ] "社区"是什么概念？
- [ ] 阅读 `hybrid_query()`
  - [ ] 如何融合 local 和 global？

### Day 4: base.py 存储抽象 ⬜
- [ ] 阅读 `BaseKVStorage`
  - [ ] 核心方法: get, set, delete
- [ ] 阅读 `BaseVectorStorage`
  - [ ] 核心方法: query, upsert, delete_entity
- [ ] 阅读 `BaseGraphStorage`
  - [ ] 核心方法: has_node, add_node, get_node
- [ ] 阅读 `BaseDocStatusStorage`
  - [ ] 用途: 追踪哪些文档已处理

---

## 第2周：存储层实现 (Feb 5 - Feb 11)

### Day 5: MongoDB 图存储 ⬜
- [ ] 阅读 `lightrag/kg/mongo_impl.py`
- [ ] 理解如何实现 `BaseGraphStorage` 接口
- [ ] 理解 workspace 字段如何实现数据隔离
- [ ] 实验: 在 MongoDB Compass 查看数据

### Day 6: Qdrant 向量存储 ⬜
- [ ] 阅读 `lightrag/kg/qdrant_impl.py`
- [ ] 理解如何实现 `BaseVectorStorage` 接口
- [ ] 理解 payload 如何实现 workspace 隔离
- [ ] 实验: 访问 Qdrant Dashboard 查看向量

### Day 7: Redis KV 存储 ⬜
- [ ] 阅读 `lightrag/kg/redis_impl.py`
- [ ] 理解如何实现 `BaseKVStorage` 接口
- [ ] 理解 key 前缀如何实现 workspace 隔离
- [ ] 实验: 使用 RedisInsight 查看缓存

### Day 8: LLM 调用机制 ⬜
- [ ] 阅读 `lightrag/llm/openai.py`
- [ ] 理解 `@wrap_embedding_func_with_attrs` 装饰器
- [ ] 理解 embedding_dim 和 max_token_size 参数
- [ ] 实验: 自定义 embedding 函数

---

## 第3周：检索和查询 (Feb 12 - Feb 18)

### Day 9-10: 深入 operate.py 检索逻辑 ⬜
- [ ] 重读 `kg_query()`
  - [ ] 知识图谱如何被遍历？
- [ ] 阅读 `mix_query()`
  - [ ] 如何结合 KG 和向量检索？
  - [ ] 为什么推荐用 reranker？
- [ ] 阅读 `naive_query()`
  - [ ] 与其他模式的区别

### Day 11: Prompt 工程 ⬜
- [ ] 阅读 `lightrag/prompt.py`
- [ ] 理解实体提取的 prompt 结构
- [ ] 理解查询重写的 prompt
- [ ] 实验: 修改 entity_types，观察提取结果

### Day 12: Reranker 机制 ⬜
- [ ] 阅读 `lightrag/rerank.py`
- [ ] 理解 reranker 如何工作
- [ ] 理解支持的 reranker 模型
- [ ] 实验: 对比有无 reranker 的查询效果

### Day 13: 参数调优 ⬜
- [ ] 实验 `top_k` 的影响
- [ ] 实验 `chunk_top_k` 的影响
- [ ] 实验 `chunk_token_size` 的影响
- [ ] 实验 `entity_extract_max_gleaning` 的影响
- [ ] 总结: 如何为不同场景选择参数

---

## 第4周：实践项目 (Feb 19 - Feb 25)

### Day 14-16: 功能扩展 ⬜
选择一个任务完成:
- [ ] 选项A: 添加新的实体类型 (修改 prompt)
- [ ] 选项B: 实现自定义相似度算法
- [ ] 选项C: 添加查询结果的引用溯源

### Day 17-18: 项目设计 ⬜
- [ ] 确定项目主题 (个人知识库/技术文档问答/课程资料检索)
- [ ] 设计数据来源和处理流程
- [ ] 选择存储后端 (开发 vs 生产)
- [ ] 设计用户交互界面 (CLI/API/Web)

### Day 19-20: API 层快速浏览 ⬜
- [ ] 阅读 `lightrag/api/lightrag_server.py` (30分钟)
- [ ] 理解 REST API 设计
- [ ] 运行 API 服务器
- [ ] 测试 API 端点

### Day 21: 总结和准备 ⬜
- [ ] 整理学习笔记
- [ ] 准备简历材料
- [ ] 规划 RAGAnything 学习路径

---

## 核心概念检查

在第 4 周结束时，你应该能流利地解释：

### 架构层面
- [ ] LightRAG 的整体架构是什么？
- [ ] 为什么要用知识图谱增强 RAG？
- [ ] 4 种存储如何协同工作？

### 实现细节
- [ ] 文档插入的完整流程是什么？
- [ ] 实体和关系是如何提取的？
- [ ] 不同查询模式的检索策略有何不同？
- [ ] workspace 隔离是如何实现的？

### 实践能力
- [ ] 能独立配置和部署 LightRAG
- [ ] 能为不同场景选择合适的查询模式和参数
- [ ] 能阅读和理解源码中的任意模块
- [ ] 能基于 LightRAG 设计自己的 RAG 应用

---

## 学习小贴士

### 时间管理
- 每天投入 2-3 小时
- 70% 读代码，30% 做实验
- 不要陷入细节，先理解主流程

### 学习方法
- 边读边画流程图
- 遇到不懂的先跳过，全局理解后再回来
- 用日志追踪代码执行
- 对比不同实现的差异

### 记录习惯
- 每天写 100 字学习笔记
- 记录遇到的坑和解决方案
- 收集有意思的代码片段

---

最后更新: 2026-01-28