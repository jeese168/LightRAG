# LightRAG 学习笔记

## 第1周：核心架构与数据流

### 学习目标
- 理解 LightRAG 的 4 种存储类型及其作用
- 掌握文档插入和查询的完整流程
- 理解不同查询模式的区别

### 每天的任务

#### Day 1 (今天) - 运行第一个实验
```bash
# 确保在项目根目录
cd /Users/jeese/Projects/LightRAG

# 激活虚拟环境
source .venv/bin/activate

# 运行实验
python my_lightrag_learning/week1_day1_basic_flow.py
```

**今天要回答的问题**:
- [X] LightRAG 有哪 4 种存储？各自存什么数据？
- 分别是四大存储BaseVectorStorage：向量存储，用于相似度计算。BaseGraphStorage：图存储，用于知识图谱推理。DocStatusStorage：这个也是键值对存储但是是文档的状态，用于文档及其状态的对应。
    -（回答错误成存文档的元数据）BaseKVStorage：键值对存储，完整的原始文档内容 (full_docs)、实体及其关联的分块列表 (entity_chunks)、关系及其关联的分块列表 (relation_chunks)、文本分块本身 (chunks)、LLM 响应缓存 (llm_response_cache)、纠正：不仅仅是元数据，而是实际内容数据。可以理解为"索引"的反向映射。
- [X] 为什么必须调用 `initialize_storages()`？
- day1没有看到这个方法，`initialize_storages()`应该是day2的内容
- [X] naive/local/global/hybrid 模式有什么区别？
- 这个应该是查询模式，lightRag支持很多查询模式，作者在注释写的很清楚，但是也没写具体是怎么实现的，所以我的猜测放在下面的括号里
    - 回答正确的：
        - "hybrid": 结合局部和全局检索方法（应该是上面两种的结合版，但是局部的话可能权重会更高一点）
        - "naive": 执行基础搜索而不使用高级技巧（这个注释写的比较抽象，那我就觉得是应该是用相似度检索，不用知识图谱）
        - "mix": 整合知识图谱和向量检索（这个的话就类似于混合检索，但是是向量检索和知识图谱（KG 检索）的检索排序融合，再进行重排序，当然前提是使用重排序）
        - "bypass": 绕过检索，直接生成响应（这个就是不检索，直接利用大模型的预训练知识来回答问题）
    - 回答错误的
        - "local": 基于局部上下文的信息检索（如周围实体）（我理解是用到了知识图谱，但是只在检索的实体及其周围的实体，也就是检索的边长或者步长有限）
        **local 模式：基于查询相关的直接实体，然后找这些实体的邻近关系（1-2 跳）与其说"周围"，不如说是"指定实体的直接邻域"**
        - "global": 利用全局知识进行检索（我理解是用到了知识图谱，而且会在整体知识图谱中来进行检索，范围较大）
        **global 模式：不关注特定实体，而是找关键社区和核心关系，它在寻找的是"知识结构"而不是"特定实体"**

#### Day 2 - 深入 lightrag.py
**阅读文件**: `lightrag/lightrag.py` (178KB，快速扫描 + 重点精读)

**第1步: `__init__()` 构造函数** (line ~154-500，看30分钟)
- 找到所有以 `_storage` 结尾的参数 → 哪些是必须的？哪些有默认值？
- `_storage` 结尾的参数找到了kv_storage、vector_storage、graph_storage、doc_status_storage，但实际上这4个参数只是字符而已，不是具体的实例，默认值的话应该是默认存储的那些JsonKVStorage、NanoVectorDBStorage、NetworkXStorage、JsonDocStatusStorage。
- 找到 `locate(f"lightrag.kg.{kv_storage}")` → 为什么用字符串动态加载类？
- 主要我觉得比较灵活，另外可以很好的结合那个配置文件来进行加载选择，具体在进行实例化的时候根据字符串结合多态的特性来选择对应加载的类
- 找到 `chunk_token_size`, `entity_extract_max_gleaning` → 默认值是多少？为什么？
- 这两个默认值，源码都写死了，一个是1200token，另外一个是1次，我觉得1200可能是一个衡量值吧，我也说不出为什么是这个值，但是我知道太少了不好，那是因为一个切块它的语义信息太少了太长了也不好那相当于我每一次检索得到的上下文和token消耗量也变高了，所以可以理解为深度学习的一个超参数吧，它这种设置是有道理的。
- 第2个参数的语义是似乎是大模型对含糊的实体抽取的重试最大次数这个设置为一的话，可能是为防止设置太长的话，会影响使用体验，毕竟我印象中插入文档是一个很费时的事情。

**带着问题读**:
1. 为什么 `llm_model_func` 和 `embedding_func` 没有默认值？
   
2. `workspace` 参数在 `__init__` 里只是赋值，真正用在哪？
    workspace工作空间，它的意意思是一个存储实力的具体存储位置，也就是说抽象的概念4种存储，文档状态存储，键值对存储，向量存储，知识实体存储，以及实际上十几种具体的存储类型，实际存储在哪个目录或者说什么空间下。应该是在`initialize_storages()` 初始化
3. 存储类只是 `locate()` 了，什么时候真正创建实例？
    这个应该是在__post_init__阶段就根据字符串解析导入了具体哪种实现类并且利用python的特殊语法解耦并实例化存储对象，所以就是在init阶段进行实例化的，在`initialize_storages()` 来进行配置。

**第2步: `initialize_storages()` 初始化** (line ~600-800，看20分钟)
- 找到 4 个存储的创建代码 → 传了哪些参数给存储类？
    具体来说，这4个存储的创创建代码，创建了很多存储实例，而且不同的工作空间有不同实例
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        workspace=self.workspace,
        global_config=global_config,
        embedding_func=self.embedding_func,
        其中global_config=global_config,这个可不加，因为前面通过那个self.key_string_value_json_storage_cls = partial(预定义了这个参数，相当于预装载了
- 找到 `__aenter__()` 的调用 → 这是什么模式？（提示：上下文管理器）
        这是什么模式？这是异步模式吧？获取锁时不等待，而是把cpu执行权让给其他通过唤醒的方式，再进行执行。
- 找到 `_load_cached_data()` → 启动时加载了什么到内存？
        这个已经取消使用了。

**带着问题读**:
1. 如果不调用 `initialize_storages()` 直接 `ainsert()`，会在哪一行报错？
    对于这个问题，每个具体的类都不一样，比如我看到redis代码里面初始化过程中，先加了锁，然后测试连接，如果不经过这一步的话，那可能直接就插入，如果连接是断开的也就是没经过测试，那么可能会插入错误
2. `finalize_storages()` 在什么情况下必须调用？
    在关闭不使用的时候肯定必须要调用，因为这是一些收尾工作，比如说将内存的数据写入磁盘，以及释放连接，释放内存，否则会导致内存泄露。
3. 为什么初始化是 `async` 的？
    因为初始化真正包含真正的io操作，所以说这是个比较费神的操作呃对于默认而言它可能是内存的磁盘，还有对于其他存储，它可能是网络io，所以这样比较提高性能，而Python规定__init__函数不能是 async 的所以这个框架来了一个新的初始化方法提高性能

**第3步: `ainsert()` 文档插入** (line ~1500-1700，看30分钟)
- 找到 `compute_mdhash_id()` → 文档 ID 如何生成？
    这玩意儿在入队的时候生成apipeline_enqueue_documents，判断有没有传入，如果没有传入的话，就用md5生成，因为md5主要文本内容要就生成是唯一的
- 找到调用 `operate.py` 的地方 → 具体是哪个函数处理实体提取？
    在处理（apipeline_process_enqueue_documents）的时候来进行提取，在第4步来进行实体提取吧，好像还要悟道大模型来抽取实体关系
- 找到写入 4 种存储的代码 → 分别在哪几行？
    具体来说，是4种存储基类，应该有十几种存储具体的东西，比如说文档的状态等等等等在入队的时候（apipeline_enqueue_documents）就有存储状态和待处理文档的记录、处理时候（apipeline_process_enqueue_documents）也会有切块的时候会存储一部分，然后最后提取实体关系之后又会存储一大部分

**带着问题读**:
1. 如果插入重复文档会怎样？（提示：看 DOC_STATUS_STORAGE 检查）
    不会怎么样，因为入队中有一部分逻辑就是在去重。而且他会两步骤去重，第1步骤去的重的话是本次插入之中不重复的，第2次检查的话是已经插入的和准备要插入的有没有重复的
2. `max_parallel_insert=2` 为什么不设置更大？
    我觉得有好几个方面原因吧，首先这个参数，它是为了限制同时处理的文档数量，也就是apipeline_process_enqueue_documents操作时，限制真正并行处理的数量，可以设置大，如果设置大的话，那内存会剧增。然后第2点的话就是官方的api提供商也会限制他的那个请求量。
3. 插入失败会如何处理？数据一致性如何保证？
    插入失败也没事，如果中间出现了失败，那么会把文档的状态标记为失败落实到数据库或者具体的存储文件之中，但是重新开始的时候会获取三种状态，其中一种就是失败的也就是会重新进行处理。

**第4步: `aquery()` 查询路由** (line ~1800-2000，看20分钟)
- 找到 `if param.mode == "local"` 的分支 → 每个 mode 调用了哪个函数？
- 找到这些函数的 import 语句 → 它们在哪个文件？（`operate.py`）
- 找到 `stream=True` 的处理 → 流式输出如何实现？

**带着问题读**:
1. 为什么 `aquery()` 本身很简单？（提示：编排层 vs 执行层）
2. `QueryParam` 参数如何传递给底层函数？
3. 如果 mode 传了不支持的值会怎样？

**实验**: 运行 `python my_lightrag_learning/week1_day2_experiment.py`

**今天要回答的问题**:
- [X] 列出 5 个影响内存/性能的配置参数
    首先上面说的这个max_parallel_insert那肯定是了，
- [X] 画出 `ainsert()` 的流程图（7-10个步骤）
    这个画了，画的很抽象因为实在太他妈复杂了，我感觉还是不要纠结于细节吧。
- [X] 为什么存储初始化用"延迟加载"而不是在 `__init__` 里？
    感觉存储初始化，比如说，你无论是磁盘io还是什么，网络io，也就是连接数据库都是一个比较耗时的操作，所以它需要那个关键字async但是Python的语法好像规定，你在构造函数，也就是init方法是不能用这个关键字的

#### Day 3 - 深入 operate.py
**阅读文件**: `lightrag/operate.py`

**重点看**:
- `extract_entities()`: 如何调用 LLM 提取实体？
- `local_query()` vs `global_query()`: 检索逻辑有何不同？
- `kg_query()`: 知识图谱如何被查询？

**实验**: 打印出 LLM 的 prompt，理解提取逻辑

#### Day 4 - 理解 base.py 和存储抽象
**阅读文件**: `lightrag/base.py`

**重点看**:
- `BaseKVStorage`: 定义了哪些接口？
- `BaseVectorStorage`: 向量存储需要实现什么？
- `BaseGraphStorage`: 图存储的核心方法
- `BaseDocStatusStorage`: 如何追踪文档状态

**实验**: 对比 JSON 存储实现 vs 你配置的 MongoDB/Qdrant/Redis

---

## 学习检查点

### 第1周结束时，你应该能回答：

**架构理解**:
1. LightRAG 的数据流向是什么？(文档 → 分块 → 提取 → 存储 → 查询)
2. 4 种存储在查询时分别发挥什么作用？
3. `workspace` 参数如何实现数据隔离？

**代码理解**:
1. `ainsert()` 内部调用了哪些函数？
2. 为什么切换 embedding 模型要清空存储？
3. 知识图谱的"实体"和"关系"是如何表示的？

**实践能力**:
1. 能独立创建一个 RAG 实例并插入文档
2. 能选择合适的查询模式解决不同问题
3. 能配置基本参数 (top_k, chunk_token_size 等)

---

## 学习资源

### 核心文件阅读顺序
1. ✅ `lightrag/base.py` (10分钟) - 理解抽象接口
2. ⬜ `lightrag/lightrag.py` (30分钟) - 核心编排
3. ⬜ `lightrag/operate.py` (60分钟) - 实体提取和检索逻辑
4. ⬜ `lightrag/prompt.py` (15分钟) - LLM prompt 模板

### 可选阅读
- `lightrag/kg/mongo_impl.py` - 你的图存储实现
- `lightrag/kg/qdrant_impl.py` - 你的向量存储实现
- `lightrag/kg/redis_impl.py` - 你的 KV 存储实现

### 调试技巧
```python
# 1. 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. 查看存储内容
# MongoDB: 使用 MongoDB Compass
# Qdrant: 访问 http://8.155.39.169:6333/dashboard
# Redis: 使用 RedisInsight

# 3. 打印中间结果
# 在 operate.py 中添加 print 语句
```

---

## 下一步计划

### 第2周: 存储层实现
- 阅读 MongoDB/Qdrant/Redis 存储实现
- 理解 workspace 隔离机制
- 实验不同存储后端的性能差异

### 第3周: 检索和查询
- 深入 operate.py 的检索逻辑
- 理解 reranker 的作用
- 实验参数调优 (top_k, chunk_top_k, etc.)

### 第4周: 实践项目
- 实现一个小功能扩展
- 设计自己的 RAG 项目
- 准备简历材料