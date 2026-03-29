# Utils 工具包

学习 LightRAG 过程中的常用工具函数集合。

---

### siliconflow_llm_complete

硅基流动 LLM 完成函数，用于调用 DeepSeek-V3.2 等模型。

#### 功能
自动从项目根目录 `.env` 读取配置，调用硅基流动 API 进行文本生成。

#### 内部参数处理
函数会自动过滤 LightRAG 内部参数，避免传给 API：
- `enable_cot` - 思维链标志
- `keyword_extraction` - 关键词提取标志
- `hashing_kv` - 缓存存储对象

这些参数由 LightRAG 传递，函数内部自动处理，不会发送给硅基流动 API。

#### 环境变量
- `LLM_MODEL` - 模型名称
- `LLM_BINDING_HOST` - API 地址
- `LLM_BINDING_API_KEY` - API 密钥

#### 使用示例
```python
from utils import siliconflow_llm_complete

rag = LightRAG(
    llm_model_func=siliconflow_llm_complete,
    # ...
)
```

---

### siliconflow_embed

硅基流动 Embedding 函数，用于文本向量化。

#### 功能
调用 BAAI/bge-m3 等 embedding 模型，将文本转换为向量。

#### 环境变量
- `EMBEDDING_MODEL` - 模型名称
- `EMBEDDING_BINDING_HOST` - API 地址
- `EMBEDDING_BINDING_API_KEY` - API 密钥
- `EMBEDDING_DIM` - 向量维度
- `EMBEDDING_TOKEN_LIMIT` - 最大 token 数

#### 使用示例
```python
from utils import siliconflow_embed

rag = LightRAG(
    embedding_func=siliconflow_embed,
    # ...
)
```

---

## 配置说明

工具包会自动从项目根目录的 `.env` 文件加载配置：
```
/Users/jeese/Projects/LightRAG/.env
```

确保在运行学习脚本前已正确配置该文件。