"""
硅基流动 API 配置工具
====================
提供 LLM 和 Embedding 函数，自动从 .env 读取配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from lightrag.llm.openai import openai_embed
from lightrag.utils import wrap_embedding_func_with_attrs

# 加载项目根目录的 .env 文件
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


async def siliconflow_llm_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot=False,
    keyword_extraction=False,
    **kwargs
) -> str:
    """
    硅基流动 LLM 完成函数

    使用环境变量:
    - LLM_MODEL: 模型名称 (默认: deepseek-ai/DeepSeek-V3.2)
    - LLM_BINDING_HOST: API 地址 (默认: https://api.siliconflow.cn/v1)
    - LLM_BINDING_API_KEY: API 密钥

    参数:
    - enable_cot: 启用思维链（由 LightRAG 传递，此处忽略）
    - keyword_extraction: 关键词提取模式（由 LightRAG 传递，此处忽略）
    """
    if history_messages is None:
        history_messages = []

    # 移除 LightRAG 内部参数，避免传给 OpenAI API
    kwargs.pop("hashing_kv", None)

    client = AsyncOpenAI(
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.siliconflow.cn/v1")
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2"),
        messages=messages,
        **kwargs
    )

    return response.choices[0].message.content


# 配置硅基流动的 Embedding (bge-m3)
@wrap_embedding_func_with_attrs(
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
    max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192"))
)
async def siliconflow_embed(texts: list[str]) -> list[list[float]]:
    """
    硅基流动 Embedding 函数

    使用环境变量:
    - EMBEDDING_MODEL: 模型名称 (默认: BAAI/bge-m3)
    - EMBEDDING_BINDING_HOST: API 地址 (默认: https://api.siliconflow.cn/v1)
    - EMBEDDING_BINDING_API_KEY: API 密钥
    - EMBEDDING_DIM: 向量维度 (默认: 1024)
    - EMBEDDING_TOKEN_LIMIT: 最大 token 数 (默认: 8192)
    """
    return await openai_embed.func(
        texts,
        model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        base_url=os.getenv("EMBEDDING_BINDING_HOST", "https://api.siliconflow.cn/v1"),
        api_key=os.getenv("EMBEDDING_BINDING_API_KEY")
    )
