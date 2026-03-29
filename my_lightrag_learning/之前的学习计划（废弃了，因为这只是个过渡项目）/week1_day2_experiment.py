"""
第1周 Day2: 验证 lightrag.py 的理解
====================================

实验目标:
1. 理解 initialize_storages() 的必要性
2. 对比不同配置参数的影响
3. 观察 workspace 数据隔离

使用环境变量配置（兼容硅基流动 API）
"""

import asyncio
from lightrag import LightRAG, QueryParam
from utils import siliconflow_llm_complete, siliconflow_embed


async def exp1_storage_init():
    """实验1: 不初始化存储会怎样？"""
    print("\n" + "="*60)
    print("实验1: 存储初始化")
    print("="*60)

    rag = LightRAG(
        working_dir="./day2_storage",
        workspace="exp1",
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )

    print("\n❌ 不调用 initialize_storages():")
    try:
        await rag.ainsert("测试")
    except (AttributeError, RuntimeError, TypeError) as e:
        print(f"   错误: {type(e).__name__}")
        print(f"   → 原因: 存储实例不存在")

    print("\n✅ 正确流程:")
    await rag.initialize_storages()
    await rag.ainsert("测试文档")
    print("   → 成功插入")
    await rag.finalize_storages()


async def exp2_chunk_size():
    """实验2: chunk_token_size 的影响"""
    print("\n" + "="*60)
    print("实验2: chunk_token_size 参数")
    print("="*60)

    text = "人工智能是计算机科学的分支。" * 20  # 重复20次

    for size in [200, 800]:
        print(f"\n🔧 chunk_token_size={size}:")
        rag = LightRAG(
            working_dir="./day2_storage",
            workspace=f"exp2_chunk{size}",
            chunk_token_size=size,
            llm_model_func=siliconflow_llm_complete,
            embedding_func=siliconflow_embed,
        )
        await rag.initialize_storages()
        await rag.ainsert(text)

        # 查看生成的文本块数量
        chunks = await rag.key_string_value_json_storage.get_by_id("text_chunks")
        print(f"   生成 {len(chunks) if chunks else 0} 个文本块")

        await rag.finalize_storages()


async def exp3_workspace():
    """实验3: workspace 数据隔离"""
    print("\n" + "="*60)
    print("实验3: Workspace 隔离")
    print("="*60)

    # 两个独立的 workspace
    rag_a = LightRAG(
        working_dir="./day2_storage",
        workspace="project_a",
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )

    rag_b = LightRAG(
        working_dir="./day2_storage",
        workspace="project_b",
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )

    await rag_a.initialize_storages()
    await rag_b.initialize_storages()

    print("\n📚 Project A 插入: AI 相关")
    await rag_a.ainsert("人工智能是计算机科学分支")

    print("📚 Project B 插入: RAG 相关")
    await rag_b.ainsert("RAG 结合检索和生成")

    print("\n🔍 在 A 中查询 'RAG':")
    result = await rag_a.aquery("什么是 RAG？")
    if "no-context" in (result or ""):
        print("   ✅ A 找不到 RAG（正确隔离）")
    else:
        print(f"   {result[:50]}...")

    print("\n🔍 在 B 中查询 'RAG':")
    result = await rag_b.aquery("什么是 RAG？")
    print(f"   {result[:50] if result else 'None'}...")

    await rag_a.finalize_storages()
    await rag_b.finalize_storages()


async def exp4_insert_flow():
    """实验4: 观察插入流程"""
    print("\n" + "="*60)
    print("实验4: 插入流程观察")
    print("="*60)

    rag = LightRAG(
        working_dir="./day2_storage",
        workspace="exp4",
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )
    await rag.initialize_storages()

    docs = ["LightRAG 使用图增强检索", "它由香港大学开发"]
    doc_ids = ["doc-001", "doc-002"]

    print("\n📝 插入2个文档...")
    await rag.ainsert(docs, ids=doc_ids)

    print("\n📊 检查存储:")
    full_docs = await rag.key_string_value_json_storage.get_by_id("full_docs")
    print(f"   KV: {len(full_docs) if full_docs else 0} 个文档")

    for doc_id in doc_ids:
        status = await rag.doc_status_storage.get_by_id(doc_id)
        if status:
            print(f"   DocStatus: {doc_id} → {status.get('status', 'unknown')}")

    await rag.finalize_storages()


async def main():
    print("\n🚀 Day 2 实验：lightrag.py 核心类\n")

    await exp1_storage_init()
    await exp2_chunk_size()
    await exp3_workspace()
    await exp4_insert_flow()

    print("\n✅ 实验完成！\n")
    print("💡 接下来:")
    print("   1. 完成 learning_checklist.md Day 2 检查项")
    print("   2. 在另一个对话中深入读 lightrag.py 代码")
    print("   3. 明天学习 operate.py")


if __name__ == "__main__":
    asyncio.run(main())