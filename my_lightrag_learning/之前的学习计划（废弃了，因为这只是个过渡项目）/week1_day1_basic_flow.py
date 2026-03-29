"""
第1周 Day1: 理解基础数据流
============================

学习目标:
1. 理解文档插入的完整流程
2. 观察4种存储的作用
3. 对比不同查询模式的结果

阅读建议:
- 运行这个脚本前，先看 lightrag/base.py 的 4 个抽象类(5分钟)
- 运行后，查看生成的 ./week1_storage/ 目录结构
- 对比不同查询模式的输出差异
"""

import asyncio
import logging
from lightrag import LightRAG, QueryParam
from utils import siliconflow_llm_complete, siliconflow_embed

# 开启详细日志，观察系统内部运行
logging.basicConfig(
    level=logging.INFO,  # 改成 DEBUG 可以看到更多细节
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def experiment_1_insert():
    """实验1: 插入文档，观察存储变化"""
    print("\n" + "="*60)
    print("实验1: 插入文档")
    print("="*60)

    # 创建 LightRAG 实例
    rag = LightRAG(
        working_dir="../week1_storage",
        workspace="learning",  # 数据隔离
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )

    # 🔴 关键步骤：必须初始化存储！
    await rag.initialize_storages()

    # 插入一段包含实体和关系的文本
    text = """
    LightRAG是一个基于图的RAG框架，由香港大学数据科学团队(HKUDS)开发。
    该框架使用知识图谱来增强检索能力，支持多种查询模式。
    LightRAG的核心优势是将实体和关系显式建模，而不是仅依赖向量检索。
    """

    print(f"\n📝 插入文档: {text[:50]}...")
    await rag.ainsert(text)

    print("\n💡 提示: 现在去查看 ./week1_storage/ 目录")
    print("   你会看到:")
    print("   - vdb_*.json          → 向量存储 (实体/关系/文本块的嵌入)")
    print("   - graph_storage_*.json → 图存储 (实体和关系的结构)")
    print("   - kv_store_*.json     → KV存储 (文本块、LLM缓存)")
    print("   - doc_status_*.json   → 文档状态 (哪些文档已处理)")

    await rag.finalize_storages()
    return rag


async def experiment_2_query_modes():
    """实验2: 对比不同查询模式"""
    print("\n" + "="*60)
    print("实验2: 对比查询模式")
    print("="*60)

    rag = LightRAG(
        working_dir="../week1_storage",
        workspace="learning",
        llm_model_func=siliconflow_llm_complete,
        embedding_func=siliconflow_embed,
    )
    await rag.initialize_storages()

    question = "LightRAG是谁开发的？"

    # 测试4种主要模式
    modes = ["naive", "local", "global", "hybrid"]

    for mode in modes:
        print(f"\n🔍 查询模式: {mode}")
        print("-" * 60)

        result = await rag.aquery(
            question,
            param=QueryParam(
                mode=mode,
                top_k=10,  # KG 检索的实体/关系数量
                chunk_top_k=5,  # 文本块检索数量
            )
        )

        print(f"回答: {result}")
        print()

        # 思考: 每种模式的回答有什么区别？
        if mode == "naive":
            print("💭 naive: 直接向量检索，不使用知识图谱")
        elif mode == "local":
            print("💭 local: 基于实体的局部检索，适合特定问题")
        elif mode == "global":
            print("💭 global: 基于全局摘要，适合概括性问题")
        elif mode == "hybrid":
            print("💭 hybrid: 结合 local 和 global")

    await rag.finalize_storages()


async def experiment_3_observe_entities():
    """实验3: 观察提取的实体和关系"""
    print("\n" + "="*60)
    print("实验3: 查看知识图谱内容")
    print("="*60)

    import networkx as nx

    # 读取 GraphML 图文件
    graph_file = "../week1_storage/learning/graph_chunk_entity_relation.graphml"

    try:
        # 加载 GraphML 格式的图
        G = nx.read_graphml(graph_file)

        print(f"\n📊 图统计信息:")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")

        print("\n📊 提取的实体 (节点):")
        for i, (node_id, node_data) in enumerate(list(G.nodes(data=True))[:5], 1):
            entity_type = node_data.get('entity_type', 'N/A')
            description = node_data.get('description', 'N/A')
            print(f"  {i}. {node_id} (类型: {entity_type})")
            if description != 'N/A':
                desc_preview = description[:60] + "..." if len(description) > 60 else description
                print(f"     描述: {desc_preview}")

        print(f"\n   总共提取了 {G.number_of_nodes()} 个实体")

        print("\n🔗 提取的关系 (边):")
        for i, (src, tgt, edge_data) in enumerate(list(G.edges(data=True))[:5], 1):
            keywords = edge_data.get('keywords', 'N/A')
            description = edge_data.get('description', 'N/A')
            print(f"  {i}. {src} -> {tgt}")
            print(f"     关键词: {keywords}")
            if description != 'N/A':
                desc_preview = description[:60] + "..." if len(description) > 60 else description
                print(f"     描述: {desc_preview}")

        print(f"\n   总共提取了 {G.number_of_edges()} 条关系")

        print("\n💡 提示: 这些实体和关系是 LLM 从文本中自动提取的")
        print("   查看 lightrag/prompt.py 可以看到提取的 prompt 模板")

    except FileNotFoundError:
        print(f"⚠️  文件未找到: {graph_file}")
        print("   请先运行 experiment_1_insert()")
    except Exception as e:
        print(f"⚠️  读取图文件时出错: {e}")


async def main():
    """主函数：按顺序运行实验"""

    print("\n" + "🚀 " + "="*56 + " 🚀")
    print("     LightRAG 第1周学习 - Day 1: 基础数据流")
    print("🚀 " + "="*56 + " 🚀\n")

    # 实验1: 插入文档
    await experiment_1_insert()

    # 实验2: 对比查询模式
    await experiment_2_query_modes()

    # 实验3: 观察提取的实体
    await experiment_3_observe_entities()

    print("\n" + "✅ " + "="*56 + " ✅")
    print("     第1天实验完成！")
    print("✅ " + "="*56 + " ✅\n")

    print("📚 接下来的学习建议:")
    print("   1. 阅读 lightrag/lightrag.py 的 ainsert() 方法 (30分钟)")
    print("   2. 阅读 lightrag/operate.py 的 extract_entities() (30分钟)")
    print("   3. 运行 week1_day2_deep_dive.py (明天)")


if __name__ == "__main__":
    asyncio.run(main())