from typing import Optional 
"""
Optional[X] 等价于 Union[X, None]，即某个变量要么是类型 X，要么是 None。
"""
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

"""
这是RAG 中重要的一步:embediding, 将文本转化为向量
"""
class EmbeddingModel:
    _instance: Optional[HuggingFaceBgeEmbeddings] = None 
    """
    这里创建了一个单例模式，即只创建一个嵌入模型实例，避免重复创建
    """

    @classmethod
    def get_instance(cls) -> HuggingFaceBgeEmbeddings:
        """获取嵌入模型单例"""
        if cls._instance is None:
            print("初始化嵌入模型...")
            cls._instance = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                query_instruction="",
            )
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        """清理嵌入模型（如果需要）"""
        cls._instance = None
