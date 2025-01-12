import os
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx
from openai import OpenAI

from graph_storage import GraphStorage
from graph_entity import GraphEntity
from graph_search import GraphSearch
from graph_visualization import GraphVisualization
from config import API_KEY, API_BASE_URL


class KnowledgeGraph:
    """知识图谱主类，整合所有功能组件"""

    def __init__(self, base_path: str):
        """
        初始化知识图谱

        Args:
            base_path: 知识图谱数据的基础路径
        """
        # 初始化LLM客户端
        self.llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

        # 初始化各个组件
        self.storage = GraphStorage(base_path)
        self.entity = GraphEntity(self.storage, self.llm_client)
        self.search = GraphSearch(self.storage, self.entity)
        self.visualization = GraphVisualization(self.storage)

        # 加载现有图谱
        if os.path.exists(base_path):
            self._load_existing_graph()
            print(f"成功加载已存在的图谱：{base_path}")
        else:
            self._initialize_new_graph()
            print(f"创建新的图谱：{base_path}")

    def _load_existing_graph(self) -> None:
        """加载现有图谱数据"""
        try:
            self.storage.load()
        except Exception as e:
            print(f"加载图谱时发生错误: {str(e)}")
            print("初始化新的图谱...")
            self._initialize_new_graph()

    def _initialize_new_graph(self) -> None:
        """初始化新的图谱"""
        try:
            self.storage._init_storage()
        except Exception as e:
            print(f"初始化时发生错误: {str(e)}")

    def save(self) -> None:
        """保存图谱数据"""
        try:
            self.storage.save()
            print("图谱保存成功")
        except Exception as e:
            print(f"保存图谱时发生错误: {str(e)}")

    # 实体管理方法
    def add_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> str:
        """
        添加实体到图谱

        Args:
            entity_id: 实体ID
            content_units: [(title, content),...] 格式的内容单元列表

        Returns:
            str: 实体主ID（可能是合并后的ID）
        """
        return self.entity.add_entity(entity_id, content_units)

    def add_relationship(
        self, entity1_id: str, entity2_id: str, relationship_type: str
    ) -> None:
        """添加实体间关系"""
        self.entity.add_relationship(entity1_id, entity2_id, relationship_type)

    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """获取实体信息"""
        return self.entity.get_entity_info(entity_id)

    def get_relationships(self, entity1_id: str, entity2_id: str) -> List[str]:
        """获取两个实体间的所有关系"""
        return self.entity.get_relationships(entity1_id, entity2_id)

    def get_related_entities(self, entity_id: str) -> List[str]:
        """获取与指定实体相关的所有实体"""
        return self.entity.get_related_entities(entity_id)

    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """合并两个实体"""
        return self.entity.merge_entities(entity_id1, entity_id2)

    # 搜索方法
    def search_similar_entities(
        self, query_entity: str, top_n: int = 5, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """搜索相似实体"""
        return self.search.search_similar_entities(query_entity, top_n, threshold)

    def search_vector_store(
        self, query: str, entity_id: Optional[str] = None, k: int = 3
    ) -> List[Tuple[str, float]]:
        """搜索向量存储"""
        return self.search.search_vector_store(query, entity_id, k)

    def search_similar_relationships(
        self, query: str, entity_id: str, k: int = 3
    ) -> List[Tuple[str, str, str, float]]:
        """搜索相似关系"""
        return self.search.search_similar_relationships(query, entity_id, k)

    def search_all_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
        max_results: int = 3,
    ) -> List[Dict]:
        """搜索实体间所有路径"""
        return self.search.search_all_paths(
            start_entity, end_entity, max_depth, max_results
        )

    def search_communities(
        self, query: str, top_n: int = 1
    ) -> List[Tuple[List[str], str]]:
        """根据查询搜索相关社区"""
        return self.search.search_communities(query, top_n)

    def tree_search(self, start_entity: str, max_depth: int = 3) -> nx.DiGraph:
        """树形搜索"""
        return self.search.tree_search(start_entity, max_depth)

    # 社区发现
    def detect_communities(
        self, resolution: float = 1.0, min_community_size: int = 4
    ) -> List[List[str]]:
        """检测社区"""
        return self.entity.detect_communities(resolution, min_community_size)

    # 图谱操作
    def merge_graphs(self, other_graph: "KnowledgeGraph") -> None:
        """合并其他图谱"""
        self.entity.merge_graphs(other_graph.entity)

    def merge_similar_entities(self) -> None:
        """合并相似实体"""
        self.entity.merge_similar_entities()

    def remove_duplicates(self) -> None:
        """移除重复内容"""
        self.entity.remove_duplicates_and_self_loops()

    # 可视化方法
    def visualize(self) -> None:
        """创建常规图谱可视化"""
        self.visualization.visualize()

    def visualize_communities(self) -> None:
        """创建社区视图的图谱可视化"""
        self.visualization.visualize_communities()

    def get_statistics(self) -> Dict[str, int]:
        """获取图谱统计信息"""
        return {
            "实体数量": self.storage.get_entity_count(),
            "关系数量": self.storage.get_relationship_count(),
            "别名数量": self.storage.get_alias_count(),
            "向量存储数量": self.storage.get_store_count(),
        }

    def generate_statistics(self, save_path: Optional[str] = None) -> str:
        """
        生成并保存图谱统计信息

        Args:
            save_path: 可选的保存路径。如果不提供，将使用默认路径

        Returns:
            str: 保存的文件路径
        """
        try:
            return self.visualization.generate_statistics(save_path)
        except Exception as e:
            print(f"生成统计信息时发生错误: {str(e)}")
            return ""

    def cleanup(self) -> None:
        """清理资源"""
        self.storage.cleanup()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源被正确释放"""
        self.cleanup()
