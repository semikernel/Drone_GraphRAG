from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graph_storage import GraphStorage
from graph_entity import GraphEntity
from embedding_model import EmbeddingModel


class GraphSearch:
    """搜索管理器，处理所有与搜索相关的操作"""

    def __init__(self, storage: GraphStorage, entity_manager: GraphEntity):
        """
        初始化搜索管理器

        Args:
            storage: 存储管理器实例
            entity_manager: 实体管理器实例
        """
        self.storage = storage
        self.entity_manager = entity_manager

    def search_vector_store(
        self, query: str, entity_id: Optional[str] = None, k: int = 3
    ) -> List[Tuple[Any, float]]:
        """
        在向量存储中搜索

        Args:
            query: 搜索查询
            entity_id: 可选的实体ID限制
            k: 返回结果数量

        Returns:
            List[Tuple[Any, float]]: 搜索结果和相似度分数
        """
        try:
            # 如果指定了实体ID，在实体的向量存储中搜索
            if entity_id:
                main_id = self.entity_manager._get_main_id(entity_id)
                if not main_id or main_id not in self.storage.vector_stores:
                    return []
                vector_store = self.storage.vector_stores[main_id]
            # 否则在全局向量存储中搜索
            else:
                if not self.storage.global_vector_store:
                    return []
                vector_store = self.storage.global_vector_store

            # 生成查询向量
            if not isinstance(query, str):
                raise ValueError("查询必须是字符串")

            # 执行相似度搜索
            results = vector_store.similarity_search_with_score(query, k=k)

            # 处理和过滤结果
            valid_results = []
            for doc, score in results:
                if hasattr(doc, "page_content"):
                    valid_results.append((doc.page_content, float(score)))

            return valid_results

        except Exception as e:
            print(f"搜索向量存储时发生错误: {str(e)}")
            return []

    def search_similar_entities(
        self, query_entity: str, top_n: int = 5, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        搜索与给定实体相似的实体

        Args:
            query_entity: 查询实体
            top_n: 返回结果数量
            threshold: 相似度阈值

        Returns:
            List[Tuple[str, float]]: (实体ID, 相似度分数)列表
        """
        try:
            # 生成查询向量
            query_embedding = EmbeddingModel.get_instance().embed_query(query_entity)
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)

            # 计算相似度
            similarities = []
            for entity_id, entity_embedding in self.storage.entity_embeddings.items():
                if not isinstance(entity_embedding, np.ndarray):
                    entity_embedding = np.array(entity_embedding)
                similarity = cosine_similarity([query_embedding], [entity_embedding])[
                    0
                ][0]
                if similarity >= threshold:
                    similarities.append((entity_id, similarity))

            # 按相似度排序
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

        except Exception as e:
            print(f"搜索相似实体时发生错误: {str(e)}")
            return []

    def search_similar_relationships(
        self, query: str, entity_id: str, k: int = 3
    ) -> List[Tuple[str, str, str, float]]:
        """
        搜索与查询相似的实体关系

        Args:
            query: 搜索查询
            entity_id: 实体ID
            k: 返回结果数量

        Returns:
            List[Tuple[str, str, str, float]]: (实体1, 关系, 实体2, 分数)列表
        """
        main_id = self.entity_manager._get_main_id(entity_id)
        if not main_id:
            return []

        try:
            query_embedding = EmbeddingModel.get_instance().embed_query(query)
            results = []
            processed_entities = set()

            def process_entity_relationships(
                entity: str,
            ) -> List[Tuple[str, str, str, float]]:
                """处理单个实体的关系"""
                relations = []

                # 处理出边
                for successor in self.storage.graph.successors(entity):
                    edges_data = self.storage.graph.get_edge_data(entity, successor)
                    if edges_data:
                        for edge_data in edges_data.values():
                            relation_text = (
                                f"{entity}与{successor}的关系是{edge_data['type']}"
                            )
                            relation_embedding = (
                                EmbeddingModel.get_instance().embed_query(relation_text)
                            )
                            similarity = cosine_similarity(
                                [query_embedding], [relation_embedding]
                            )[0][0]
                            if similarity >= 0.5:
                                relations.append(
                                    (entity, edge_data["type"], successor, similarity)
                                )

                # 处理入边
                for predecessor in self.storage.graph.predecessors(entity):
                    edges_data = self.storage.graph.get_edge_data(predecessor, entity)
                    if edges_data:
                        for edge_data in edges_data.values():
                            relation_text = (
                                f"{predecessor}与{entity}的关系是{edge_data['type']}"
                            )
                            relation_embedding = (
                                EmbeddingModel.get_instance().embed_query(relation_text)
                            )
                            similarity = cosine_similarity(
                                [query_embedding], [relation_embedding]
                            )[0][0]
                            if similarity >= 0.5:
                                relations.append(
                                    (predecessor, edge_data["type"], entity, similarity)
                                )

                return relations

            # 首先处理主实体的关系
            results.extend(process_entity_relationships(main_id))
            processed_entities.add(main_id)

            # 如果结果不足k个，搜索相似实体的关系
            while len(results) < k:
                similar_entities = []
                main_embedding = self.storage.entity_embeddings[main_id]

                # 查找相似实体
                for entity, embedding in self.storage.entity_embeddings.items():
                    if entity not in processed_entities:
                        similarity = cosine_similarity([main_embedding], [embedding])[
                            0
                        ][0]
                        if similarity >= 0.8:
                            similar_entities.append((entity, similarity))

                if not similar_entities:
                    break

                # 处理相似实体
                similar_entities.sort(key=lambda x: x[1], reverse=True)
                found_new_relations = False

                for similar_entity, _ in similar_entities:
                    new_relations = process_entity_relationships(similar_entity)
                    if new_relations:
                        results.extend(new_relations)
                        found_new_relations = True
                    processed_entities.add(similar_entity)

                    if len(results) >= k:
                        break

                if not found_new_relations:
                    break

            # 按相似度排序并返回前k个结果
            return sorted(results, key=lambda x: x[3], reverse=True)[:k]

        except Exception as e:
            print(f"搜索相似关系时发生错误: {str(e)}")
            return []

    def search_all_paths(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
        max_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        搜索两个实体之间的路径，优先返回较短的路径

        Args:
            start_entity: 起始实体
            end_entity: 目标实体
            max_depth: 最大搜索深度
            max_results: 最大返回结果数量，默认为3

        Returns:
            List[Dict[str, Any]]: 按路径长度排序的路径信息列表，每个字典包含：
                - path: 路径上的实体列表
                - relationships: 路径上的关系描述列表
                - length: 路径长度
        """
        start_main_id = self.entity_manager._get_main_id(start_entity)
        end_main_id = self.entity_manager._get_main_id(end_entity)

        if not start_main_id or not end_main_id or start_main_id == end_main_id:
            return []

        # 结果列表
        all_paths = []

        # 用于BFS的队列：(当前节点, 当前路径, 关系列表, 当前深度)
        queue = deque([(start_main_id, [start_main_id], [], 0)])

        # 访问记录：(节点, 深度) -> 是否访问过
        # 同一节点在不同深度可以重复访问，但需要优先访问较短路径
        visited = set()

        while queue and len(all_paths) < max_results:
            current, current_path, relations, depth = queue.popleft()

            # 如果超过最大深度，跳过
            if depth > max_depth:
                continue

            # 当前状态标记
            state = (current, depth)
            if state in visited:
                continue
            visited.add(state)

            # 如果找到目标节点
            if current == end_main_id:
                path_info = {
                    "path": current_path,
                    "relationships": relations,
                    "length": len(current_path) - 1,
                }
                all_paths.append(path_info)
                continue

            # 获取所有相邻节点
            neighbors = []

            # 处理出边
            for successor in self.storage.graph.successors(current):
                edges_data = self.storage.graph.get_edge_data(current, successor)
                for edge_data in edges_data.values():
                    neighbors.append(("out", successor, edge_data["type"]))

            # 处理入边
            for predecessor in self.storage.graph.predecessors(current):
                edges_data = self.storage.graph.get_edge_data(predecessor, current)
                for edge_data in edges_data.values():
                    neighbors.append(("in", predecessor, edge_data["type"]))

            # 遍历所有相邻节点
            for direction, next_node, relation_type in neighbors:
                # 检查下一个状态是否访问过
                next_state = (next_node, depth + 1)
                if next_state in visited:
                    continue

                # 构建关系描述
                if direction == "out":
                    relation_desc = f"{current} -{relation_type}-> {next_node}"
                else:
                    relation_desc = f"{next_node} -{relation_type}-> {current}"

                # 构建新路径和关系列表
                new_path = current_path + [next_node]
                new_relations = relations + [relation_desc]

                # 将新状态加入队列
                queue.append((next_node, new_path, new_relations, depth + 1))

        return all_paths  # 已经按长度排序，因为使用BFS

    def tree_search(self, start_entity: str, max_depth: int = 3) -> nx.DiGraph:
        """
        从起始实体开始进行树形搜索

        Args:
            start_entity: 起始实体ID
            max_depth: 最大搜索深度

        Returns:
            nx.DiGraph: 搜索树
        """
        start_main_id = self.entity_manager._get_main_id(start_entity)
        if start_main_id:
            return nx.bfs_tree(self.storage.graph, start_main_id, depth_limit=max_depth)
        return nx.DiGraph()

    def search_communities(
        self, query: str, top_n: int = 1, threshold: float = 0.5
    ) -> List[Tuple[List[str], str]]:
        """
        根据查询搜索相关的社区,并返回社区包含的实体列表及社区简介

        Args:
            query: 用户的查询字符串
            top_n: 返回的最大社区数量
            threshold: 相似度阈值，分数需要高于此值才会返回结果（分数越高表示相似度越高）

        Returns:
            List[Tuple[List[str], str]]: 每个元组包含(社区实体列表, 社区简介)。
            当所有结果的相似度分数都低于阈值时，返回空列表。
        """
        if not self.storage.community_vector_store:
            return []

        try:
            # 进行相似性搜索
            results = self.storage.community_vector_store.similarity_search_with_score(
                query, k=top_n
            )
            communities_data = []

            for doc, score in results:
                # 只有当相似度分数高于阈值时才处理该结果
                if score > threshold:
                    # 从metadata中获取社区ID
                    community_id = doc.metadata["Community"].split("_")[
                        1
                    ]  # 从 'Community_25' 提取 '25'

                    # 从社区数据中获取相关信息
                    community_data = self.storage.communities[community_id]

                    members = community_data["members"]
                    summary = community_data["summary"]
                    communities_data.append((members, summary))

            return communities_data

        except Exception as e:
            print(f"搜索社区时发生错误: {str(e)}")
            return []
