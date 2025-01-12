from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import re
from knowledgeGraph import KnowledgeGraph


class RetrievalMode(Enum):
    FAST = "1"  # 快速检索：直接检索最相关的内容
    ASSOCIATE = "2"  # 联想检索：基于初始检索结果进行联想
    RELATION = "3"  # 关系检索：关注实体间的关系网络
    COMMUNITY = "4"  # 社区检索：检索社区内的相关讨论


class KnowledgeRetriever:
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        """初始化知识检索服务"""
        self.retrieval_cache: Dict[str, int] = {}  # 检索结果缓存
        self.initial_cache_rounds = 5  # 缓存初始轮数

        try:
            print(f"\n[Info] 正在加载知识图谱...")
            self.kg = KnowledgeGraph(knowledge_base_path)
            print(f"[Info] 知识图谱加载成功！")
        except Exception as e:
            print(f"加载知识图谱时出错: {str(e)}")
            self.kg = None

    def _get_cached_results(self, results: List[Tuple[Any, float]]) -> List[str]:
        """处理检索结果的缓存逻辑
        返回未缓存的第一个内容，如果所有内容都在缓存中则返回第一个内容
        """
        if not results:
            return []

        # 处理第一个结果
        first_content = results[0][0]
        content = (
            first_content
            if isinstance(first_content, str)
            else first_content.page_content
        )

        # 如果第一个内容不在缓存中，直接返回
        if content not in self.retrieval_cache:
            self.retrieval_cache[content] = self.initial_cache_rounds
            return [content]

        # 第一个内容在缓存中，增加计数
        self.retrieval_cache[content] += 1

        # 遍历剩余结果，寻找未缓存的内容
        for doc, _ in results[1:]:
            next_content = doc if isinstance(doc, str) else doc.page_content
            if next_content not in self.retrieval_cache:
                # 找到未缓存内容，添加到缓存并返回
                self.retrieval_cache[next_content] = self.initial_cache_rounds
                return [next_content]
            else:
                # 内容在缓存中，增加计数
                self.retrieval_cache[next_content] += 1

        # 所有内容都在缓存中，返回第一个
        return [content]

    def _cleanup_cache(self):
        """清理缓存"""
        expired_entries = [
            content for content, rounds in self.retrieval_cache.items() if rounds <= 0
        ]

        for content in expired_entries:
            del self.retrieval_cache[content]

    def update_cache_counts(self):
        """更新缓存计数"""
        for content in list(self.retrieval_cache.keys()):
            self.retrieval_cache[content] -= 1
        self._cleanup_cache()

    def parse_query_response(self, response: str) -> Dict:
        """解析LLM返回的检索信息"""
        default_response = {"query": "", "entities": [], "reply": ""}

        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            json_str = json_match.group(1) if json_match else response
            query_info = json.loads(json_str)

            if all(key in query_info for key in ["query", "entities"]):
                return query_info
            else:
                print("[Info] 缺少必要字段")
                return default_response

        except Exception as e:
            print(f"[Info] 解析出错: {str(e)}")
            return default_response

    def fast_retrieval(self, query: str) -> Optional[str]:
        """快速检索：直接在向量存储中搜索"""
        try:
            results = self.kg.search_vector_store(query, k=5)
            filtered_results = [(doc, score) for doc, score in results if score >= 0.55]

            if filtered_results:
                selected_contents = self._get_cached_results(filtered_results)
                return "\n\n".join(selected_contents)

        except Exception as e:
            print(f"快速检索时发生错误: {str(e)}")
        return None

    def associate_retrieval(self, query: str, entities: List[str]) -> Optional[str]:
        """
        联想检索：基于实体和关系网络进行扩展检索
        - 如果在指定实体向量库中找到结果，继续进行关联搜索
        - 如果在指定实体向量库中没有结果，则在全局向量库中搜索
        """
        try:
            retrieval_results = []
            retrieved_contents = set()  # 用于记录已检索的内容，避免重复
            found_in_entity_store = False  # 标记是否在实体向量库中找到结果

            for entity in entities:
                # 查找相似实体
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.8
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # 在主实体的向量存储中搜索
                entity_results = self.kg.search_vector_store(
                    query, entity_id=main_entity, k=5
                )
                filtered_results = [
                    (doc, score) for doc, score in entity_results if score >= 0.5
                ]

                if filtered_results:
                    found_in_entity_store = True
                    selected_contents = self._get_cached_results(filtered_results)
                    for content in selected_contents:
                        # 检查内容是否已存在
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(
                                f"[{main_entity}]相关内容：\n{content}"
                            )

            # 如果在实体向量库中没有找到内容，进行全局搜索并返回
            if not found_in_entity_store:
                global_results = self.kg.search_vector_store(query, k=5)
                filtered_global_results = [
                    (doc, score) for doc, score in global_results if score >= 0.5
                ]

                if filtered_global_results:
                    selected_contents = self._get_cached_results(
                        filtered_global_results
                    )
                    for content in selected_contents:
                        if content not in retrieved_contents:
                            retrieved_contents.add(content)
                            retrieval_results.append(f"[全局搜索]相关内容：\n{content}")

                return "\n\n".join(retrieval_results) if retrieval_results else None

            # 如果在实体向量库中找到了结果，继续进行关联实体搜索
            for entity in entities:
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if not similar_entities:
                    continue

                main_entity = similar_entities[0][0]

                # 获取相关关系
                relationships = self.kg.search_similar_relationships(
                    query, main_entity, k=3
                )
                if not relationships:
                    continue

                # 在关联实体中搜索
                related_entities = set()  # 记录关联实体
                relations_added = set()  # 记录已添加的关系描述
                entity_relations = {}  # 记录实体对应的关系描述

                for source, relation, target, score in relationships:
                    # 构建完整的关系查询语句
                    relation_query = f"{source} 与 {target} 的关系是：{relation}"

                    # 记录关系信息
                    if relation_query not in relations_added:
                        relations_added.add(relation_query)
                        retrieval_results.append(f"[关联关系]：\n- {relation_query}")

                    # 将关系中的实体加入集合（排除主实体）并记录对应的关系描述
                    if source != main_entity:
                        related_entities.add(source)
                        entity_relations[source] = relation_query
                    if target != main_entity:
                        related_entities.add(target)
                        entity_relations[target] = relation_query

                # 在关联实体中搜索，使用关系查询语句
                for related_entity in related_entities:
                    relation_query = entity_relations[related_entity]
                    results = self.kg.search_vector_store(
                        query=relation_query, entity_id=related_entity, k=5
                    )
                    filtered_results = [
                        (doc, score) for doc, score in results if score >= 0.5
                    ]

                    if filtered_results:
                        selected_contents = self._get_cached_results(filtered_results)
                        for content in selected_contents:
                            # 检查内容是否与已有内容重复
                            content_is_unique = True
                            normalized_content = "".join(content.split())

                            for existing_content in retrieved_contents:
                                normalized_existing = "".join(existing_content.split())
                                if (
                                    normalized_content in normalized_existing
                                    or normalized_existing in normalized_content
                                ):
                                    content_is_unique = False
                                    break

                            if content_is_unique:
                                retrieved_contents.add(content)
                                retrieval_results.append(
                                    f"[关联实体 - {related_entity}]：\n{content}"
                                )

            return "\n\n".join(retrieval_results) if retrieval_results else None

        except Exception as e:
            print(f"联想检索时发生错误: {str(e)}")
        return None

    def relation_retrieval(self, entities: List[str]) -> Optional[str]:
        """
        检索实体列表中每对实体之间的路径关系，对第一条路径进行向量检索
        """
        try:
            if len(entities) < 2:
                return None

            # 1. 实体匹配
            main_entities = []
            matched_indices = []  # 记录成功匹配的原始实体索引

            for i, entity in enumerate(entities):
                similar_entities = self.kg.search_similar_entities(
                    entity, top_n=1, threshold=0.85
                )
                if similar_entities:
                    main_entities.append(similar_entities[0][0])
                    matched_indices.append(i)

            if len(main_entities) < 2:
                return None

            result_parts = []
            seen_paths = set()  # 用于去重路径
            seen_contents = set()  # 用于去重内容

            # 2. 对匹配成功的实体对进行路径搜索
            for i in range(len(main_entities)):
                for j in range(i + 1, len(main_entities)):
                    # 使用匹配后的实体和对应的原始实体索引
                    entity1 = main_entities[i]
                    entity2 = main_entities[j]
                    original_entity1 = entities[matched_indices[i]]
                    original_entity2 = entities[matched_indices[j]]

                    # 避免重复路径
                    path_key = f"{min(entity1, entity2)}-{max(entity1, entity2)}"
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    # 搜索两个实体之间的所有路径
                    paths = self.kg.search_all_paths(entity1, entity2, max_depth=5)

                    if paths:
                        result_parts.append(
                            f"\n{original_entity1} - {original_entity2} 的关系:"
                        )

                        # 显示所有路径
                        for path_idx, path_info in enumerate(paths, 1):
                            result_parts.append(f"路径 {path_idx}:")
                            result_parts.append(
                                f"实体路径: {' -> '.join(path_info['path'])}"
                            )
                            result_parts.append("关系链:")
                            result_parts.extend(
                                f"  {rel}" for rel in path_info["relationships"]
                            )

                            # 只对第一条路径进行向量检索
                            if path_idx == 1:
                                for relationship in path_info["relationships"]:
                                    try:
                                        start_end = relationship.split("->")
                                        if len(start_end) == 2:
                                            start_part = start_end[0].strip()
                                            end_entity = start_end[1].strip()

                                            start_relation = start_part.split("-", 1)
                                            if len(start_relation) == 2:
                                                start_entity = start_relation[0].strip()
                                                relation = start_relation[1].strip()

                                                relation_query = f"{start_entity} 与 {end_entity} 的关系是：{relation}"

                                                entity_results = (
                                                    self.kg.search_vector_store(
                                                        query=relation_query,
                                                        entity_id=start_entity,
                                                        k=3,
                                                    )
                                                )

                                                filtered_results = [
                                                    (doc, score)
                                                    for doc, score in entity_results
                                                    if score >= 0.5
                                                ]
                                                if filtered_results:
                                                    selected_contents = (
                                                        self._get_cached_results(
                                                            filtered_results
                                                        )
                                                    )
                                                    for content in selected_contents:
                                                        normalized_content = "".join(
                                                            content.split()
                                                        )
                                                        if (
                                                            normalized_content
                                                            not in seen_contents
                                                        ):
                                                            seen_contents.add(
                                                                normalized_content
                                                            )
                                                            result_parts.append(
                                                                f"[{start_entity}->{end_entity}]相关内容：\n{content}\n\n"
                                                            )

                                    except Exception as e:
                                        print(
                                            f"处理关系 '{relationship}' 时发生错误: {str(e)}"
                                        )
                                        continue

                        result_parts.append("-" * 50)  # 添加分隔线

            return "\n".join(result_parts) if result_parts else None

        except Exception as e:
            print(f"关系检索时发生错误: {str(e)}")
        return None

    def community_retrieval(self, query: str) -> Optional[str]:
        """
        社区检索：查找相关的社区信息和全局文档中的相关表述

        Args:
            query: 用户的查询字符串

        Returns:
            Optional[str]: 返回检索结果，包括社区信息和相关表述。如果没有找到相关内容则返回 None
        """
        try:
            result_parts = []

            # 1. 社区检索
            community_results = self.kg.search_communities(query, top_n=1)
            if community_results:
                members, summary = community_results[0]
                result_parts.append("【请参考社区观点】")
                result_parts.append("相关社区成员:")
                result_parts.append(f"- {', '.join(members)}")
                result_parts.append("\n社区简介:")
                result_parts.append(summary)

            # 2. 全局文档检索
            doc_results = self.kg.search_vector_store(query, k=5)
            filtered_results = [
                (doc, score) for doc, score in doc_results if score >= 0.5
            ]

            if filtered_results:
                selected_contents = self._get_cached_results(filtered_results)
                if selected_contents:
                    if result_parts:  # 如果前面有社区结果，加一个分隔
                        result_parts.append("\n")
                    result_parts.append(
                        "【以下信息仅为表达风格参考，回答中请使用上述社区观点】"
                    )
                    result_parts.extend(selected_contents)

            # 只要有任一种检索结果就返回
            if result_parts:
                return "\n".join(result_parts)

        except Exception as e:
            print(f"社区检索时发生错误: {str(e)}")
        return None

    def retrieve(
        self, mode: RetrievalMode, query: str, entities: List[str]
    ) -> Optional[str]:
        """统一的检索接口"""
        if not self.kg:
            return None

        self.update_cache_counts()

        try:
            if mode == RetrievalMode.FAST:
                return self.fast_retrieval(query)

            elif mode == RetrievalMode.ASSOCIATE:
                return self.associate_retrieval(query, entities)

            elif mode == RetrievalMode.RELATION:
                if len(entities) >= 2:
                    return self.relation_retrieval(entities)

            elif mode == RetrievalMode.COMMUNITY:
                return self.community_retrieval(query)

        except Exception as e:
            print(f"检索时发生错误: {str(e)}")

        return None
