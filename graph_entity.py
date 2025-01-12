from typing import List, Tuple, Dict, Optional, Any
from openai import OpenAI
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from graph_storage import GraphStorage
from embedding_model import EmbeddingModel

ENTITY_MERGE_PROMPT = "prompt/entity_merge.txt"
RELATIONSHIP_MERGE_PROMPT = "prompt/relationship_merge.txt"
COMMUNITY_SUMMARY_PROMPT = "prompt/community_summary.txt"


class GraphEntity:
    """实体管理器，处理所有与实体和关系相关的操作"""

    def __init__(self, storage: GraphStorage, llm_client: OpenAI):
        """
        初始化实体管理器

        Args:
            storage: 存储管理器实例
            llm_client: LLM客户端实例，用于实体合并判断
        """
        self.storage = storage
        self.llm_client = llm_client

    def add_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> str:
        """
        添加实体到图谱，如果存在相似实体则进行合并判断

        Args:
            entity_id: 实体ID
            content_units: [(title, content),...] 格式的内容单元列表

        Returns:
            str: 实体的主ID（可能是合并后的ID）
        """
        # 检查是否已存在
        main_id = self._get_main_id(entity_id)
        if main_id:
            print(f"发现已存在实体 '{entity_id}'，正在与主实体 '{main_id}' 合并...")
            self._merge_entity_content(main_id, content_units)
            return main_id

        # 生成新实体的嵌入
        new_embedding = EmbeddingModel.get_instance().embed_query(entity_id)

        # 检查相似实体
        for existing_id, existing_embedding in self.storage.entity_embeddings.items():
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            if similarity > 0.85:
                print(
                    f"发现高相似度实体：'{entity_id}' 与 '{existing_id}' 的相似度为 {similarity:.3f}"
                )
                should_merge = self._llm_merge_judgment(entity_id, existing_id)
                if should_merge:
                    print(
                        f"大模型判定可以合并，正在将 '{entity_id}' 合并到 '{existing_id}'..."
                    )
                    self._merge_entity_content(existing_id, content_units)
                    self._add_alias(existing_id, entity_id)
                    return existing_id

        # 添加为新实体
        print(f"添加新实体：'{entity_id}'")
        self.storage.graph.add_node(entity_id)
        self.storage.entity_embeddings[entity_id] = new_embedding
        self.storage.alias_to_main_id[entity_id] = entity_id
        self.storage.save_entity(entity_id, content_units)

        return entity_id

    def add_relationship(
        self, entity1_id: str, entity2_id: str, relationship_type: str
    ) -> None:
        """
        添加实体间的关系，与最相似的同向关系合并
        当相似度超过0.95时，保留原有关系不做修改
        当相似度在0.85-0.95之间时，进行关系合并
        当相似度低于0.85时，添加新关系

        Args:
            entity1_id: 起始实体ID
            entity2_id: 目标实体ID
            relationship_type: 关系类型
        """
        main_id1 = self._get_main_id(entity1_id)
        main_id2 = self._get_main_id(entity2_id)

        if not (main_id1 and main_id2):
            if not main_id1:
                print(f"实体 '{entity1_id}' 不存在")
            if not main_id2:
                print(f"实体 '{entity2_id}' 不存在")
            return

        # 获取这对实体间的现有同向关系
        existing_relationships = [
            (d["type"], k)
            for u, v, k, d in self.storage.graph.edges(data=True, keys=True)
            if u == main_id1 and v == main_id2  # 只获取同向的关系
        ]

        # 如果没有现有关系，直接添加
        if not existing_relationships:
            self.storage.graph.add_edge(main_id1, main_id2, type=relationship_type)
            print(f"添加新关系: {main_id1} -{relationship_type}-> {main_id2}")
            return

        # 获取所有关系的嵌入向量（包括新关系）
        new_embedding = EmbeddingModel.get_instance().embed_query(relationship_type)
        rel_embeddings = [
            (rel, key, EmbeddingModel.get_instance().embed_query(rel))
            for rel, key in existing_relationships
        ]

        # 计算与所有现有关系的相似度
        similarities = []
        for rel, key, embedding in rel_embeddings:
            similarity = cosine_similarity([new_embedding], [embedding])[0][0]
            similarities.append((similarity, rel, key))

        # 找出最相似的关系
        if similarities:
            max_similarity, most_similar_rel, edge_key = max(
                similarities, key=lambda x: x[0]
            )

            print(f"发现最相似关系：'{relationship_type}' 与 '{most_similar_rel}'")
            print(f"相似度为：{max_similarity:.3f}")

            # 如果相似度超过0.95，保留原有关系
            if max_similarity > 0.95:
                print(f"相似度超过0.95，保留原有关系：'{most_similar_rel}'")
                return

            # 如果相似度在0.85-0.95之间，进行合并
            elif max_similarity > 0.85:
                merged_relation = self._llm_merge_relationships(
                    main_id1, main_id2, relationship_type, most_similar_rel
                )
                print(f"合并关系为：'{merged_relation}'")

                # 更新关系
                self.storage.graph.remove_edge(main_id1, main_id2, edge_key)
                self.storage.graph.add_edge(main_id1, main_id2, type=merged_relation)
                return

        # 如果没有相似关系或相似度较低，添加新关系
        self.storage.graph.add_edge(main_id1, main_id2, type=relationship_type)
        print(f"添加新关系: {main_id1} -{relationship_type}-> {main_id2}")

    def get_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实体的详细信息

        Args:
            entity_id: 实体ID

        Returns:
            Optional[Dict[str, Any]]: 实体信息字典，包含主ID、内容和别名
        """
        main_id = self._get_main_id(entity_id)
        if not main_id:
            return None

        content = self.storage.load_entity(main_id)
        aliases = list(self.storage.entity_aliases.get(main_id, []))

        return {"main_id": main_id, "content": content, "aliases": aliases}

    def get_relationships(self, entity1_id: str, entity2_id: str) -> List[str]:
        """
        获取两个实体间的所有关系

        Args:
            entity1_id: 第一个实体ID
            entity2_id: 第二个实体ID

        Returns:
            List[str]: 关系类型列表
        """
        main_id1 = self._get_main_id(entity1_id)
        main_id2 = self._get_main_id(entity2_id)

        if main_id1 and main_id2:
            return [
                d["type"]
                for u, v, d in self.storage.graph.edges(data=True)
                if u == main_id1 and v == main_id2
            ]
        return []

    def get_related_entities(self, entity_id: str) -> List[str]:
        """
        获取与指定实体相关的所有实体

        Args:
            entity_id: 实体ID

        Returns:
            List[str]: 相关实体ID列表
        """
        main_id = self._get_main_id(entity_id)
        if main_id:
            successors = list(self.storage.graph.successors(main_id))
            predecessors = list(self.storage.graph.predecessors(main_id))
            return list(set(successors + predecessors))
        return []

    def merge_entities(self, entity_id1: str, entity_id2: str) -> str:
        """
        手动合并两个实体

        Args:
            entity_id1: 第一个实体ID
            entity_id2: 第二个实体ID

        Returns:
            str: 合并后的主实体ID
        """
        main_id1 = self._get_main_id(entity_id1)
        main_id2 = self._get_main_id(entity_id2)

        if not (main_id1 and main_id2):
            if not main_id1:
                print(f"实体 '{entity_id1}' 不存在")
            if not main_id2:
                print(f"实体 '{entity_id2}' 不存在")
            return ""

        # 选择保留ID较短的实体作为主实体
        main_entity = main_id1 if len(main_id1) <= len(main_id2) else main_id2
        merged_entity = main_id2 if main_entity == main_id1 else main_id1

        # 合并内容
        merged_content = self.storage.load_entity(merged_entity)
        self._merge_entity_content(main_entity, merged_content)

        # 合并关系
        self._merge_entity_relationships(main_entity, merged_entity)

        # 合并别名
        self._merge_entity_aliases(main_entity, merged_entity)

        # 删除被合并的实体
        self._remove_entity(merged_entity)

        return main_entity

    def merge_similar_entities(self) -> None:
        """自动检查并合并相似实体"""
        print("\n开始检查和合并相似实体...")

        # 获取所有实体对的相似度
        entity_pairs = []
        entities = list(self.storage.graph.nodes())

        for i, entity1 in enumerate(entities):
            embedding1 = self.storage.entity_embeddings[entity1]
            for entity2 in entities[i + 1 :]:
                embedding2 = self.storage.entity_embeddings[entity2]
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                if similarity > 0.85:
                    if self._llm_merge_judgment(entity1, entity2):
                        entity_pairs.append((entity1, entity2, similarity))

        # 按相似度排序
        entity_pairs.sort(key=lambda x: x[2], reverse=True)

        # 执行合并
        merged_entities = set()
        for entity1, entity2, similarity in entity_pairs:
            if entity1 not in merged_entities and entity2 not in merged_entities:
                print(f"\n合并实体：{entity1} 和 {entity2}（相似度：{similarity:.3f}）")
                merged_id = self.merge_entities(entity1, entity2)
                merged_entities.add(entity2 if merged_id == entity1 else entity1)

    def _get_main_id(self, entity_id: str) -> Optional[str]:
        """获取实体的主ID"""
        if entity_id in self.storage.alias_to_main_id:
            return self.storage.alias_to_main_id[entity_id]
        if entity_id in self.storage.graph.nodes():
            return entity_id
        return None

    def _merge_entity_content(
        self, main_id: str, content_units: List[Tuple[str, str]]
    ) -> None:
        """合并实体内容"""
        existing_content = self.storage.load_entity(main_id)

        # 使用集合去重
        existing_set = {
            (title.strip(), content.strip()) for title, content in existing_content
        }
        new_set = {(title.strip(), content.strip()) for title, content in content_units}
        merged_set = existing_set.union(new_set)

        # 保存合并后的内容
        self.storage.save_entity(main_id, list(merged_set))

    def _merge_entity_relationships(self, main_id: str, merged_id: str) -> None:
        """合并实体的关系"""
        # 处理入边
        for predecessor in self.storage.graph.predecessors(merged_id):
            if predecessor != main_id:  # 避免自环
                edges_data = self.storage.graph.get_edge_data(predecessor, merged_id)
                for edge_data in edges_data.values():
                    # 避免添加自环
                    if predecessor != main_id:
                        self.add_relationship(predecessor, main_id, edge_data["type"])

        # 处理出边
        for successor in self.storage.graph.successors(merged_id):
            if successor != main_id:  # 避免自环
                edges_data = self.storage.graph.get_edge_data(merged_id, successor)
                for edge_data in edges_data.values():
                    # 避免添加自环
                    if successor != main_id:
                        self.add_relationship(main_id, successor, edge_data["type"])

    def _merge_entity_aliases(self, main_id: str, merged_id: str) -> None:
        """合并实体的别名"""
        if merged_id in self.storage.entity_aliases:
            for alias in self.storage.entity_aliases[merged_id]:
                self._add_alias(main_id, alias)
            del self.storage.entity_aliases[merged_id]
        self._add_alias(main_id, merged_id)

    def _add_alias(self, main_id: str, alias: str) -> None:
        """添加别名"""
        if main_id not in self.storage.entity_aliases:
            self.storage.entity_aliases[main_id] = set()
        self.storage.entity_aliases[main_id].add(alias)
        self.storage.alias_to_main_id[alias] = main_id

    def _remove_entity(self, entity_id: str) -> None:
        """删除实体"""
        self.storage.graph.remove_node(entity_id)
        if entity_id in self.storage.entity_embeddings:
            del self.storage.entity_embeddings[entity_id]

    def _llm_merge_relationships(
        self, entity1: str, entity2: str, rel1: str, rel2: str
    ) -> str:
        """
        使用LLM合并两个关系描述，考虑实体上下文

        Args:
            entity1: 起始实体
            entity2: 目标实体
            rel1: 第一个关系类型
            rel2: 第二个关系类型

        Returns:
            str: 合并后的关系描述
        """
        try:
            with open(RELATIONSHIP_MERGE_PROMPT, "r", encoding="utf-8") as file:
                template = file.read()

            prompt = template.format(
                entity1=entity1, entity2=entity2, rel1=rel1, rel2=rel2
            )

            messages = [{"role": "user", "content": prompt}]

            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-8k", messages=messages, temperature=0.5
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM合并关系时发生错误: {str(e)}")
            return rel1  # 出错时保留第一个关系

    def _llm_merge_judgment(self, entity1: str, entity2: str) -> bool:
        """使用LLM判断两个实体是否应该合并"""
        try:
            with open(ENTITY_MERGE_PROMPT, "r", encoding="utf-8") as file:
                template = file.read()

            prompt = template.format(entity1=entity1, entity2=entity2)
            messages = [{"role": "user", "content": prompt}]

            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-8k", messages=messages, temperature=0.5
            )

            result = response.choices[0].message.content.strip().lower()
            return result == "yes"

        except Exception as e:
            print(f"LLM判断发生错误: {str(e)}")
            return False

    def remove_duplicates_and_self_loops(self) -> None:
        """移除重复边和自循环(包括别名)"""
        changes_made = False  # 追踪是否有任何改动

        # 移除直接自循环
        for u, v, data in list(nx.selfloop_edges(self.storage.graph, data=True)):
            print(f"移除自循环边: {u} -> {v}, 关系类型: {data.get('type')}")
            self.storage.graph.remove_edge(u, v)
            changes_made = True

        # 移除别名导致的自循环
        for source, target, data in list(self.storage.graph.edges(data=True)):
            source_main = self.storage.alias_to_main_id.get(source, source)
            target_main = self.storage.alias_to_main_id.get(target, target)

            if source_main == target_main:
                print(
                    f"移除别名自循环边: {source} -> {target}, 关系类型: {data.get('type')}, 主实体: {source_main}"
                )
                self.storage.graph.remove_edge(source, target)
                changes_made = True

        # 移除重复边
        edges_to_remove = []
        for u, v, keys, data in self.storage.graph.edges(keys=True, data=True):
            edge_type = data.get("type")
            edge_data = self.storage.graph.get_edge_data(u, v)

            if edge_data:
                existing_edges = [
                    (k, d)
                    for k, d in edge_data.items()
                    if k != keys and d.get("type") == edge_type
                ]
                for k, _ in existing_edges:
                    edges_to_remove.append((u, v, k))
                    print(f"移除重复边: {u} -> {v}, 关系类型: {edge_type}")

        for edge in edges_to_remove:
            self.storage.graph.remove_edge(*edge)
            changes_made = True

        # 如果有任何改动，保存更新后的图谱
        if changes_made:
            print("检测到图谱改动，正在保存更新...")
            self.storage.save()
            print("图谱已更新并保存")
        else:
            print("未发现重复边或自循环，无需更新")

    def merge_graphs(self, other_entity: "GraphEntity") -> None:
        """
        将另一个图谱的实体和关系合并到当前图谱

        Args:
            other_entity: 要合并的图谱的实体管理器实例
        """
        print("开始合并图谱...")

        # 1. 合并节点和内容
        for node in other_entity.storage.graph.nodes():
            print(f"\n处理节点: {node}")
            node_info = other_entity.get_entity_info(node)
            if node_info:
                # 检查节点是否已存在
                main_id = self._get_main_id(node)
                if main_id:
                    print(f"节点 '{node}' 已存在，主实体ID为 '{main_id}'")
                    # 合并内容
                    self._merge_entity_content(main_id, node_info["content"])

                    # 合并别名
                    for alias in node_info["aliases"]:
                        if alias not in self.storage.alias_to_main_id:
                            self._add_alias(main_id, alias)
                            print(f"添加别名: {alias} -> {main_id}")
                else:
                    # 添加新节点
                    print(f"添加新节点: {node}")
                    new_id = self.add_entity(node, node_info["content"])
                    # 添加别名
                    for alias in node_info["aliases"]:
                        self._add_alias(new_id, alias)

        # 2. 合并关系
        for edge in other_entity.storage.graph.edges(data=True):
            source, target, data = edge
            source_main = self._get_main_id(source)
            target_main = self._get_main_id(target)

            if source_main and target_main:
                # 检查关系是否已存在
                existing_relationships = self.get_relationships(
                    source_main, target_main
                )
                if data["type"] not in existing_relationships:
                    self.add_relationship(source_main, target_main, data["type"])
                    print(f"添加关系: {source_main} -{data['type']}-> {target_main}")
                else:
                    print(f"关系已存在: {source_main} -{data['type']}-> {target_main}")

        # 3. 重建向量库
        print("\n更新向量库...")
        # 更新实体向量库
        for node in self.storage.graph.nodes():
            content = self.storage.load_entity(node)
            if content:
                self.storage._create_entity_vector_store(node, content)
                print(f"已更新实体 '{node}' 的向量库")

        # 4. 保存更新后的图谱
        self.storage.save()
        print("\n图谱合并完成！")

        # 5. 打印合并统计信息
        print("\n合并统计:")
        print(f"- 总节点数: {len(self.storage.graph.nodes())}")
        print(f"- 总关系数: {len(self.storage.graph.edges())}")
        print(
            f"- 总别名数: {sum(len(aliases) for aliases in self.storage.entity_aliases.values())}"
        )
        print(f"- 向量库数量: {len(self.storage.vector_stores)}")

    def detect_communities(
        self, resolution: float = 1.2, min_community_size: int = 4
    ) -> Dict[int, Dict]:
        """
        检测和分析社区

        Args:
            resolution: 社区划分的分辨率参数
            min_community_size: 最小社区大小

        Returns:
            Dict[int, Dict]: 社区信息字典
        """
        # 获取图的副本并移除自环
        G = self.storage.graph.copy()
        G.remove_edges_from(nx.selfloop_edges(G))
        print("开始社区检测：图的节点数:", len(G.nodes), "图的边数:", len(G.edges))

        # 使用Louvain方法检测社区
        raw_communities = nx.community.louvain_communities(
            G, resolution=resolution, seed=42
        )
        print("检测到的社区数:", len(raw_communities))

        # 分析每个社区
        communities_data = {}
        for idx, members in enumerate(raw_communities):
            if len(members) < min_community_size:
                print(
                    f"社区 {idx} 被跳过，因为成员数 {len(members)} 小于阈值 {min_community_size}"
                )
                continue

            print(f"\n正在处理社区 {idx}，成员数: {len(members)}")
            members_list = list(members)

            # 获取核心成员
            central_members = self._identify_central_members(members_list)
            print(f"社区 {idx} 的核心成员: {central_members}")

            # 获取社区内所有关系
            community_relations = self._get_community_relations(members_list)
            print(f"社区 {idx} 的关系数: {len(community_relations)}")

            # 生成社区摘要
            summary = self._generate_community_summary(
                members_list, central_members, community_relations
            )
            print(f"社区 {idx} 的摘要: {summary[:200]}...")  # 只打印前200字符

            # 创建社区信息字典
            communities_data[idx] = {
                "members": members_list,
                "central_members": central_members,
                "relations": community_relations,
                "summary": summary,
            }

        # 保存社区数据和摘要
        self.storage.save_communities(communities_data)
        self.storage.save_community_summaries(communities_data)
        print("\n社区检测完成，结果已保存。")

        return communities_data

    def _identify_central_members(self, members: List[str]) -> List[str]:
        """
        识别社区的核心成员

        Args:
            members: 社区成员列表

        Returns:
            List[str]: 核心成员列表
        """
        # 计算每个成员的连接度
        member_degrees = {}
        for member in members:
            successors = set(self.storage.graph.successors(member))
            predecessors = set(self.storage.graph.predecessors(member))
            # 只考虑社区内的连接
            community_connections = len(
                [n for n in successors.union(predecessors) if n in members]
            )
            member_degrees[member] = community_connections

        # 选择连接度最高的前4个成员
        central_members = sorted(
            member_degrees.items(), key=lambda x: x[1], reverse=True
        )[:4]

        return [member for member, _ in central_members]

    def _get_community_relations(self, members: List[str]) -> List[Dict]:
        """
        获取社区内的所有关系

        Args:
            members: 社区成员列表

        Returns:
            List[Dict]: 关系列表，每个关系包含 source, target, type
        """
        relations = []
        for source in members:
            for target in self.storage.graph.successors(source):
                if target in members:
                    edges = self.storage.graph.get_edge_data(source, target)
                    for edge_data in edges.values():
                        relations.append(
                            {
                                "source": source,
                                "target": target,
                                "type": edge_data["type"],
                            }
                        )
        return relations

    def _generate_community_summary(
        self, members: List[str], central_members: List[str], relations: List[Dict]
    ) -> str:
        """
        生成社区摘要

        Args:
            members: 所有社区成员列表
            central_members: 核心成员列表
            relations: 社区内的关系列表

        Returns:
            str: 社区摘要
        """
        try:
            # 1. 格式化核心成员信息
            core_entities_info = [f"- {entity}" for entity in central_members]

            # 2. 处理关系信息
            # 2.1 计算实体的连接度
            entity_connections = {member: 0 for member in members}
            for rel in relations:
                entity_connections[rel["source"]] = (
                    entity_connections.get(rel["source"], 0) + 1
                )
                entity_connections[rel["target"]] = (
                    entity_connections.get(rel["target"], 0) + 1
                )

            # 2.2 按关系类型分组并计算权重
            relation_groups = {}
            for rel in relations:
                rel_type = rel["type"]
                if rel_type not in relation_groups:
                    relation_groups[rel_type] = []

                # 计算该关系的权重（两个实体的总连接度）
                weight = (
                    entity_connections[rel["source"]]
                    + entity_connections[rel["target"]]
                )
                relation_groups[rel_type].append(
                    {
                        "source": rel["source"],
                        "target": rel["target"],
                        "weight": weight,
                        "type": rel_type,
                    }
                )

            # 2.3 对每种关系类型内的实体对按权重排序
            relation_info = []
            for rel_type, rel_list in relation_groups.items():
                # 按权重排序实体对
                sorted_rels = sorted(rel_list, key=lambda x: x["weight"], reverse=True)
                # 格式化关系信息
                examples = [
                    f"{rel['source']}-{rel['type']}-{rel['target']}"
                    for rel in sorted_rels[:3]
                ]  # 只取权重最高的前3个
                relation_info.append(f"- {'; '.join(examples)}")

            # 3. 读取并填充提示模板
            with open(COMMUNITY_SUMMARY_PROMPT, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # 4. 填充模板
            prompt = prompt_template.format(
                core_entities="\n".join(core_entities_info),
                relationships="\n".join(relation_info),
            )

            # 5. 生成摘要
            response = self.llm_client.chat.completions.create(
                model="moonshot-v1-auto",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            return response.choices[0].message.content.strip().replace("\n", " ")

        except Exception as e:
            print(f"生成社区摘要时发生错误: {str(e)}")
