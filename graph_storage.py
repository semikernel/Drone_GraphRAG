import os
import json
import base64
import shutil
from typing import List, Tuple, Dict, Optional, Set, Any
import networkx as nx
import numpy as np
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from embedding_model import EmbeddingModel


class GraphStorage:
    """图谱存储管理器，处理所有与存储相关的操作"""

    def __init__(self, base_path: str):
        """
        初始化存储管理器

        Args:
            base_path: 基础存储路径
        """
        # 基础路径
        self.base_path = base_path

        # 核心文件路径（直接在base_path下）
        self.graph_file = os.path.join(base_path, "graph.json")  # 图结构文件
        self.embeddings_file = os.path.join(
            base_path, "embeddings.json"
        )  # 实体嵌入文件
        self.global_doc_path = os.path.join(base_path, "global.md")  # 全局文档

        # 子文件夹路径
        self.entity_path = os.path.join(base_path, "entities")  # 实体文档文件夹
        self.vector_path = os.path.join(base_path, "vectors")  # 向量存储文件夹

        # 核心组件
        self.graph = nx.MultiDiGraph()

        # 实体管理
        self.entity_embeddings: Dict[str, np.ndarray] = {}  # 实体嵌入
        self.entity_aliases: Dict[str, Set[str]] = {}  # 实体别名
        self.alias_to_main_id: Dict[str, str] = {}  # 别名到主实体的映射

        # 向量存储
        self.vector_stores: Dict[str, FAISS] = {}  # 实体向量库
        self.global_vector_store: Optional[FAISS] = None  # 全局向量库
        self.global_content: Set[str] = set()  # 全局文档内容

        # 添加社区相关的存储路径
        self.community_file = os.path.join(base_path, "communities.json")
        self.community_summary_path = os.path.join(base_path, "community_summaries.md")
        self.communities: Dict[int, Dict] = {}  # {community_id: community_data}
        self.community_vector_store: Optional[FAISS] = None

        # 变更追踪：仅追踪实体修改
        self.modified_entities: Set[str] = set()

    def _init_storage(self) -> None:
        """初始化存储结构"""
        # 创建必要的目录和文件
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.entity_path, exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

        # 创建全局文档（如果不存在）
        if not os.path.exists(self.global_doc_path):
            with open(self.global_doc_path, "w", encoding="utf-8") as f:
                pass

    def save(self) -> None:
        """保存图谱数据"""
        # 保存图结构和别名信息
        graph_data = {
            "graph": nx.node_link_data(self.graph),
            "aliases": {k: list(v) for k, v in self.entity_aliases.items()},
            "alias_to_main_id": self.alias_to_main_id,
        }
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 保存实体嵌入
        embeddings_data = {}
        for k, v in self.entity_embeddings.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            embeddings_data[k] = v.tolist()
        with open(self.embeddings_file, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f)

        # 更新修改过的实体的向量库
        for entity_id in self.modified_entities:
            if entity_id in self.graph.nodes():
                content = self.load_entity(entity_id)
                if content:
                    self._create_entity_vector_store(entity_id, content)
                    print(f"更新实体 '{entity_id}' 的向量库")

        # 更新全局向量库
        if os.path.exists(self.global_doc_path):
            self._create_global_vector_store()
            print(f"更新全局向量库")

        # 清空变更追踪
        self.modified_entities.clear()

    def load(self) -> None:
        """加载图谱数据"""
        # 加载图结构和别名
        if os.path.exists(self.graph_file):
            print(f"检测到已存在的知识图谱在 '{self.base_path}'，正在加载...")
            with open(self.graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data["graph"], multigraph=True)
            self.entity_aliases = {k: set(v) for k, v in data["aliases"].items()}
            self.alias_to_main_id = data["alias_to_main_id"]

            # 加载实体嵌入
            if os.path.exists(self.embeddings_file):
                print("正在加载实体嵌入...")
                with open(self.embeddings_file, "r", encoding="utf-8") as f:
                    embeddings_data = json.load(f)
                # 直接将列表转换为numpy数组，因为加载的数据已经是列表形式
                self.entity_embeddings = {
                    k: np.array(v) for k, v in embeddings_data.items()
                }
            else:
                print("未找到实体嵌入文件，正在重新生成...")
                self._regenerate_embeddings()

            # 加载全局文档内容
            if os.path.exists(self.global_doc_path):
                with open(self.global_doc_path, "r", encoding="utf-8") as f:
                    self.global_content = set(f.read().split("\n\n"))

            # 加载向量库
            self._load_vector_stores()

            # 加载社区数据
            self._load_community_data()

    def _load_community_data(self) -> None:
        """尝试加载社区相关数据"""
        # 检查并加载社区JSON数据
        if os.path.exists(self.community_file):
            print("检测到社区数据，正在加载...")
            try:
                with open(self.community_file, "r", encoding="utf-8") as f:
                    self.communities = json.load(f)
                print(f"已加载 {len(self.communities)} 个社区的数据")

                # 加载社区向量存储
                store_path = os.path.join(self.vector_path, "community_summaries")
                if os.path.exists(store_path):
                    print("正在加载社区摘要向量存储...")
                    self.community_vector_store = FAISS.load_local(
                        store_path,
                        EmbeddingModel.get_instance(),
                        allow_dangerous_deserialization=True,
                    )
                    print("社区摘要向量存储加载完成")
                else:
                    print("正在为社区摘要创建向量存储...")
                    self._create_community_summary_store()

            except Exception as e:
                print(f"加载社区数据时发生错误: {str(e)}")
                self.communities = {}
                self.community_vector_store = None
        else:
            print("未检测到社区数据，跳过加载")
            self.communities = {}
            self.community_vector_store = None

    def save_entity(self, entity_id: str, content_units: List[Tuple[str, str]]) -> None:
        """
        保存实体数据

        Args:
            entity_id: 实体ID
            content_units: [(title, content),...] 格式的内容单元列表
        """
        # 保存到markdown文件
        file_path = os.path.join(self.entity_path, f"{entity_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            for title, content in content_units:
                f.write(f"# {title}\n\n{content}\n\n")

        # 更新全局文档
        self._update_global_document(content_units)

        # 标记实体为已修改
        self.modified_entities.add(entity_id)

    def load_entity(self, entity_id: str) -> List[Tuple[str, str]]:
        """
        加载实体数据

        Args:
            entity_id: 实体ID

        Returns:
            List[Tuple[str, str]]: 内容单元列表
        """
        file_path = os.path.join(self.entity_path, f"{entity_id}.md")
        if not os.path.exists(file_path):
            return []

        content_units = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            title = ""
            content = ""
            for line in lines:
                if line.startswith("# "):
                    if title and content:
                        content_units.append((title.strip(), content.strip()))
                        content = ""
                    title = line[2:].strip()
                else:
                    content += line
            if title and content:
                content_units.append((title.strip(), content.strip()))
        return content_units

    def save_communities(self, communities_data: Dict[int, Dict]) -> None:
        """保存社区数据到JSON"""
        self.communities = communities_data
        with open(self.community_file, "w", encoding="utf-8") as f:
            json.dump(communities_data, f, ensure_ascii=False, indent=2)

    def save_community_summaries(self, communities_data: Dict[int, Dict]) -> None:
        """生成并保存社区摘要文档"""
        # 生成markdown格式的社区摘要
        doc_content = []
        for comm_id, comm_data in communities_data.items():
            doc_content.append(f"# Community_{comm_id}\n")
            doc_content.append(f"{comm_data['summary']}\n\n")

        # 保存摘要文档
        with open(self.community_summary_path, "w", encoding="utf-8") as f:
            f.write("".join(doc_content))

        # 创建向量存储
        self._create_community_summary_store()

    def get_entity_count(self) -> int:
        """获取实体数量"""
        return len(self.graph.nodes())

    def get_relationship_count(self) -> int:
        """获取关系数量"""
        return len(self.graph.edges())

    def get_alias_count(self) -> int:
        """获取别名数量"""
        return sum(len(aliases) for aliases in self.entity_aliases.values())

    def get_store_count(self) -> int:
        """获取向量存储数量"""
        return len(self.vector_stores)

    def _regenerate_embeddings(self) -> None:
        """重新生成所有实体的嵌入向量"""
        self.entity_embeddings = {}
        for node in self.graph.nodes():
            embedding = EmbeddingModel.get_instance().embed_query(node)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            self.entity_embeddings[node] = embedding

    def _load_vector_stores(self) -> None:
        """加载向量存储"""
        # 加载全局向量库
        global_store_path = os.path.join(self.vector_path, "global")
        if os.path.exists(global_store_path):
            self.global_vector_store = FAISS.load_local(
                global_store_path,
                EmbeddingModel.get_instance(),
                allow_dangerous_deserialization=True,
            )
        elif os.path.exists(self.global_doc_path):
            self._create_global_vector_store()

        # 加载实体向量库
        for node in self.graph.nodes():
            store_path = os.path.join(self.vector_path, self._encode_filename(node))
            if os.path.exists(store_path):
                try:
                    vector_store = FAISS.load_local(
                        store_path,
                        EmbeddingModel.get_instance(),
                        allow_dangerous_deserialization=True,
                    )
                    self.vector_stores[node] = vector_store
                except Exception as e:
                    print(f"加载实体'{node}'的向量库失败: {str(e)}")
                    content = self.load_entity(node)
                    if content:
                        self._create_entity_vector_store(node, content)
            else:
                print(f"实体'{node}'的向量库不存在，正在生成...")
                content = self.load_entity(node)
                if content:
                    self._create_entity_vector_store(node, content)

    def _create_community_summary_store(self) -> None:
        """为社区摘要创建向量存储"""
        if not os.path.exists(self.community_summary_path):
            print("创建失败，没有社区文档")
            return

        store_path = os.path.join(self.vector_path, "community_summaries")

        # 使用标题分割文档
        headers_to_split_on = [("#", "Community")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        with open(self.community_summary_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 分割文档
        docs = splitter.split_text(content)

        # 创建向量存储
        self.community_vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # 保存向量存储
        self.community_vector_store.save_local(store_path)
        print("社区摘要向量存储创建完成")

    def _create_entity_vector_store(
        self, entity_id: str, content_units: List[Tuple[str, str]]
    ) -> None:
        """
        为实体创建向量存储

        Args:
            entity_id: 实体ID
            content_units: 内容单元列表
        """
        store_path = os.path.join(self.vector_path, self._encode_filename(entity_id))

        # 构建markdown文本
        markdown_text = ""
        for title, content in content_units:
            markdown_text += f"# {title}\n\n{content}\n\n"

        # 分割文档
        headers_to_split_on = [("#", "Header 1")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        docs = md_splitter.split_text(markdown_text)

        # 创建向量存储
        vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # 保存
        vector_store.save_local(store_path)
        self.vector_stores[entity_id] = vector_store

    def _create_global_vector_store(self) -> None:
        """创建全局向量存储"""
        if not os.path.exists(self.global_doc_path):
            return

        store_path = os.path.join(self.vector_path, "global")

        # 读取和分割文档
        with open(self.global_doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        headers_to_split_on = [("#", "Header 1")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        docs = md_splitter.split_text(content)

        # 创建向量存储
        self.global_vector_store = FAISS.from_documents(
            documents=docs,
            embedding=EmbeddingModel.get_instance(),
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

        # 保存
        self.global_vector_store.save_local(store_path)

    def _update_global_document(self, content_units: List[Tuple[str, str]]) -> None:
        """
        更新全局文档

        Args:
            content_units: 新的内容单元列表
        """
        new_content = set()
        for title, content in content_units:
            new_content.add(f"# {title}\n\n{content}\n\n")

        new_entries = new_content - self.global_content
        if new_entries:
            with open(self.global_doc_path, "a", encoding="utf-8") as f:
                for entry in new_entries:
                    f.write(entry)
            self.global_content.update(new_entries)

    @staticmethod
    def _encode_filename(filename: str) -> str:
        """文件名编码"""
        filename_bytes = filename.encode("utf-8")
        encoded_bytes = base64.urlsafe_b64encode(filename_bytes)
        return encoded_bytes.decode("utf-8")

    @staticmethod
    def _decode_filename(encoded_filename: str) -> str:
        """文件名解码"""
        try:
            decoded_bytes = base64.urlsafe_b64decode(encoded_filename.encode("utf-8"))
            return decoded_bytes.decode("utf-8")
        except Exception as e:
            print(f"解码文件名时发生错误: {str(e)}")
            return encoded_filename

    def cleanup(self) -> None:
        """
        清理资源
        主要清理内存中的缓存数据
        """
        try:
            # 保存当前状态
            self.save()

            # 清空内存中的向量存储引用
            self.vector_stores.clear()
            self.global_vector_store = None
            self.community_vector_store = None

            # 清空其他内存缓存
            self.entity_embeddings.clear()
            self.global_content.clear()
            self.modified_entities.clear()
            self.communities.clear()

        except Exception as e:
            print(f"清理资源时发生错误: {str(e)}")

    def remove_entity(self, entity_id: str) -> None:
        """
        删除实体及其相关数据

        Args:
            entity_id: 实体ID
        """
        try:
            # 从修改追踪中移除
            self.modified_entities.discard(entity_id)

            # 删除实体文档
            file_path = os.path.join(self.entity_path, f"{entity_id}.md")
            if os.path.exists(file_path):
                os.remove(file_path)

            # 删除向量存储
            store_path = os.path.join(
                self.vector_path, self._encode_filename(entity_id)
            )
            if os.path.exists(store_path):
                shutil.rmtree(store_path)  # 直接删除向量库文件夹
                if entity_id in self.vector_stores:
                    del self.vector_stores[entity_id]  # 从内存中移除引用

            # 删除实体嵌入
            if entity_id in self.entity_embeddings:
                del self.entity_embeddings[entity_id]

            # 从图中移除节点（这会自动移除相关的边）
            if entity_id in self.graph:
                self.graph.remove_node(entity_id)

            # 更新别名
            if entity_id in self.entity_aliases:
                aliases = self.entity_aliases[entity_id]
                for alias in aliases:
                    if alias in self.alias_to_main_id:
                        del self.alias_to_main_id[alias]
                del self.entity_aliases[entity_id]

            print(f"成功删除实体 '{entity_id}' 及其相关数据")

        except Exception as e:
            print(f"删除实体 '{entity_id}' 时发生错误: {str(e)}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源被正确释放"""
        self.cleanup()
