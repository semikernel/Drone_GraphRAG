import os
import sys
import json
import re
from openai import OpenAI
from knowledgeGraph import KnowledgeGraph
from config import *

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


class KnowledgeGraphExtractor:
    def __init__(self):
        # 初始化知识图谱
        self.kg = KnowledgeGraph("./knowledge_base")

        # 读取提示词模板
        self.entity_prompt = self.read_prompt("prompt/entity_extraction.txt")
        self.relation_prompt = self.read_prompt("prompt/relationship_extraction.txt")

        # 已处理文件记录
        self.progress_file = "data/processed_files.txt"
        self.processed_files = self.load_progress()

    def load_progress(self):
        """加载已处理的文件列表"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        return set()

    def save_progress(self, item_id):
        """记录已处理的文件"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(f"{item_id}\n")

    @staticmethod
    def read_prompt(file_path):
        """读取提示词模板"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def read_json_file(file_path):
        """读取JSON文件"""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def parse_ai_response(self, response):
        """解析AI响应"""
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"解析JSON失败: {json_str}")
            return {}

    def chat_with_LLM(self, messages):
        """与LLM交互"""
        try:
            response = client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=messages,
                temperature=0.5,
                response_format={"type": "json_object"},
                stream=True,
            )

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()
            return full_response

        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            raise

    def extract_entities(self, text):
        """提取实体"""
        messages = [
            {"role": "system", "content": self.entity_prompt},
            {
                "role": "user",
                "content": f"请从以下文本中提取与话题相关的核心实体：\n\n{text}",
            },
        ]
        response = self.chat_with_LLM(messages)
        return self.parse_ai_response(response)

    def extract_relations(self, text, entities):
        """提取关系"""
        entities_str = ", ".join(entities)
        messages = [
            {"role": "system", "content": self.relation_prompt},
            {
                "role": "user",
                "content": f"已知实体列表：{entities_str}\n\n请从以下文本中提取这些实体之间的关系：\n\n{text}",
            },
        ]
        response = self.chat_with_LLM(messages)
        return self.parse_ai_response(response)

    def process_item(self, item_id, item_data):
        """处理单个数据项"""
        try:
            title = item_data.get("title", "")
            clusters = item_data.get("clusters", [])

            print(f"正在处理数据 {item_id}: {title}")
            print(f"该数据包含 {len(clusters)} 个评论簇")

            entity_contents = {}
            all_relations = []

            for i, cluster in enumerate(clusters):
                print(f"处理第 {i+1}/{len(clusters)} 个评论簇")

                comments = [
                    comment.replace("\n", " ").strip()
                    for comment in cluster.get("comments", [])
                ]
                context = f"话题：{title}\n所有评论：\n" + "\n".join(comments)

                entities_result = self.extract_entities(context)
                entities = entities_result.get("entities", [])

                if not entities:
                    continue

                relations_result = self.extract_relations(context, entities)
                relations = relations_result.get("relations", [])

                if not relations:
                    continue

                content_unit_title = ", ".join(entities)
                content_unit = context

                for entity in entities:
                    if entity not in entity_contents:
                        entity_contents[entity] = []
                    entity_contents[entity].append((content_unit_title, content_unit))

                all_relations.extend(relations)

            if not entity_contents or not all_relations:
                print(f"数据 {item_id} 没有提取到有效的实体和关系")
                return False

            print(
                f"向知识图谱添加 {len(entity_contents)} 个实体和 {len(all_relations)} 个关系"
            )

            for entity, content_units in entity_contents.items():
                self.kg.add_entity(entity, content_units)

            for relation in all_relations:
                self.kg.add_relationship(
                    relation["source"], relation["target"], relation["relation"]
                )

            return True

        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            return False

    def process_data(self, input_file="data/results.json"):
        """处理数据文件"""
        try:
            data = self.read_json_file(input_file)

            unprocessed_items = [
                (item_id, data)
                for item_id, data in list(data.items())
                if item_id not in self.processed_files
            ]

            if not unprocessed_items:
                print("没有新的数据需要处理")
                return self.kg

            print(f"将处理 {len(unprocessed_items)} 个数据项")

            for item_id, item_data in unprocessed_items:
                if self.process_item(item_id, item_data):
                    self.save_progress(item_id)
                    self.processed_files.add(item_id)
                    self.kg.save()
                    print(f"数据 {item_id} 处理完成并保存")

            self.kg.merge_similar_entities()
            self.kg.remove_duplicates()
            self.kg.visualize()

            print("\n数据处理完成")
            return self.kg

        except Exception as e:
            print(f"处理文件出错: {str(e)}")
            raise


def main():
    try:
        print("\n开始处理数据")
        extractor = KnowledgeGraphExtractor()
        kg = extractor.process_data()
        print("知识图谱创建完成并可视化")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
