from langchain_text_splitters import MarkdownHeaderTextSplitter
from FlagEmbedding import BGEM3FlagModel
import json
import pandas as pd
import re
import numpy as np

# load mapping brand
with open("chunking/brand_mapping.json", 'r', encoding='utf-8') as f:
    brand_mapping = json.load(f)

# load mapping category
with open("chunking/category_mapping.json", 'r', encoding='utf-8') as f:
    category_mapping = json.load(f)


def split_sentences(text):
    text = text.lower()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences]
    return sentences

def create_window_text(sentences, window_size=3, overlap=1):
    window_text = []
    for i in range(0, len(sentences) - window_size + 1, window_size - overlap):
        window_text.append(" ".join(sentences[i: i + window_size]))
    return window_text

def cosin_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

class EmbeddingBGE:
    def __init__(self, model_name, max_length, batch_size=1):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)

    def embedding(self, windows):
        output = self.model.encode(windows, batch_size=self.batch_size, max_length=self.max_length)
        return output["dense_vecs"]

class ChunkData:
    def __init__(self):
        self.chunks = []
        self.metadata = []
        self.len_chunks = 0

    def markdown_chunking(self, filename: str) -> None:
        # Read markdown file
        with open(filename, 'r', encoding='utf-8') as f:
            data_markdown = f.read()

        # Call markdown chunking form langchain
        headers_to_split_on = [
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_hearder_split = markdown_splitter.split_text(data_markdown)

        for doc in md_hearder_split:
            self.chunks.append(doc.page_content)
            self.metadata.append(doc.metadata)
            self.len_chunks += 1

    def semantic_chunking(
            self,
            filename: str,
            percentile_threshold=90,
            min_chunk_chars=120,
            max_chunk_chars=1200
    ) -> None:

        # Load model embedding
        model = EmbeddingBGE(model_name='BAAI/bge-m3', max_length=1000, batch_size=8)
        model.load_model()

        # Chunking data
        product_data = pd.read_csv(filename)

        for pro in product_data.itertuples():

            # Normalize base fields
            name = str(pro.name).lower()
            brand = brand_mapping[str(pro.brand_id)].lower()
            category = category_mapping[str(pro.category_id)].lower()
            colors = str(pro.colors).lower()
            tech_specs = str(pro.tech_specs).lower()
            description = str(pro.description).lower()

            # Chunk product intro
            chunk_pro = (
                f"sản phẩm {name} thuộc loại sản phẩm {category} "
                f"và thương hiệu {brand}. "
                f"{name} có các loại màu sắc như: {colors}. "
                f"thông số kĩ thuật gồm: {tech_specs}."
            )

            # Base text for LLM
            base_text = (
                f"sản phẩm {name} thuộc loại sản phẩm {category} "
                f"và thương hiệu {brand}. "
                f"{name} có các loại màu sắc như: {colors}. "
                f"thông số kĩ thuật gồm: {tech_specs}. "
                f"nó có giá {pro.price} vnđ. "
                f"khuyến mãi {pro.discount} % (tức chỉ cần trả "
                f"{(pro.price - (pro.discount * pro.price / 100))} vnđ, "
                f"tiết kiệm được {pro.discount * pro.price / 100} vnđ). "
                f"{name} còn {pro.stock} sản phẩm."
            )

            base_meta = {
                "name": name,
                "brand": brand,
                "category": category,
                "price": pro.price,
                "discount": pro.discount,
                "stock": pro.stock,
                "colors": colors,
                "type": "product_info"
            }

            self.chunks.append(chunk_pro)
            self.metadata.append({**base_meta, "text_answer": base_text})

            # Split description
            des_sents = split_sentences(description)
            if len(des_sents) <= 1:
                continue

            embeddings = model.embedding(des_sents)
            normalied_embeddings = normalize_embeddings(embeddings)

            # Calculate semantic distances
            distances = []
            for i in range(len(normalied_embeddings) - 1):
                sim = cosin_similarity(normalied_embeddings[i], normalied_embeddings[i + 1])
                distances.append(1 - sim)
            distances = np.array(distances)

            # Adaptive threshold
            threshold = np.percentile(distances, percentile_threshold)

            # Build semantic chunks
            current_chunk = [des_sents[0]]

            for i, dis in enumerate(distances):

                if dis > threshold and len(" ".join(current_chunk)) > min_chunk_chars:
                    chunk_text = (
                            f"sản phẩm {name} thuộc loại sản phẩm {category} "
                            f"và thương hiệu {brand} có mô tả: "
                            + " ".join(current_chunk)
                    )

                    self.chunks.append(chunk_text)
                    self.metadata.append({
                        **base_meta,
                        "type": "product_description",
                        "text_answer": chunk_text
                    })

                    current_chunk = [des_sents[i + 1]]
                else:
                    current_chunk.append(des_sents[i + 1])

                # hard split by max length
                if len(" ".join(current_chunk)) > max_chunk_chars:
                    chunk_text = (
                            f"sản phẩm {name} thuộc loại sản phẩm {category} "
                            f"và thương hiệu {brand} có mô tả: "
                            + " ".join(current_chunk)
                    )

                    self.chunks.append(chunk_text)
                    self.metadata.append({
                        **base_meta,
                        "type": "product_description",
                        "text_answer": chunk_text
                    })

                    current_chunk = []

            # Last chunk
            if current_chunk:
                chunk_text = (
                        f"sản phẩm {name} thuộc loại sản phẩm {category} "
                        f"và thương hiệu {brand} có mô tả: "
                        + " ".join(current_chunk)
                )

                self.chunks.append(chunk_text)
                self.metadata.append({
                    **base_meta,
                    "type": "product_description",
                    "text_answer": chunk_text
                })

    def save_chunk(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            for chunk, meta in zip(self.chunks, self.metadata):
                data = {"text_chunk": chunk, **meta}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    # chunker_markdown = ChunkData()
    # chunker_markdown.markdown_chunking('chunking/policy.md')
    # chunker_markdown.save_chunk('chunking/policy.jsonl')

    chunker = ChunkData()
    chunker.semantic_chunking("chunking/products.csv")
    chunker.save_chunk("chunking/chunks_product_bge.jsonl")

if __name__ == '__main__':
    main()
