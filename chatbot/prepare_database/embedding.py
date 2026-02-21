from FlagEmbedding import BGEM3FlagModel
import json
import uuid
import pickle

class EmbeddingModel:
    def __init__(self, model_name, batch_size=1, max_length=1000):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
            self.model = BGEM3FlagModel(self.model_name)

    def embedding(self, texts):
        output = self.model.encode(texts, return_dense=True, return_sparse=True, batch_size=self.batch_size, max_length=self.max_length)
        return output["dense_vecs"], output['lexical_weights']

def embedding_products(input_file: str, output_file: str):
    # Load file chunks
    print("Đọc file chunks")
    chunks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                chunks.append(obj)

    print("Khởi tạo mô hình embedding")
    model = EmbeddingModel(model_name='BAAI/bge-m3', batch_size=8, max_length=1000)
    model.load_model()

    print("Tiến hành embedding")
    text_to_embedding = [chunk['text_chunk'] for chunk in chunks]
    embeddings_dense, embeddings_sparse = model.embedding(text_to_embedding)

    final_data = []
    for i in range(len(chunks)):
        record = {
            "id": str(uuid.uuid4()),
            "dense_vector": embeddings_dense[i].tolist(),
            "sparse_vector": embeddings_sparse[i],
            "payload": {
                "text_answer": chunks[i]['text_answer'],
                "name": chunks[i]['name'],
                "brand": chunks[i]['brand'],
                "category": chunks[i]['category'],
                "price": chunks[i]['price'],
                "discount": chunks[i]['discount'],
                "stock": chunks[i]['stock'],
                "colors": chunks[i]['colors'],
                "type": chunks[i]['type']
            }
        }
        final_data.append(record)

    print("Lưu dữ liệu ra file")
    with open(output_file, "wb") as f:
        pickle.dump(final_data, f)

    print("Đã lưu trữ thành công")

def embedding_policy(input_file: str, output_file: str):
    print("Đọc file chunks policy")
    chunks_policy = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_policy.append(json.loads(line))

    print("Load model embedding")
    model = EmbeddingModel(model_name='BAAI/bge-m3', batch_size=8, max_length=1000)
    model.load_model()

    print("Tiến hành embedding")
    text_to_embedding = []
    for chunk in chunks_policy:
        text = "chủ đề " + chunk.get('section', '').lower() + " mục " +  chunk.get('subsection', '').lower() + " có nội dung: " + chunk.get('text_chunk', '').lower()
        text_to_embedding.append(text)
    embeddings_dense, embeddings_sparse = model.embedding(text_to_embedding)

    final_data = []
    for i in range(len(chunks_policy)):
        record = {
            "id": str(uuid.uuid4()),
            "dense_vector": embeddings_dense[i].tolist(),
            "sparse_vector": embeddings_sparse[i],
            "payload": {
                "text_answer": text_to_embedding[i],
                "type": "policy"
            }
        }
        final_data.append(record)

    with open(output_file, "wb") as f:
        pickle.dump(final_data, f)

    print("Đã lưu file thành công")


def main():
    # input_file = "chunking/chunks_product_bge.jsonl"
    # output_file = "chunking/chunks_product_bge_embedding.pkl"
    # embedding_products(input_file, output_file)

    input_file = "prepare_database/policy.jsonl"
    output_file = "prepare_database/chunks_policy_embedding.pkl"
    embedding_policy(input_file, output_file)

if __name__ == "__main__":
    # main()
    a = ["I love you", "Hello friend", "No embedding", "No promplel"]
    print("Khởi tạo mô hình embedding")
    model = EmbeddingModel(model_name='BAAI/bge-m3', batch_size=8, max_length=1000)
    model.load_model()
    dense, sparse = model.embedding(a)
    print(len(dense))
    print(len(sparse))




