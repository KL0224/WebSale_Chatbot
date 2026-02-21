import json
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from FlagEmbedding import BGEM3FlagModel
from ragas.testset.synthesizers.generate import TestsetGenerator
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import os
import time
import pandas as pd

# Load api key
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "api_chatbot" / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_groq_key = os.getenv("API_GROQ_KEY")

if not api_groq_key:
    raise ValueError("API_GROQ_KEY không được tìm thấy trong file .env")

print(f"Đã tìm thấy API key: {api_groq_key[:5]}...")


def load_jsonl_to_ragas_docs(file_path: str) -> List[Document]:
    """Đọc file JSONL và chuyển thành list Document"""
    datasets = []

    # Kiểm tra file tồn tại
    if not os.path.exists(file_path):
        print(f"File không tồn tại: {file_path}")
        return datasets

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # Lấy nội dung content
            text_chunk = data.pop("text_chunk", None)
            if not text_chunk:
                continue

            # Loại bỏ trường text_answer để giảm token
            data.pop('text_answer', None)

            # Tạo document theo langchain
            doc = Document(page_content=text_chunk, metadata=data)
            datasets.append(doc)

    print(f"✅ Đã load {len(datasets)} documents từ {file_path}")
    return datasets


def generate(documents: List[Document]):
    """Tạo dữ liệu test bằng ragas"""

    if not documents:
        print("Không có documents để tạo testset")
        return

    print(f"Tổng số documents: {len(documents)}")

    MAX_DOCS = 300  # Giảm xuống 300 để test
    if len(documents) > MAX_DOCS:
        print(f"⚠️ Giới hạn số documents từ {len(documents)} xuống {MAX_DOCS}")
        documents = documents[:MAX_DOCS]  # Lấy documents đầu tiên

    print(f"📊 Tổng số documents: {len(documents)}")

    raw_generator = ChatGroq(
        api_key=api_groq_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=4096,
        request_timeout=120,
        max_retries=3
    )

    raw_critic = ChatGroq(
        api_key=api_groq_key,
        model="qwen/qwen3-32b",
        temperature=0.0,
        max_tokens=8192,
        request_timeout=120,
        max_retries=3
    )

    # Tạo mô hình embedding
    print("🔄 Đang load embedding model...")
    embeddings = BGEM3FlagModel("BAAI/bge-m3")
    print("✅ Đã load embedding model")

    # Tạo generator
    generator = TestsetGenerator.from_langchain(
        raw_generator,
        raw_critic,
        embeddings
    )


    total_questions = 100
    questions_per_batch = 10
    batches = total_questions // questions_per_batch  # 10 đợt
    all_testsets = []

    print(f"\n🚀 Bắt đầu quy trình tạo {total_questions} câu hỏi...")
    print(f"Sẽ thực hiện {batches} đợt, mỗi đợt {questions_per_batch} câu")
    print("=" * 50)

    for i in range(batches):
        print(f"\n📌 Đợt {i + 1}/{batches}")

        try:
            # Tạo 10 câu cho mỗi đợt
            batch_testset = generator.generate_with_langchain_docs(
                documents,
                testset_size=questions_per_batch,
                query_distribution={
                    "simple": 0.4,
                    "reasoning": 0.3,
                    "multi_context": 0.3
                }
            )

            # Chuyển sang DataFrame và lưu vào danh sách
            df_batch = batch_testset.to_pandas()
            all_testsets.append(df_batch)

            print(f"✅ Đã tạo {len(df_batch)} câu hỏi trong đợt {i + 1}")
            print(f"📝 Ví dụ câu hỏi: {df_batch['question'].iloc[0][:100]}...")

            # Nghỉ 60 giây trước khi sang đợt tiếp theo (trừ đợt cuối)
            if i < batches - 1:
                print("⏳ Tạm nghỉ 60s để tránh Rate Limit...")
                time.sleep(60)

        except Exception as e:
            print(f"Lỗi tại đợt {i + 1}: {e}")
            print("cĐang thử nghỉ thêm 30s rồi tiếp tục...")
            time.sleep(30)
            continue

    # Gộp tất cả các đợt lại thành 1 file duy nhất
    if all_testsets:
        final_df = pd.concat(all_testsets, ignore_index=True)

        # Thêm metadata
        final_df['timestamp'] = pd.Timestamp.now()
        final_df['total_documents'] = len(documents)

        # Lưu file
        output_file = "pak_final_100_questions.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 50)
        print("THÀNH CÔNG!")
        print(f"Đã lưu {len(final_df)} câu hỏi vào file: {output_file}")
        print(f"Phân bố câu hỏi:")
        print(final_df['evolution_type'].value_counts())
        print("=" * 50)

        # Hiển thị 5 câu hỏi đầu tiên
        print("\n5 câu hỏi đầu tiên:")
        for idx, row in final_df.head(5).iterrows():
            print(f"{idx + 1}. {row['question']}")
    else:
        print("Không tạo được câu hỏi nào!")


if __name__ == "__main__":
    # Load documents từ cả 2 file
    print("Đang load documents...")

    # Load sản phẩm
    product_docs = load_jsonl_to_ragas_docs("prepare_database/chunks_product_bge.jsonl")

    # Load policy
    policy_docs = load_jsonl_to_ragas_docs("prepare_database/policy.jsonl")

    # Gộp tất cả documents
    all_documents = policy_docs + product_docs

    print(f"Tổng số documents: {len(all_documents)}")
    print(f"Sản phẩm: {len(product_docs)} documents")
    print(f"Policy: {len(policy_docs)} documents")

    # Chạy generate
    generate(all_documents)