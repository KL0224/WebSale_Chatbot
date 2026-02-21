import pandas as pd
import ast
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_groq import ChatGroq
from pathlib import Path
from dotenv import load_dotenv

# 1. Cấu hình LLM làm "Giám khảo" (Judge)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_groq_key = os.getenv("API_GROQ_KEY")
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def main():
    # 2. Load dữ liệu từ CSV
    print("🔄 Đang nạp dữ liệu đánh giá...")
    df = pd.read_csv("rag_evaluation_dataset.csv")

    # QUAN TRỌNG: Chuyển cột contexts từ chuỗi (string) về lại dạng List
    df['contexts'] = df['contexts'].apply(ast.literal_eval)

    # Chuyển sang định dạng Dataset của HuggingFace
    eval_dataset = Dataset.from_pandas(df)

    # 3. Tiến hành đánh giá
    print("🚀 Đang bắt đầu chấm điểm (Scoring)... Vui lòng đợi.")

    # Chúng ta chỉ chọn các metrics không cần ground_truth
    metrics = [
        faithfulness,
        answer_relevance,
        context_precision
    ]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        raise_exceptions=False
    )

    # 4. Xuất kết quả
    print("\n" + "=" * 30)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TỔNG QUÁT")
    print(result)
    print("=" * 30)

    # Lưu chi tiết điểm số của từng câu hỏi để phân tích lỗi
    result_df = result.to_pandas()
    result_df.to_csv("evaluation_report_detailed.csv", index=False, encoding='utf-8-sig')
    print("✅ Đã lưu báo cáo chi tiết vào file: evaluation_report_detailed.csv")


if __name__ == "__main__":
    main()