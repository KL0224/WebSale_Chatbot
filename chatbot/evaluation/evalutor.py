import os
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from langchain_groq.chat_models import ChatGroq
from langchain_core.embeddings import Embeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from FlagEmbedding import BGEM3FlagModel


# =========================
# Embedding Wrapper
# =========================
class M3Embedder(Embeddings):
    def __init__(self, model: BGEM3FlagModel):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        out = self.model.encode([text])
        v = out[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self.model.encode(texts)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in out]


# =========================
# Safe context parser
# =========================
def safe_parse_context(x):
    """
    Parse contexts safely from CSV.
    Accepts:
    - JSON list
    - Python list string
    - Already list
    """
    if isinstance(x, list):
        return x

    if not isinstance(x, str):
        return []

    x = x.strip()
    if not x:
        return []

    # Try JSON
    try:
        data = json.loads(x)
        if isinstance(data, list):
            return data
    except:
        pass

    # Try python literal manually
    if x.startswith("[") and x.endswith("]"):
        try:
            # replace smart quotes
            x2 = x.replace("“", '"').replace("”", '"').replace("’", "'")
            data = eval(x2, {"__builtins__": None}, {})
            if isinstance(data, list):
                return data
        except:
            return []

    return []


# =========================
# Dataset cleaning
# =========================
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"question", "answer", "contexts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing columns: {missing}")

    print("🧹 Cleaning dataset...")

    df["contexts"] = df["contexts"].apply(safe_parse_context)

    invalid_rows = []

    for i, row in df.iterrows():
        if not isinstance(row["contexts"], list) or len(row["contexts"]) == 0:
            invalid_rows.append(i)

    if invalid_rows:
        print("\nDATASET STILL DIRTY ❌")
        print(f"Found invalid rows: {len(invalid_rows)}\n")

        for i in invalid_rows:
            print("Row:", i)
            print("Question:", df.loc[i, "question"])
            print("Answer:", df.loc[i, "answer"])
            print("Contexts(raw):", df.loc[i, "contexts"])
            print("------")

        print("⚠️ Auto-removing invalid rows...")
        df = df.drop(index=invalid_rows).reset_index(drop=True)

    print(f"✅ Dataset clean. Total rows after clean: {len(df)}")
    return df


# =========================
# Main
# =========================
def main():
    base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / "api_chatbot" / ".env"
    load_dotenv(dotenv_path=env_path)

    api_groq_key = os.getenv("API_GROQ_KEY")
    if not api_groq_key:
        raise RuntimeError("Missing API_GROQ_KEY in environment or `.env`")

    # -------- LLM --------
    judge_llm = ChatGroq(
        api_key=api_groq_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
    )

    # -------- Embeddings --------
    embeddings = M3Embedder(BGEM3FlagModel("BAAI/bge-m3"))

    # -------- Load data --------
    print("📥 Loading evaluation data...")
    df = pd.read_csv("rag_evaluation_dataset.csv", encoding="utf-8")

    # -------- Clean data --------
    df = clean_dataset(df)

    # -------- HuggingFace dataset --------
    eval_dataset = Dataset.from_pandas(df)

    # -------- Evaluate --------
    print("\n🚀 Evaluating with RAGAS...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # -------- Save --------
    print("\n📊 Evaluation result:")
    print(result)

    result.to_pandas().to_csv(
        "evaluation_report_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n✅ Saved: evaluation_report_detailed.csv")


if __name__ == "__main__":
    main()