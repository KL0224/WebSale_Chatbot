from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_groq import ChatGroq
from prepare_database.embedding import EmbeddingModel
from prompt.prompt import get_llm_prompt, get_rewrite_prompt
from reranking.rerank import Reranker
from prepare_database.qdrant import parse_query_to_filter, QdrantDB
from router.identity_classification import IdentityClassificationLLM
from qdrant_client import QdrantClient, models
import pandas as pd
from tqdm import tqdm
import time

# Load env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_groq_key = os.getenv("API_GROQ_KEY")
tem_product = 0.5
tem_policy = 0.2
tem_chat = 0.7

# LLM
class LLM:
    def __init__(self, api_key, model="llama-3.1-8b-instant", temperature=0.5, max_tokens=1024, max_retries=2):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries)

    def generate(self, identity_flow, user_query=None, context=None, chat_history=""):
        if identity_flow == "product":
            temperature = tem_product
        elif identity_flow == "policy":
            temperature = tem_policy
        else:
            temperature = tem_chat

        message = get_llm_prompt(identity_flow, user_query, context, chat_history)

        if not message:
            return ""

        # Change temperature with flow
        current_llm = self.llm.bind(temperature=temperature)

        try:
            response = current_llm.invoke(message)
            return response.content
        except Exception as e:
            return f"[ERROR]: {str(e)}"

    def stream(self, identity_flow, user_query=None, context=None, chat_history=""):
        if identity_flow == "product":
            temperature = tem_product
        elif identity_flow == "policy":
            temperature = tem_policy
        else:
            temperature = tem_chat

        message = get_llm_prompt(identity_flow, user_query, context, chat_history)

        if not message:
            return ""

        # Change temperature with flow
        current_llm = self.llm.bind(temperature=temperature)

        try:
            for chunk in current_llm.stream(message):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"[ERROR]: {str(e)}"

    def rewrite_query(self, chat_history: str, user_query: str) -> str:
        """Dùng LLM để viết lại câu hỏi mơ hồ thành câu hỏi độc lập"""
        message = get_rewrite_prompt(chat_history, user_query)
        current_llm = self.llm.bind(temperature=0.1)
        try:
            response = current_llm.invoke(message)
            rewritten = response.content.strip()
            return rewritten if rewritten else user_query
        except Exception as e:
            print(f"[REWRITE ERROR]: {str(e)}")
            return user_query

class RAGPipeLine:
    def __init__(
            self,
            llm,
            embedding_model,
            qdrant_client,
            router,
            reranker,
            memory_manager=None,
            product_collection="products",
            policy_collection="policies",
    ):
        print("Đang tạo RAGPipeLine")
        self.llm = llm
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client

        self.product_collection = product_collection
        self.policy_collection = policy_collection

        self.rerank_model = reranker
        self.router_model = router
        self.memory = memory_manager

        print("Đã khởi tạo toàn bộ model RAGPipeLine thành công")

    def route(self, query: str) -> str:
        return self.rerank_model.classify()

    def retrieval(self, query: str, flow: str, k=30):
        if flow == "product":
            parsed_filter = parse_query_to_filter(query)

            # Xây dựng danh sách điều kiện lọc (must_conditions)
            must_conditions = []  # Dùng cho các điều kiện cứng như: category hay stock
            should_conditions = []  # Các điều kiện mềm theo cảm tính như: brand, price, discount, colors
            for key, val in parsed_filter.items():
                if key == 'category' and isinstance(val, str):
                    must_conditions.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=val))
                    )

                    # ĐIỀU KIỆN CỨNG (Must): Phải còn hàng (nếu có filter stock)
                elif key == 'stock' and isinstance(val, dict):
                    must_conditions.append(
                        models.FieldCondition(key=key, range=models.Range(**val))
                    )

                    # ĐIỀU KIỆN MỀM (Should): Ưu tiên Brand, Màu sắc, Giá cả
                elif key == 'brand' and isinstance(val, str):
                    should_conditions.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=val))
                    )

                elif key == 'colors':
                    if isinstance(val, list):
                        should_conditions.append(
                            models.FieldCondition(key=key, match=models.MatchAny(any=val))
                        )
                    else:
                        should_conditions.append(
                            models.FieldCondition(key=key, match=models.MatchValue(value=val))
                        )

                elif key == 'price' and isinstance(val, dict):
                    should_conditions.append(
                        models.FieldCondition(key=key, range=models.Range(**val))
                    )

            # Tạo Filter kết hợp
            search_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None
            )

            return self.qdrant_client.hybrid_search(
                query=query,
                model=self.embedding_model,
                collection_name=self.product_collection,
                k=k,
                search_filter=search_filter
            )

        elif flow == "policy":
            return self.qdrant_client.hybrid_search(
                query=query,
                model=self.embedding_model,
                collection_name=self.policy_collection,
                k=k,
                search_filter=None
            )

        else:
            return None

    def rerank(self, query, candidates, top_n=5):
        return self.rerank_model.rerank(query, candidates, top_n=top_n)

    def build_context(self, ranked_docs):
        blocks = []
        for i, item in enumerate(ranked_docs):
            doc = item['point']
            payload = doc.payload
            text = payload.get("text_answer", "")
            blocks.append(f"[DOC {i+1}] {text}")
        return "\n\n".join(blocks)


    def _format_history(self, messages) -> str:
        """Chuyển LangChain messages thành chuỗi text cho prompt"""
        if not messages:
            return ""
        lines = []
        for msg in messages:
            role = "Khách hàng" if msg.type == "human" else "Trợ lý"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _get_memory(self, session_id: str):
        """Lấy chat history từ Redis, trả về (messages, formatted_str)"""
        if not self.memory or not session_id:
            return [], ""
        messages = self.memory.get_history(session_id, limit=6)
        return messages, self._format_history(messages)

    def _rewrite_if_needed(self, query: str, chat_history_str: str, intent: str) -> str:
        """Rewrite câu hỏi nếu có history và intent cần retrieval"""
        if chat_history_str and intent in ("product", "policy"):
            rewritten = self.llm.rewrite_query(chat_history_str, query)
            if rewritten and rewritten != query:
                print(f"[REWRITE] '{query}' → '{rewritten}'")
                return rewritten
        return query

    def run(self, query: str, session_id: str = None):
        """Pipeline RAG with output non streaming"""
        # Lấy memory
        messages, chat_history_str = self._get_memory(session_id)

        # Router
        intent = self.router_model.classify(query)

        # Rewrite query nếu cần
        rewritten_query = self._rewrite_if_needed(query, chat_history_str, intent)

        # Chatchit flow
        if intent == "chitchat":
            answer = self.llm.generate(intent, query, context=None, chat_history=chat_history_str)
            # Lưu vào Redis
            if self.memory and session_id:
                self.memory.save(session_id, query, answer)
            return {
                "intent": intent,
                "query": query,
                "documents": None,
                "answer": answer
            }

        # Retrieval (dùng rewritten_query để search chính xác hơn)
        results = self.retrieval(rewritten_query, intent)

        if results is None:
            answer = self.llm.generate(intent, rewritten_query, context="", chat_history=chat_history_str)
            if self.memory and session_id:
                self.memory.save(session_id, query, answer)
            return {
                "intent": intent,
                "query": query,
                "documents": None,
                "answer": answer
            }

        # Rerank (dùng rewritten_query)
        reranked_results = self.rerank(rewritten_query, results, top_n=5)

        # Build context
        context = self.build_context(reranked_results)

        # LLM Answer (truyền cả chat_history)
        answer = self.llm.generate(intent, rewritten_query, context, chat_history=chat_history_str)

        # Lưu vào Redis
        if self.memory and session_id:
            self.memory.save(session_id, query, answer)

        return {
            'intent': intent,
            'query': query,
            'documents': reranked_results,
            'answer': answer,
        }

    def stream(self, query: str, session_id: str = None):
        """Pipeline RAG with output streaming"""
        # Lấy memory
        messages, chat_history_str = self._get_memory(session_id)

        # Router
        intent = self.router_model.classify(query)

        # Rewrite query nếu cần
        rewritten_query = self._rewrite_if_needed(query, chat_history_str, intent)

        answer_buffer = []

        try:
            # Chatchit flow
            if intent == "chitchat":
                for token in self.llm.stream(intent, query, context=None, chat_history=chat_history_str):
                    answer_buffer.append(token)
                    yield token
                return

            # RAG stream
            results = self.retrieval(rewritten_query, intent)

            if results is None:
                for token in self.llm.stream(intent, rewritten_query, context="", chat_history=chat_history_str):
                    answer_buffer.append(token)
                    yield token
                return

            reranked_results = self.rerank(rewritten_query, results, top_n=5)
            context = self.build_context(reranked_results)

            for token in self.llm.stream(intent, rewritten_query, context, chat_history=chat_history_str):
                answer_buffer.append(token)
                yield token

        finally:
            # Lưu vào Redis sau khi stream xong
            if self.memory and session_id and answer_buffer:
                full_answer = "".join(answer_buffer)
                self.memory.save(session_id, query, full_answer)

    def run_eval_batch(self, query: str):
        """
        Phiên bản rút gọn dành riêng cho Evaluation:
        - Không lưu Memory/Redis
        - Không dùng Chat History
        - Trả về định dạng chuẩn để Eval
        """
        try:
            # 1. Router: Xác định ý định
            intent = self.router_model.classify(query)

            # 2. Retrieval: Tìm kiếm dữ liệu
            results = self.retrieval(query, intent)

            if results is None or len(results) == 0:
                # Nếu không tìm thấy docs (chitchat hoặc lỗi search)
                answer = self.llm.generate(intent, query, context="", chat_history="")
                return {
                    "question": query,
                    "answer": answer,
                    "contexts": [],  # Trả về list rỗng nếu không có docs
                    "intent": intent
                }

            # 3. Rerank: Sắp xếp lại
            reranked_results = self.rerank(query, results, top_n=5)

            # 4. Build Context & Get Raw Chunks: Ragas cần list các chuỗi văn bản gốc
            raw_contexts = [
                item['point'].payload.get("text_answer", "")
                for item in reranked_results
            ]

            # Context gộp để nạp vào Prompt cho LLM
            context_for_llm = self.build_context(reranked_results)

            # 5. LLM Generate Answer
            answer = self.llm.generate(intent, query, context_for_llm, chat_history="")

            return {
                "question": query,
                "answer": answer,
                "contexts": raw_contexts,  # List of strings chuẩn Ragas
                "intent": intent
            }
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            return None

def main():
    # --- Bước 1: Khởi tạo Pipeline ---
    print("Khởi tạo model LLM")
    llm = LLM(api_key=api_groq_key, model="llama-3.3-70b-versatile", max_tokens=512)

    print("Khởi tạo router")
    router = IdentityClassificationLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    print("Khởi tạo embedding model")
    embedding = EmbeddingModel(model_name="BAAI/bge-m3")
    embedding.load_model()

    print("Khởi tạo QdrantDB")
    qdrant = QdrantDB(host="localhost", port=6333)

    print("Khởi tạo Reranker")
    reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")

    print("Khởi tạo pipeline")
    pipeline = RAGPipeLine(
        llm=llm,
        embedding_model=embedding,
        qdrant_client=qdrant,
        router=router,
        reranker=reranker,
        memory_manager=None,
        product_collection="products",
        policy_collection="policies",
    )

    # --- Bước 2: Đọc file 50 câu hỏi ---
    input_txt = "evaluation/questions.txt"  # File của bạn
    output_csv = "rag_evaluation_dataset.csv"

    if not os.path.exists(input_txt):
        print(f"❌ Không tìm thấy file {input_txt}")
        return

    with open(input_txt, "r", encoding="utf-8") as f:
        # Đọc từng dòng, bỏ khoảng trắng thừa và loại bỏ dòng trống
        questions = [line.strip() for line in f if line.strip()]

    final_eval_results = []
    print(f"🚀 Đã tìm thấy {len(questions)} câu hỏi. Bắt đầu xử lý...")

    print(f"🚀 Đang chạy đánh giá cho {len(questions)} câu hỏi...")

    # --- Bước 3: Chạy vòng lặp tạo dữ liệu ---
    for q in tqdm(questions):
        res = pipeline.run_eval_batch(q)
        if res:
            final_eval_results.append(res)

        # Nghỉ 1s để tránh Rate Limit của Groq
        time.sleep(1)

    # --- Bước 4: Lưu kết quả ---
    df_output = pd.DataFrame(final_eval_results)

    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ Đã lưu file đánh giá tại: {output_csv}")


if __name__ == "__main__":
    main()









