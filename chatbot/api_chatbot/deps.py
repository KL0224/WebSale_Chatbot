from dotenv import load_dotenv
import os
from prepare_database.embedding import EmbeddingModel
from reranking.rerank import Reranker
from prepare_database.qdrant import QdrantDB
from router.identity_classification import IdentityClassificationLLM
from api_chatbot.pipeline import LLM, RAGPipeLine
from history.memory import RedisMemoryManager
from history.history import MongoHistoryManager
from pathlib import Path

# Load env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_groq_key = os.getenv("API_GROQ_KEY")
# Global singletons
pipeline = None
mongo_history = None

def init_mongo():
    global mongo_history
    if mongo_history is None:
        print("Khởi tạo MongoDB History")
        mongo_history = MongoHistoryManager(uri="mongodb://root:example@localhost:27017", db_name="chatbot")
        print("MongoDB History đã khởi tạo thành công")
    return mongo_history

def init_pipeline():
    global pipeline
    if pipeline is None:
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

        print("Khởi tạo Redis Memory")
        memory = RedisMemoryManager(host="localhost", port=6379, ttl=3600)

        print("Khởi tạo pipeline")
        pipeline = RAGPipeLine(
            llm=llm,
            embedding_model=embedding,
            qdrant_client=qdrant,
            router=router,
            reranker=reranker,
            memory_manager=memory,
            product_collection="products",
            policy_collection="policies",
        )

        print("Pipeline đã khởi tạo thành công")

    return pipeline