import redis
import json
from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict, messages_from_dict


class RedisMemoryManager:
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl  # Thời gian sống của session (1 tiếng)

    def get_history(self, session_id: str, limit: int = 6):
        if not session_id: return []
        key = f"chat_history:{session_id}"
        # Lấy list tin nhắn từ Redis
        raw_data = self.redis.lrange(key, -limit, -1)
        if not raw_data: return []

        # Chuyển đổi từ JSON string sang LangChain Message objects
        dicts = [json.loads(d) for d in raw_data]
        return messages_from_dict(dicts)

    def save(self, session_id: str, query: str, answer: str):
        if not session_id: return
        key = f"chat_history:{session_id}"
        # Tạo objects
        msgs = [HumanMessage(content=query), AIMessage(content=answer)]
        # Chuyển sang dict để lưu JSON
        for msg in messages_to_dict(msgs):
            self.redis.rpush(key, json.dumps(msg))

        self.redis.expire(key, self.ttl)