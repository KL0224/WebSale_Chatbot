from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING


class MongoHistoryManager:
    def __init__(self, uri: str = "mongodb://root:example@localhost:27017",
                 db_name: str = "chatbot"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["conversations"]
        # Index
        self.collection.create_index("session_id", unique=True)
        self.collection.create_index([("user_id", 1), ("updated_at", -1)])

    def save_message(self, session_id: str, query: str, answer: str,
                     intent: str = "unknown", user_id: str = None):
        """Lưu 1 cặp (query, answer) vào conversation. Tạo mới nếu chưa có."""
        if not session_id:
            return

        now = datetime.now(timezone.utc)

        user_msg = {
            "role": "user",
            "content": query,
            "intent": intent,
            "timestamp": now,
        }
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "timestamp": now,
        }

        # Upsert: tạo mới nếu chưa có, append messages nếu đã có
        self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {"$each": [user_msg, assistant_msg]}
                },
                "$set": {
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "title": query[:80] if query else "Cuộc trò chuyện mới",
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def get_conversations(self, user_id: str = None, limit: int = 50, skip: int = 0):
        """Lấy danh sách conversations, lọc theo user_id nếu có"""
        query_filter = {}
        if user_id:
            query_filter["user_id"] = user_id

        cursor = self.collection.find(
            query_filter,
            {"session_id": 1, "title": 1, "updated_at": 1, "_id": 0}
        ).sort("updated_at", DESCENDING).skip(skip).limit(limit)
        return list(cursor)

    def get_conversation(self, session_id: str, user_id: str = None):
        """Lấy chi tiết 1 conversation, verify user ownership nếu có user_id"""
        query_filter = {"session_id": session_id}
        if user_id:
            query_filter["user_id"] = user_id
        return self.collection.find_one(query_filter, {"_id": 0})

    def delete_conversation(self, session_id: str, user_id: str = None):
        """Xoá 1 conversation"""
        query_filter = {"session_id": session_id}
        if user_id:
            query_filter["user_id"] = user_id
        result = self.collection.delete_one(query_filter)
        return result.deleted_count > 0

    def delete_all_conversations(self, user_id: str):
        """Xoá toàn bộ conversations của 1 user"""
        result = self.collection.delete_many({"user_id": user_id})
        return result.deleted_count
