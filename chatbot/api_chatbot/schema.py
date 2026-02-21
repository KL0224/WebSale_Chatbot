from pydantic import BaseModel
from typing import Optional, Any, List, Dict
from datetime import datetime

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class Documents(BaseModel):
    text: str
    score: float

class ChatResponse(BaseModel):
    intent: str
    query: str
    answer: str
    documents: Optional[List[Dict[str, Any]]] = None

# ======================== History Schemas ========================

class MessageItem(BaseModel):
    role: str
    content: str
    intent: Optional[str] = None
    timestamp: Optional[datetime] = None

class ConversationSummary(BaseModel):
    session_id: str
    title: Optional[str] = None
    updated_at: Optional[datetime] = None

class ConversationDetail(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[MessageItem] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

