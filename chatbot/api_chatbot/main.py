from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api_chatbot.schema import ChatRequest, ChatResponse, ConversationSummary, ConversationDetail
from api_chatbot.deps import init_pipeline, init_mongo
from typing import List

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    description="RAG: Router + Qdrant + Rerank + LLM + Memory + History",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
mongo_history = None

@app.on_event("startup")
def startup():
    global pipeline, mongo_history
    pipeline = init_pipeline()
    mongo_history = init_mongo()

@app.get('/health')
def health_check():
    return {'status': 'ok', 'service': 'RAG'}

# ======================== Chat Endpoints ========================

@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = pipeline.run(request.query, session_id=request.session_id)

        # Lưu vào MongoDB (long-term history)
        if mongo_history and request.session_id:
            mongo_history.save_message(
                session_id=request.session_id,
                query=request.query,
                answer=result['answer'],
                intent=result['intent'],
                user_id=request.user_id,
            )

        return ChatResponse(
            intent=result['intent'],
            query=result['query'],
            answer=result['answer'],
            documents=result['documents']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/stream')
def chat_stream(request: ChatRequest):
    def token_generator():
        answer_buffer = []
        for token in pipeline.stream(request.query, session_id=request.session_id):
            answer_buffer.append(token)
            # SSE format: multi-line data needs separate "data:" per line
            lines = token.split("\n")
            sse_data = "\n".join(f"data: {line}" for line in lines)
            yield sse_data + "\n\n"

        # Lưu vào MongoDB sau khi stream xong
        if mongo_history and request.session_id and answer_buffer:
            full_answer = "".join(answer_buffer)
            mongo_history.save_message(
                session_id=request.session_id,
                query=request.query,
                answer=full_answer,
                intent="stream",
                user_id=request.user_id,
            )

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
    )

# ======================== History Endpoints ========================

@app.get('/conversations', response_model=List[ConversationSummary])
def list_conversations(
    user_id: str = Query(None, description="User ID để lọc conversations"),
    limit: int = Query(default=50, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
):
    """Lấy danh sách conversations (phân trang, lọc theo user_id nếu có)"""
    try:
        conversations = mongo_history.get_conversations(user_id=user_id, limit=limit, skip=skip)
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/conversations/{session_id}', response_model=ConversationDetail)
def get_conversation(
    session_id: str,
    user_id: str = Query(None, description="User ID để verify ownership"),
):
    """Lấy chi tiết 1 conversation"""
    try:
        conversation = mongo_history.get_conversation(session_id, user_id=user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/conversations/{session_id}')
def delete_conversation(
    session_id: str,
    user_id: str = Query(None, description="User ID để verify ownership"),
):
    """Xoá 1 conversation"""
    try:
        deleted = mongo_history.delete_conversation(session_id, user_id=user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "ok", "message": f"Conversation {session_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


