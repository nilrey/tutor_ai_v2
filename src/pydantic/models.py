'''
Pydantic модели данных
'''

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class AddDocumentRequest(BaseModel):
    """Запрос на добавление документа"""
    content: str = Field(..., description="Текст документа (можно с переносами строк)")
    metadata: Optional[dict] = Field(None, description="Метаданные документа")

class QuestionRequest(BaseModel):
    """Запрос на вопрос"""
    question: str = Field(..., min_length=1, max_length=1000, description="Вопрос пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии (если не указан, создается новый)")
    stream: bool = Field(False, description="Стриминг ответа (пока не реализован)")

class QuestionResponse(BaseModel):
    """Ответ на вопрос"""
    session_id: str
    question: str
    answer: str
    timestamp: str
    rag_used: bool = True
    retrieved_docs_count: int = 0
    processing_time: str = "0.0 сек"

class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    version: str
    model: str
    rag_enabled: bool
    vector_db_size: int
    active_sessions: int

class StatResponse(BaseModel):
    """Статистика сервиса"""
    total_requests: int
    avg_response_time_ms: float
    last_minute_requests: int
    requests_by_hour: Dict[str, int]
    sessions_stats: Dict[str, Any]

class SessionInfo(BaseModel):
    """Информация о сессии"""
    session_id: str
    message_count: int
    created_at: str
    last_active: str