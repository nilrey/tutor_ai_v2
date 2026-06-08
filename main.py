"""
History AI Tutor - FastAPI сервис
Endpoints:
- POST /ask      - задать вопрос
- GET  /health   - проверка здоровья сервиса
- GET  /stat     - статистика запросов
- GET  /sessions - список активных сессий
- DELETE /session/{session_id} - очистить память сессии
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Импорты из rag_agent (предполагаем, что rag_agent.py в той же папке)
from rag_agent import (
    init_rag_system, 
    run_dialog, 
    get_session_history, 
    store
)

# ========== 1. Модели данных (Pydantic) ==========

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

# ========== 2. Создание FastAPI приложения ==========

app = FastAPI(
    title="History AI Tutor API",
    description="RAG-агент по истории с LangChain и Ollama",
    version="2.0.0"
)

# CORS (для фронтенда)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 3. Глобальное состояние ==========

# Статистика запросов
request_stats = {
    "total_requests": 0,
    "response_times": [],  # список последних времен ответа (для среднего)
    "last_requests": [],    # кортежи (timestamp, session_id)
    "hourly_counts": defaultdict(int),  # счетчики по часам
    "sessions_stats": defaultdict(lambda: {"count": 0, "first_seen": None, "last_seen": None})
}

# RAG система (инициализируется при старте)
rag_system = None

# ========== 4. Инициализация и shutdown ==========

@app.on_event("startup")
async def startup_event():
    """Загружает RAG систему при старте сервера"""
    global rag_system
    print("Запуск History AI Tutor API...")
    
    # Инициализируем RAG систему (один раз при старте)
    rag_system = init_rag_system(use_rag=True, force_reload=False)
    
    # Сохраняем модель в конфиг для health check
    from rag_agent import get_llm
    app.state.model_name = "qwen3:8b"
    app.state.rag_enabled = True
    app.state.vector_db_size = count_vector_db_documents()
    
    print("API готов к работе")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при выключении"""
    global request_stats
    print("Остановка API...")


# ========== 5. Вспомогательные функции ==========

def count_vector_db_documents() -> int:
    """Возвращает количество документов в векторной БД"""
    try:
        from rag_agent import Chroma, get_embeddings
        import os
        if os.path.exists("./chroma_db"):
            embeddings = get_embeddings()
            vstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            # Приблизительное количество (Chroma не хранит count напрямую)
            return len(vstore.get()["ids"])
        return 0
    except Exception as e:
        print(f"Ошибка подсчета документов: {e}")
        return -1

def update_stats(session_id: str, response_time_ms: float):
    """Обновляет статистику запросов"""
    current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
    
    request_stats["total_requests"] += 1
    request_stats["response_times"].append(response_time_ms)
    
    # Храним последние 1000 времен ответа
    if len(request_stats["response_times"]) > 1000:
        request_stats["response_times"] = request_stats["response_times"][-1000:]
    
    # Последние запросы (для last_minute)
    request_stats["last_requests"].append((time.time(), session_id))
    request_stats["last_requests"] = request_stats["last_requests"][-100:]
    
    # Почасовая статистика
    request_stats["hourly_counts"][current_hour] += 1
    
    # Статистика по сессиям
    now = datetime.now().isoformat()
    stats = request_stats["sessions_stats"][session_id]
    stats["count"] += 1
    if stats["first_seen"] is None:
        stats["first_seen"] = now
    stats["last_seen"] = now

def get_last_minute_requests() -> int:
    """Возвращает количество запросов за последнюю минуту"""
    now = time.time()
    one_minute_ago = now - 60
    return sum(1 for ts, _ in request_stats["last_requests"] if ts > one_minute_ago)


# ========== 6. Endpoints ==========

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Задать вопрос боту по истории.
    
    - **question**: текст вопроса (обязательно)
    - **session_id**: ID для сохранения контекста диалога (если не указан, создается новый)
    - **stream**: стриминг ответа (пока в разработке)
    """
    global rag_system
    
    start_time = time.time()
    
    # Генерируем session_id если не передан
    session_id = request.session_id or str(uuid.uuid4())
    
    # Проверяем, что RAG система загружена
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG система не инициализирована")
    
    try:
        # Вызываем RAG цепочку
        response = rag_system.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Получаем ответ (в зависимости от формата rag_agent)
        if isinstance(response, dict):
            answer = response.get("output", response.get("answer", str(response)))
        else:
            answer = str(response)
        
        # Подсчет времени
        response_time_ms = (time.time() - start_time) * 1000
        
        # Обновляем статистику
        update_stats(session_id, response_time_ms)
        
        # Сколько документов было найдено? (можно расширить)
        retrieved_docs_count = 0  # TODO: можно передавать из rag_agent

        processing_time_sec = time.time() - start_time
        
        return QuestionResponse(
            session_id=session_id,
            question=request.question,
            answer=answer,
            timestamp=datetime.now().isoformat(),
            rag_used=app.state.rag_enabled,
            retrieved_docs_count=retrieved_docs_count,
            processing_time=f"{round(processing_time_sec, 1)} сек" 
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


@app.post("/add_document")
async def add_document(request: AddDocumentRequest):
    """
    Добавить новый документ в векторную БД (без перезагрузки всего сервера)
    """
    global rag_system
    
    try:
        from rag_agent import get_embeddings, Chroma
        
        # Загружаем существующую БД
        embeddings = get_embeddings()
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        # Создаем новый документ
        new_doc = Document(
            page_content=request.content,
            metadata=request.metadata or {"source": "api", "added_at": datetime.now().isoformat()}
        )
        
        # Добавляем в БД
        vectorstore.add_documents([new_doc])
        # vectorstore.persist()
        
        # Пересоздаем RAG цепочку с обновленным ретривером
        from rag_agent import init_rag_system
        rag_system = init_rag_system(use_rag=True, force_reload=False)
        
        return {
            "status": "ok",
            "message": "Документ добавлен",
            "content_preview": request.content[:100]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка работоспособности сервиса.
    Используется для мониторинга (Kubernetes liveness/readiness probe).
    """
    global rag_system
    
    # Подсчет активных сессий
    active_sessions = len(store)
    
    # Проверка Ollama (опционально)
    ollama_healthy = True
    try:
        from langchain_ollama import ChatOllama
        test_llm = ChatOllama(model="qwen3:8b", temperature=0)
        test_llm.invoke("ping")  # проверка доступности
    except Exception:
        ollama_healthy = False
    
    status = "healthy" if (rag_system is not None and ollama_healthy) else "degraded"
    
    return HealthResponse(
        status=status,
        version="2.0.0",
        model=app.state.model_name,
        rag_enabled=app.state.rag_enabled,
        vector_db_size=count_vector_db_documents(),
        active_sessions=active_sessions
    )


@app.get("/stat", response_model=StatResponse)
async def get_statistics():
    """
    Получить статистику запросов.
    - Общее количество запросов
    - Среднее время ответа
    - Количество запросов за последнюю минуту
    - Распределение по часам
    - Статистика по сессиям
    """
    # Среднее время ответа
    if request_stats["response_times"]:
        avg_time = sum(request_stats["response_times"]) / len(request_stats["response_times"])
    else:
        avg_time = 0.0
    
    # Преобразуем hourly_counts в обычный dict
    hourly_dict = dict(request_stats["hourly_counts"])
    
    # Статистика по сессиям
    sessions_info = {}
    for sid, stats in request_stats["sessions_stats"].items():
        sessions_info[sid] = {
            "requests": stats["count"],
            "first_seen": stats["first_seen"],
            "last_seen": stats["last_seen"]
        }
    
    return StatResponse(
        total_requests=request_stats["total_requests"],
        avg_response_time_ms=avg_time,
        last_minute_requests=get_last_minute_requests(),
        requests_by_hour=hourly_dict,
        sessions_stats={
            "total_sessions": len(request_stats["sessions_stats"]),
            "sessions_detail": sessions_info
        }
    )


@app.get("/sessions")
async def list_sessions():
    """
    Получить список всех активных сессий и их состояние.
    """
    sessions = []
    for session_id, history in store.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            message_count=len(history.messages) if hasattr(history, 'messages') else 0,
            created_at="unknown",  # можно расширить
            last_active=datetime.now().isoformat()  # приблизительно
        ))
    
    return {"sessions": [s.dict() for s in sessions], "total": len(sessions)}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Очистить память конкретной сессии.
    """
    if session_id in store:
        from rag_agent import InMemoryChatMessageHistory
        store[session_id] = InMemoryChatMessageHistory()
        return {"status": "ok", "message": f"Сессия {session_id} очищена"}
    else:
        raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")


@app.get("/reload")
async def reload_vector_db():
    """
    Перезагрузить векторную БД (при обновлении history.txt).
    Требует прав администратора (можно защитить токеном).
    """
    global rag_system
    
    try:
        from rag_agent import init_rag_system
        # Переинициализируем с force_reload=True
        rag_system = init_rag_system(use_rag=True, force_reload=True)
        app.state.vector_db_size = count_vector_db_documents()
        return {"status": "ok", "message": "Векторная БД перезагружена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перезагрузки: {str(e)}")


# ========== 7. Запуск (если файл запущен напрямую) ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # автоматическая перезагрузка при изменении кода
        log_level="info"
    )