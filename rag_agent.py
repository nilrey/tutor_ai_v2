"""
History AI Tutor - Modern LangChain 2026
С работающим RAG, памятью через RunnableWithMessageHistory
"""

import os
import warnings
from operator import itemgetter

# Подавляем deprecation warnings (для чистоты вывода)
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Core imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Standalone packages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ========== 1. Конфигурация модели ==========
def get_llm():
    """Возвращает LLM для генерации ответов"""
    return ChatOllama(model="qwen3:8b", temperature=0.5)


def get_embeddings():
    """Возвращает модель для эмбеддингов"""
    return OllamaEmbeddings(model="nomic-embed-text")


# ========== 2. Загрузка документов ==========
def load_documents(file_path: str = "data/history.txt"):
    """Загружает тексты и разбивает на чанки"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": file_path}) 
                 for chunk in chunks]
    
    print(f"Загружено {len(documents)} чанков")
    return documents


# ========== 3. Векторная БД ==========
def create_vectorstore(documents, persist_directory="./chroma_db"):
    """Создает или загружает векторную БД"""
    embeddings = get_embeddings()
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"Загружена существующая БД")
    else:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"Создана новая БД")
    
    return vectorstore


# ========== 4. Хранилище сессий для памяти ==========
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Возвращает историю чата для конкретной сессии"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ========== 5. Создание цепочки с RAG и памятью (ГЛАВНОЕ) ==========
def create_rag_chat_chain(retriever, llm):
    """
    Создает цепочку чата с RAG и памятью
    - RAG: поиск релевантных документов из векторной БД
    - Память: сохранение истории диалога
    """
    
    # Промпт с контекстом из RAG
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты - эксперт по истории. Отвечай на вопрос, используя ТОЛЬКО контекст.
Если контекст не содержит ответа, скажи "В предоставленных документах нет этой информации".

Контекст из учебника:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Функция для получения контекста из ретривера
    def get_context(input_data):
        # input_data может быть строкой или dict
        query = input_data if isinstance(input_data, str) else input_data.get("input", "")
        docs = retriever.invoke(query)
        
        # Отладочный вывод (показывает, что RAG работает)
        print(f"\n[RAG] Найдено {len(docs)} релевантных документов:")
        for i, doc in enumerate(docs):
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"   {i+1}. {preview}...")
        
        context = "\n\n".join(d.page_content for d in docs)
        if not context:
            context = "Контекст не найден."
        return context
    
    # Собираем RAG цепочку
    rag_chain = (
        RunnablePassthrough.assign(
            context=get_context
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Оборачиваем в поддержку памяти
    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_memory


# ========== 6. Создание простого чата (без RAG, только память) ==========
def create_simple_chat_chain(llm):
    """
    Альтернативная цепочка: только память, без RAG
    """
    
    simple_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты - эксперт по истории. Отвечай на вопросы дружелюбно и по делу."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    simple_chain = simple_prompt | llm | StrOutputParser()
    
    chain_with_memory = RunnableWithMessageHistory(
        simple_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_memory


# ========== 7. Инициализация системы ==========
def init_rag_system(use_rag: bool = True, force_reload: bool = False):
    """
    Инициализирует RAG систему:
    - загружает документы
    - создает векторную БД
    - возвращает чат цепочку
    
    Args:
        use_rag: использовать ли RAG (True) или только память (False)
        force_reload: пересоздать векторную БД (игнорировать существующую)
    """
    print("History AI Tutor (LangChain 2026)")
    print("=" * 50)
    
    # Создаем папку data если нет
    os.makedirs("data", exist_ok=True)
    
    # Проверяем наличие файла с данными
    if not os.path.exists("data/history.txt"):
        print("Файл data/history.txt не найден. Создаю пример...")
        with open("data/history.txt", "w", encoding="utf-8") as f:
            f.write("""
Великая Отечественная война началась 22 июня 1941 года.
Сталинградская битва началась 17 июля 1942 года.
Сталинградская битва закончилась 2 февраля 1943 года.
Курская битва произошла в июле-августе 1943 года.
Берлин был взят советскими войсками 2 мая 1945 года.
Первая мировая война началась в 1914 году и закончилась в 1918 году.
Александр Невский победил шведов в Невской битве в 1240 году.
Куликовская битва произошла в 1380 году.
День Победы отмечается 9 мая.
""")
        print("Создан пример файла data/history.txt")
    
    # Загружаем документы
    documents = load_documents()
    
    # Создаем векторную БД
    if force_reload and os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")
        print("Удалена старая БД")
    
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Создаем LLM
    llm = get_llm()
    
    # Создаем чат цепочку
    if use_rag:
        print("Режим: RAG + память")
        chat_chain = create_rag_chat_chain(retriever, llm)
    else:
        print("Режим: только память (без RAG)")
        chat_chain = create_simple_chat_chain(llm)
    
    print("Готово! Задавайте вопросы.")
    print("=" * 50)
    
    return chat_chain


# ========== 8. Запуск диалога ==========
def run_dialog(chat_chain, session_id: str = "user_1"):
    """Запускает интерактивный диалог"""
    
    print("\nВопросы пишите после 'Вы: '")
    print("Команды: /exit - выход, /new - новый диалог (очистить память)")
    print("=" * 50)
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["/exit", "/quit", "exit", "quit", "выход"]:
            print("До свидания!")
            break
        
        if user_input.lower() in ["/new", "new"]:
            # Очищаем память для этой сессии
            if session_id in store:
                store[session_id] = InMemoryChatMessageHistory()
            print("Память очищена! Начинаем новый диалог.")
            continue
        
        try:
            response = chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nTutor: {response}")
        except Exception as e:
            print(f"\nОшибка: {e}")
            print("Попробуйте переформулировать вопрос или запустите /new")


# ========== 9. Точка входа (если запускаем этот файл напрямую) ==========
if __name__ == "__main__":
    # При запуске rag_agent.py напрямую
    chat_chain = init_rag_system(use_rag=True, force_reload=False)
    run_dialog(chat_chain)