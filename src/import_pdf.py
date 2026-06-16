"""Импорт PDF файла в векторную БД Chroma с промежуточным сохранением текста"""

import os
import re
import sys
import warnings
from datetime import datetime
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Пробуем разные библиотеки для PDF
try:
    import pdfplumber

    USE_PDFPLUMBER = True
    print("Используется pdfplumber (лучшее извлечение текста)")
except ImportError:
    try:
        USE_PDFPLUMBER = False
        print("Используется pypdf")
    except ImportError:
        print("Ошибка: установите pdfplumber или pypdf: pip install pdfplumber")
        sys.exit(1)


def clean_text(text: str) -> str:
    """
    Очистка текста от мусорных символов и нормализация

    Args:
        text: Исходный текст из PDF

    Returns:
        Очищенный текст
    """
    if not text:
        return ""

    # Удаляем управляющие символы, оставляя только читаемые
    # \x00-\x1F - управляющие символы, кроме \n (0x0A) и \t (0x09)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # Замена множественных пробелов на один
    cleaned = re.sub(r" +", " ", cleaned)

    # Замена множественных переносов строк на двойные
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Удаление пробелов в начале строк
    cleaned = re.sub(r"\n +", "\n", cleaned)

    # Удаление пустых строк в начале и конце
    cleaned = cleaned.strip()

    return cleaned


def extract_text_with_pdfplumber(pdf_path: str) -> Tuple[str, List[dict]]:
    """Извлекает текст с помощью pdfplumber (лучшее качество)"""
    page_info = []
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                text = page.extract_text()
                if text:
                    # Получаем статистику страницы
                    chars_before = len(text)
                    cleaned = clean_text(text)
                    chars_after = len(cleaned)

                    page_info.append(
                        {
                            "page": page_num,
                            "chars_before": chars_before,
                            "chars_after": chars_after,
                            "removed": chars_before - chars_after,
                        }
                    )

                    full_text.append(cleaned)
                    print(
                        f"  Стр. {page_num}: {chars_before} -> {chars_after} символов (удалено {chars_before - chars_after})"
                    )
                else:
                    page_info.append(
                        {
                            "page": page_num,
                            "chars_before": 0,
                            "chars_after": 0,
                            "removed": 0,
                        }
                    )
                    print(f"  Стр. {page_num}: пустая")
            except Exception as e:
                print(f"  Ошибка на стр. {page_num}: {e}")
                continue

    result = "\n\n".join(full_text)
    return result, page_info


def extract_text_with_pypdf(pdf_path: str) -> Tuple[str, List[dict]]:
    """Извлекает текст с помощью pypdf (альтернатива)"""
    reader = PdfReader(pdf_path)
    page_info = []
    full_text = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text()
            if text:
                chars_before = len(text)
                cleaned = clean_text(text)
                chars_after = len(cleaned)

                page_info.append(
                    {
                        "page": page_num,
                        "chars_before": chars_before,
                        "chars_after": chars_after,
                        "removed": chars_before - chars_after,
                    }
                )

                full_text.append(cleaned)
                print(
                    f"  Стр. {page_num}: {chars_before} -> {chars_after} символов (удалено {chars_before - chars_after})"
                )
            else:
                page_info.append(
                    {
                        "page": page_num,
                        "chars_before": 0,
                        "chars_after": 0,
                        "removed": 0,
                    }
                )
                print(f"  Стр. {page_num}: пустая")
        except Exception as e:
            print(f"  Ошибка на стр. {page_num}: {e}")
            continue

    result = "\n\n".join(full_text)
    return result, page_info


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[dict]]:
    """Извлекает текст из PDF с выбором оптимального метода"""
    print(f"Чтение PDF: {pdf_path}")

    if USE_PDFPLUMBER:
        return extract_text_with_pdfplumber(pdf_path)
    else:
        return extract_text_with_pypdf(pdf_path)


def save_intermediate_text(
    text: str, page_info: List[dict], output_dir: str = "tmp"
) -> str:
    """
    Сохраняет очищенный текст в промежуточный файл

    Args:
        text: Очищенный текст
        page_info: Информация о страницах
        output_dir: Директория для сохранения

    Returns:
        Путь к сохраненному файлу
    """
    # Создаем директорию если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Генерируем имя файла с датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"tmp_{timestamp}.txt")

    # Сохраняем текст и метаинформацию
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ИМПОРТ PDF: Гай_Юлий_Цезарь.pdf\n")
        f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Метод извлечения: {'pdfplumber' if USE_PDFPLUMBER else 'pypdf'}\n")
        f.write("=" * 80 + "\n\n")

        # Статистика по страницам
        f.write("СТАТИСТИКА ПО СТРАНИЦАМ:\n")
        f.write("-" * 40 + "\n")
        total_chars_before = sum(p["chars_before"] for p in page_info)
        total_chars_after = sum(p["chars_after"] for p in page_info)

        for info in page_info:
            if info["chars_before"] > 0:
                f.write(
                    f"Стр. {info['page']:3d}: {info['chars_before']:6d} -> {info['chars_after']:6d} символов (удалено {info['removed']:4d})\n"
                )

        f.write("-" * 40 + "\n")
        f.write(
            f"ИТОГО: {total_chars_before:6d} -> {total_chars_after:6d} символов (удалено {total_chars_before - total_chars_after})\n"
        )
        f.write(
            f"Удалено %: {(total_chars_before - total_chars_after) / total_chars_before * 100:.1f}%\n\n"
        )

        # Статистика по чанкам (будет заполнена позже)
        f.write("=" * 80 + "\n")
        f.write("ОЧИЩЕННЫЙ ТЕКСТ (первые 2000 символов):\n")
        f.write("=" * 80 + "\n\n")
        f.write(text[:2000])

        if len(text) > 2000:
            f.write("\n\n... (текст обрезан, полный текст ниже) ...\n\n")

        f.write("=" * 80 + "\n")
        f.write("ПОЛНЫЙ ОЧИЩЕННЫЙ ТЕКСТ:\n")
        f.write("=" * 80 + "\n\n")
        f.write(text)

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("КОНЕЦ ФАЙЛА\n")

    print(f"\nПромежуточный текст сохранен в: {output_file}")
    return output_file


def chunk_documents(
    text: str, chunk_size: int = 800, chunk_overlap: int = 100
) -> List[Document]:
    """Разбивает текст на чанки с метаданными"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        # Дополнительная очистка каждого чанка
        clean_chunk = clean_text(chunk)

        if len(clean_chunk.strip()) < 20:  # Пропускаем слишком короткие чанки
            continue

        doc = Document(
            page_content=clean_chunk,
            metadata={
                "source": "Гай_Юлий_Цезарь.pdf",
                "chunk_id": i,
                "topic": "Древний Рим",
                "person": "Юлий Цезарь",
                "century": "I век до н.э.",
                "char_length": len(clean_chunk),
                "import_date": datetime.now().isoformat(),
            },
        )
        documents.append(doc)

    print(
        f"Создано {len(documents)} чанков (пропущено {len(chunks) - len(documents)} пустых/коротких)"
    )
    return documents


def save_chunks_info(
    documents: List[Document], output_dir: str = "tmp", original_text_file: str = None
):
    """Сохраняет информацию о чанках для верификации"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunks_file = os.path.join(output_dir, f"chunks_{timestamp}.txt")

    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ИНФОРМАЦИЯ О ЧАНКАХ\n")
        f.write(f"Дата обработки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for i, doc in enumerate(documents):
            f.write(f"\n{'=' * 40}\n")
            f.write(f"ЧАНК #{doc.metadata['chunk_id']}\n")
            f.write(f"Длина: {doc.metadata['char_length']} символов\n")
            f.write(f"Источник: {doc.metadata['source']}\n")
            f.write(f"{'-' * 40}\n")
            f.write(doc.page_content)
            f.write(f"\n{'=' * 40}\n")

    print(f"Информация о чанках сохранена в: {chunks_file}")


def add_to_vectorstore(
    documents: List[Document], persist_directory: str = "./chroma_db"
):
    """Добавляет документы в существующую векторную БД"""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Загружаем существующую БД
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
        print("Загружена существующая БД")

        # Получаем старую статистику
        old_docs = vectorstore.get()
        old_count = len(old_docs["ids"])

        # Добавляем новые документы
        vectorstore.add_documents(documents)
        print(f"Добавлено {len(documents)} документов")

        # Новая статистика
        new_docs = vectorstore.get()
        new_count = len(new_docs["ids"])
        print(f"Было: {old_count}, стало: {new_count}")
    else:
        # Создаем новую БД
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print(f"Создана новая БД с {len(documents)} документами")

    return vectorstore


def verify_stored_documents():
    """Проверяет, что документы сохранены корректно"""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # Получаем все документы
    all_docs = vectorstore.get()

    # Фильтруем документы о Цезаре
    caesar_docs = []
    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and metadata.get("source") == "Гай_Юлий_Цезарь.pdf":
            caesar_docs.append(
                {
                    "id": all_docs["ids"][i],
                    "content_preview": all_docs["documents"][i][:150],
                    "metadata": metadata,
                }
            )

    print("\n=== ВЕРИФИКАЦИЯ ===")
    print(f"Всего документов в БД: {len(all_docs['ids'])}")
    print(f"Документов о Цезаре: {len(caesar_docs)}")

    if caesar_docs:
        print("\nПример документа о Цезаре (первые 3):")
        for i, doc in enumerate(caesar_docs[:3]):
            print(f"\n  [{i + 1}] ID: {doc['id']}")
            print(f"      Метаданные: {doc['metadata']}")
            print(f"      Содержание: {doc['content_preview']}...")

    return caesar_docs


if __name__ == "__main__":
    # Путь к PDF файлу
    pdf_path = "./raw/cezar.pdf"

    if not os.path.exists(pdf_path):
        print(f"Ошибка: файл {pdf_path} не найден!")
        print("Убедитесь, что файл находится в текущей директории")
        sys.exit(1)

    print("=" * 60)
    print("ИМПОРТ PDF О ЮЛИИ ЦЕЗАРЕ В ВЕКТОРНУЮ БД")
    print("=" * 60)

    # 1. Извлечение текста из PDF
    print("\n[1/5] Извлечение текста из PDF...")
    text, page_info = extract_text_from_pdf(pdf_path)

    if not text:
        print("Ошибка: не удалось извлечь текст из PDF")
        sys.exit(1)

    # 2. Сохранение промежуточного текста для проверки
    print("\n[2/5] Сохранение очищенного текста...")
    temp_file = save_intermediate_text(text, page_info)

    # 3. Разбиение на чанки
    print("\n[3/5] Разбиение текста на чанки...")
    documents = chunk_documents(text, chunk_size=800, chunk_overlap=100)

    if not documents:
        print("Ошибка: не создано ни одного чанка")
        sys.exit(1)

    # 4. Сохранение информации о чанках
    print("\n[4/5] Сохранение информации о чанках...")
    save_chunks_info(documents)

    # 5. Добавление в векторную БД
    print("\n[5/5] Добавление в векторную БД...")
    vectorstore = add_to_vectorstore(documents)

    # 6. Проверка
    caesar_docs = verify_stored_documents()

    print("\n" + "=" * 60)
    print("✅ ИМПОРТ УСПЕШНО ЗАВЕРШЕН!")
    print(f"📄 Промежуточный файл: {temp_file}")
    print(f"📊 Всего чанков: {len(documents)}")
    print("💾 БД: ./chroma_db")
    print("=" * 60)
    print("\nТеперь вы можете перезагрузить API через GET /reload")
    print("Или перезапустите сервер для использования новых данных")
