"""
обработка входящего PDF, этап 1 - "выгрузка сырых данных"
"""

import argparse
import sys
from pathlib import Path

import pdfplumber


def load_pdf(path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
                if text:
                    parts.append(text)

            except Exception as e:
                print(f"Ошибка на странице {page.page_number}: {e}")
                continue

    return "\n".join(parts)


def count_pages(path) -> int:
    pages_counter = 0
    with pdfplumber.open(path) as pdf:
        pages_counter = len(pdf.pages)
    return pages_counter


def save_txt(pdf_path: Path, content) -> bool:
    try:
        txt_dir = Path("raw")
        txt_dir.mkdir(exist_ok=True)
        txt_name = f"raw_{pdf_path.stem}.txt"
        txt_path = txt_dir / txt_name
        with txt_path.open("w", encoding="utf-8") as txt:
            txt.write(content)
        return True

    except Exception as e:
        print(f"Ошибка записи в файл {txt_path}: {e}")
        return False


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", required=True)
    args = parser.parse_args()

    print("Путь к файлу:", args.pdf_path)

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"Файл не существует. '{pdf_path}'")
        sys.exit(1)

    pages_counter = count_pages(pdf_path)
    print("Количество страниц: " + str(pages_counter))

    if pages_counter > 0:
        content = load_pdf(pdf_path)
        if save_txt(pdf_path, content):
            print("Текст успешно сохранен.")

    print("Скрипт завершен.")
