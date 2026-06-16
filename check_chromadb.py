import chromadb

persist_directory = "./chroma_db"  # ваш путь
client = chromadb.PersistentClient(path=persist_directory)

# Получаем все коллекции
collections = client.list_collections()

for collection in collections:
    print(f"\n{'=' * 50}")
    print(f"Имя коллекции: {collection.name}")
    print(f"Количество документов: {collection.count()}")

    # Получаем первые 5 документов
    results = collection.get(limit=5)

    if results["documents"]:
        for i, doc in enumerate(results["documents"], 1):
            print(f"\n  [{i}] {doc[:150]}...")
            if results["metadatas"]:
                print(f"      Метаданные: {results['metadatas'][i - 1]}")
    else:
        print("  Документы не найдены или хранятся только векторы")
