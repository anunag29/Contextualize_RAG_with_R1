import os
from dotenv import load_dotenv
load_dotenv()

from src import *

def sync_data(data_dir: str = os.getenv("DATA_DIR","./data")):
    print("Syncing Data ....")
    #feteching list of documents which are already processed
    stored_docs = []
    with open(os.getenv("STORED_DOCS_TXT"), 'r') as file:
        for file_name in file:
            stored_docs.append(file_name.strip())

    try:
        vector_store = load_vector_store(os.getenv("FAISS_INDEX_DIR"))
    except Exception as e:
        vector_store = None
        print(f"ERROR : {e}")

    try:
        bm25_index, tokenized_chunks = load_bm25_index(os.getenv("BM25_INDEX_PKL"))
    except Exception as e:
        bm25_index, tokenized_chunks = None, None
        print(f"ERROR : {e}")

    new_docs = load_pdf_documents(data_dir, stored_docs)
    contextualized_chunks = []

    with open(os.getenv("STORED_DOCS_TXT"), 'a') as file:
        for doc in new_docs:
            _, new_contextualized_chunks = process_document(document=doc["content"])
            contextualized_chunks += new_contextualized_chunks
            file.write(doc["filename"] + '\n')

    if contextualized_chunks:
        vector_store = add_to_vectorstore(new_chunks=contextualized_chunks , vector_store=vector_store, save_path=os.getenv("FAISS_INDEX_DIR"))
        bm25_index, tokenized_chunks = add_new_chunks_to_bm25(new_chunks=contextualized_chunks , existing_tokenized_chunks=tokenized_chunks, save_path=os.getenv("BM25_INDEX_PKL"))
        print("Successfully synced new data.")
        return vector_store, bm25_index, tokenized_chunks

    print("Successfully synced data.")
    return vector_store, bm25_index, tokenized_chunks


if __name__ == '__main__':
    vector_store, bm25_index, tokenized_chunks = sync_data()

    retreival(vector_store)