from typing import List, Tuple

import pickle
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .models import embeddings


def add_to_vectorstore(new_chunks: List[Document], vector_store: FAISS = None, save_path: str = "../data/index/faiss"):
    """
    Add new chunks to an existing FAISS vector store and save the updated vector store.
    """
    if vector_store:
        vector_store.add_documents(new_chunks)
    else:
        vector_store = FAISS.from_documents(new_chunks, embeddings)

    # Save the updated vector store
    vector_store.save_local(save_path)
    print(f"Vector store saved at {save_path}")

    print("New chunks added and vector store updated.")
    return vector_store

def load_vector_store(load_path: str = "../data/index/faiss", embeddings=embeddings) -> FAISS:
    """
    Load the FAISS vector store from the disk.
    """
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded from {load_path}")
    return vector_store



def add_new_chunks_to_bm25(new_chunks: List[Document], existing_tokenized_chunks: List[List[str]] = None, save_path: str = "../data/index/bm25_index.pkl"):
    """
    Add new chunks to an existing BM25 index and save the updated index and tokenized chunks.
    """
    # Tokenize the new chunks
    tokenized_chunks = [chunk.page_content.split() for chunk in new_chunks]
    
    if existing_tokenized_chunks:
        # Combine existing tokenized chunks with the new ones
        tokenized_chunks = existing_tokenized_chunks + tokenized_chunks
    
    # Update the BM25 index by combining the existing and new tokenized chunks
    bm25_index = BM25Okapi(tokenized_chunks)
    
    ## Save both the BM25 index and tokenized chunks
    with open(save_path, 'wb') as f:
        pickle.dump((bm25_index, tokenized_chunks), f)
    print(f"BM25 index saved at {save_path}")
    
    print("New chunks added and BM25 index updated.")
    return bm25_index, tokenized_chunks

def load_bm25_index(load_path: str = "../data/index/bm25_index.pkl") -> Tuple[BM25Okapi, List]:
    """
    Load the BM25 index and tokenized chunks from disk.
    """
    with open(load_path, 'rb') as f:
        bm25_index, tokenized_chunks = pickle.load(f)
    
    print(f"BM25 index loaded from {load_path}")
    
    return bm25_index, tokenized_chunks