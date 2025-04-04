import os
import pypdf
from tqdm import tqdm
from typing import List, Tuple

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from .models import text_splitter, llm



def load_pdf_documents(data_dir='../data', stored_docs=[]):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf') and filename not in stored_docs:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append({"filename": filename, "content": text})
    return documents


def process_document(document: str) -> Tuple[List[Document], List[Document]]:
        """
        Process a document by splitting it into chunks and generating context for each chunk.
        """
        chunks = text_splitter.create_documents([document])
        contextualized_chunks = _generate_contextualized_chunks(document, chunks)
        return chunks, contextualized_chunks


def _generate_contextualized_chunks(document: str, chunks: List[Document]) -> List[Document]:
    """
    Generate contextualized versions of the given chunks.
    """
    contextualized_chunks = []
    for chunk in tqdm(chunks):
        context = _generate_context(document, chunk.page_content)
        contextualized_content = f"{context}\n\n{chunk.page_content}"
        contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
    return contextualized_chunks


def _generate_context(whole_document: str, chunk: str) -> str:
    """
    Generate context for a specific chunk using the language model.
    """
    prompt = ChatPromptTemplate.from_template("""
    <document> 
    {{whole_document}} 
    </document> 
    Here is the chunk we want to situate within the whole document 
    <chunk> 
    {{chunk}} 
    </chunk> 
    Please give a concise context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 

    Context:
    """)
    messages = prompt.format_messages(whole_document=whole_document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content