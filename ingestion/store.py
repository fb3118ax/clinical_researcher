from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import VECTOR_STORE_DIR, EMBEDDING_MODEL, TOP_K

def get_vector_store():
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )
    return vector_store

def get_retriever():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return retriever