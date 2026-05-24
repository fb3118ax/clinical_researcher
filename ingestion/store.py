from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import VECTOR_STORE_DIR, EMBEDDING_MODEL, TOP_K

def get_vector_store():
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )
    print(f"DB chunk count: {vector_store._collection.count()}")
    return vector_store


def get_retriever(author_year: str = None):
    vector_store = get_vector_store()
    
    search_kwargs = {"k": TOP_K}
    
    if author_year:
        search_kwargs["filter"] = {"author_year": {"$eq": author_year}}
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever