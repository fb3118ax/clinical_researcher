from langchain_core.tools import tool
from tavily import TavilyClient
from exception import ToolExecutionError
from ingestion.store import get_vector_store, get_retriever
from config import TAVILY_API_KEY
from config import TOP_K

tavily = TavilyClient(api_key=TAVILY_API_KEY)
vector_store = get_vector_store()
retriever = get_retriever()

@tool
def search_studies(query: str, author_year: str = None) -> list:
    '''Search the clinical research corpus for studies relevant to a query.
    If the user mentions a specific study or author, pass their full name as author_year.
    
    Available studies:
    - "Honghao 2025"
    - "Mengxin Wang 2025"
    - "Catherine Scott 2025"
    - "Grace Joshy 2025"
    - "Ping Wang 2025"
    - "Yun Feng 2025"
    - "Yutung Yen 2025"
    
    Examples:
    - user says "Honghao" or "Honghao study" -> author_year="Honghao 2025"
    - user says "Mengxin" or "Wang 2025" -> author_year="Mengxin Wang 2025"
    - user says "compare all studies" -> author_year=None'''
    print("search_studies agent invoked")
    try:
        search_kwargs = {"k": TOP_K}
        if author_year:
            print(f"Filtering by author_year: {author_year}")
            search_kwargs["filter"] = {"author_year": {"$eq": author_year}}
        
        filtered_retriever = get_vector_store().as_retriever(search_kwargs=search_kwargs)
        
        docs = filtered_retriever.invoke(query)
        print(f"Docs retrieved: {len(docs)}")
        result = []
        for doc in docs:
            meta = doc.metadata
            print(meta)
            print(doc.page_content[:200])
            result.append({
                "study_id": meta.get("study_id"),
                "author_year": meta.get("author_year"),
                "page_number": meta.get("page_number"),
                "content": doc.page_content[:500]
            })
        return result
    except Exception as e:
        raise ToolExecutionError(f"search_studies failed: {e}") from e 


@tool
def compare_studies(query: str, ids: list) -> list:
    """Search the clinical research corpus for studies relevant to a query for comparison"""
    print("compare_studies agent invoked")
    result = []
    try:
        for id in ids:
            str_id = str(id)
            print(f"Searching for id: {str_id}, type: {type(str_id)}")
            docs = vector_store.similarity_search(query, k=3, filter={"study_id": str_id})
            print(f"Docs found: {len(docs)}")
            for doc in docs:
                meta = doc.metadata
                result.append({
                    "study_id": meta.get("study_id"),
                    "author_year": meta.get("author_year"),
                    "page_number": meta.get("page_number"),
                    "content": doc.page_content[:500]
                })
        return result
    except Exception as e:
        raise ToolExecutionError(f"compare_studies failed: {e}") from e
    


@tool
def web_search(query : str)->list:
    """Search the web for information not found in the local research corpus.
Use this tool when:
- User asks about general medical knowledge not specific to the 7 studies
- User asks for latest news or guidelines
- Local search returns no relevant results
Do NOT use this for queries about the 7 studies in the corpus:
"Honghao 2025", "Mengxin Wang 2025", "Catherine Scott 2025",
"Grace Joshy 2025", "Ping Wang 2025", "Yun Feng 2025", "Yutung Yen 2025"
"""
    print("web_search agent invoked")
    result = []
    try:
        docs = tavily.search(query, max_results=5)
        for doc in docs['results']: # this loop will fetch details for each id comes from above loop
            result.append({
                "URL": doc.get("url"),
                "Title": doc.get("title"),
                "content": doc.get("content")
                })
        return result
    except Exception as e:
        raise ToolExecutionError(f"web_search failed: {e}") from e
 