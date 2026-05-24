from langchain_core.tools import tool
from tavily import TavilyClient
from exception import ToolExecutionError
from ingestion.store import get_vector_store, get_retriever
from config import TAVILY_API_KEY

tavily = TavilyClient(api_key=TAVILY_API_KEY)
vector_store = get_vector_store()
retriever = get_retriever()

@tool
def search_studies(query: str) ->list:
    '''Search the clinical research corpus for studies relevant to a query.'''
    print("search_studies agent invoked")
    try:
        docs = retriever.invoke(query) # -- it's LangChain's built in method
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
        raise ToolExecutionError(f"compare_studies failed: {e}") from e    


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
    """Search the details on the web as per queary by user"""
    print("web_search agent invoked")
    result = []
    try:
        docs = tavily.search(query)
        for doc in docs['results']: # this loop will fetch details for each id comes from above loop
            result.append({
                "URL": doc.get("url"),
                "Title": doc.get("title"),
                "content": doc.get("content")
                })
        return result
    except Exception as e:
        raise ToolExecutionError(f"web_search failed: {e}") from e
 