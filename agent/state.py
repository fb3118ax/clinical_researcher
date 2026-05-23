from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]   
    final_result: str
    sources : list