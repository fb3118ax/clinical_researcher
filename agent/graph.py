from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI # — for the LLM
from agent.state import AgentState # — for the state
from agent.tools import search_studies, compare_studies, web_search # — to bind to LLM and pass to ToolNode
from config import LLM_MODEL # — for the model name
#----------------------------------------------------------------------------------------------------------

llm= ChatOpenAI(model=LLM_MODEL, temperature=0)
llm_with_tools = llm.bind_tools([search_studies, compare_studies, web_search])
#----------------------------------------------------------------------------------------------------------

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"
#----------------------------------------------------------------------------------------------------------

def call_model(state: AgentState):
    system = SystemMessage(content="""You are a Clinical Research Assistant.

TOOL USAGE RULES (follow in order):
1. If the query is about guidelines, policies, or recent developments → call web_search first
2. For all other clinical questions → call search_studies first
3. If asked to compare specific studies → call compare_studies
4. Always use the exact user query when calling any tool
5. Always cite Study ID and Author/Year in your final answer""")
    messages = [system] + state["messages"]
    print(f"Total messages going to LLM: {len(messages)}")
    print(f"First message type: {type(messages[0])}")
    response = llm_with_tools.invoke(messages)
    print(f"Tool calls in response: {response.tool_calls}")
    return {"messages": [response]}
#----------------------------------------------------------------------------------------------------------

call_tools = ToolNode([search_studies, compare_studies, web_search]) 
#----------------------------------------------------------------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node('LLM', call_model)
graph.add_node('tools',call_tools)
graph.set_entry_point('LLM')
graph.add_edge('tools', 'LLM') # --means — after tools finishes executing, always go back to LLM. No condition, no decision, just always.add_edge creates a fixed, unconditional connection between two nodes.
graph.add_conditional_edges(
    'LLM',
    should_continue,
    {
        'tools': 'tools',
        'end': END
    }
)

#----------------------------------------------------------------------------------------------------------
app = graph.compile()
#==========================================================================================================