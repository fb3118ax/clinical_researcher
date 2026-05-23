from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import app as agent_app

api = FastAPI() # FastAPI() creates your web application instance.

class QueryRequest(BaseModel):
    query: str

@api.post("/query")
def query(request: QueryRequest):
    result = agent_app.invoke({
        "messages": [{"role": "user", "content": request.query}],
        "query": request.query,
        "sources": [],
        "final_answer": ""
    })
    return {"answer": result["messages"][-1].content}

# Think of it like this — FastAPI() is the container that holds all your routes, handles incoming HTTP requests, and sends back responses. Without it there's no web server.
# When you write @api.post("/query") — you're registering that route on this instance. When someone sends a POST request to /query, FastAPI knows to run that function.
# It's the same concept as StateGraph(AgentState) in LangGraph — you create the container first, then add things to it.