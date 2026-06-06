from agent import NPCAgent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="AI Co-Worker Engine", version="1.0.0")

npc_agent = NPCAgent()

class MessageContext(BaseModel):
    role: str 
    content: str

class ChatRequest(BaseModel):
    persona_id: str
    message: str
    history: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    assistant_message: str
    state_update: List[Dict[str, Any]]
    safety_flags: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Co-Worker Engine API"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_npc(request: ChatRequest):
    try:
        
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        langchain_history = []
        if request.history:
            for msg in request.history:
                if msg.get("type") == "human":
                    langchain_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("type") == "ai":
                    langchain_history.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("type") == "system":
                    langchain_history.append(SystemMessage(content=msg.get("content", "")))

        result = npc_agent.invoke(
            persona_id=request.persona_id, 
            user_message=request.message,
            history=langchain_history
        )

        serialized_state = []
        for msg in result["state_update"]:
            if isinstance(msg, HumanMessage):
                serialized_state.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                serialized_state.append({"type": "ai", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                serialized_state.append({"type": "system", "content": msg.content})
        
        return ChatResponse(
            assistant_message=result["assistant_message"],
            state_update=serialized_state,
            safety_flags=result["safety_flags"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
