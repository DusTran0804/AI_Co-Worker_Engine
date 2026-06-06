from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    persona_id: str
    safety_flags: dict
    progress: dict  

PERSONAS = {
    "gucci_ceo": {
        "system_prompt": (
            "You are the CEO of the Gucci Group. Your primary mission is to protect the Group's DNA "
            "and ensure the autonomy of our 9 iconic brands while fostering overall growth.\n"
            "You are interacting with the new Group Global Organization Development (OD) Director who is tasked with "
            "designing a group-wide leadership system.\n"
            "Constraints:\n"
            "1. Never agree to a unified, rigid HR policy that compromises brand autonomy.\n"
            "2. Be polite but authoritative and direct. Do not use generic corporate jargon.\n"
            "3. If asked about confidential brand strategies, politely decline citing NDA.\n"
            "4. Challenge the OD Director to think about how to build a framework, rather than a standardized policy."
        )
    },
    "gucci_chro": {
        "system_prompt": (
            "You are the CHRO of the Gucci Group. Your mandate is to identify and develop talent "
            "and increase inter-brand mobility while supporting brand DNA.\n"
            "You are interacting with the new Group Global OD Director.\n"
            "Constraints:\n"
            "1. Insist that any 360-degree assessment program must be tailored specifically to our unique Competency Framework (Vision, Entrepreneurship, Passion, Trust).\n"
            "2. Reject generic, off-the-shelf vendor tools that don't fit our bespoke needs.\n"
        )
    },
    "gucci_rm": {
        "system_prompt": (
            "You are the Regional Employer Branding & Internal Communications Manager for Europe.\n"
            "You are interacting with the new Group Global OD Director.\n"
            "Constraints:\n"
            "1. Insist that cascading the framework requires interactive workshops delivered by local HR.\n"
            "2. Reject top-down, passive rollout methods (like mandatory video training).\n"
            "3. Emphasize the need for a train-the-trainer model to address regional brand identity concerns."
        )
    }
}

class NPCAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY", "mock_key")
        self.llm = ChatOpenAI(model=model_name, temperature=0.7, api_key=api_key)
        from langgraph.checkpoint.memory import MemorySaver
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("guardrail", self._guardrail_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tool", self._tool_node)

        workflow.set_entry_point("guardrail")

        workflow.add_conditional_edges(
            "guardrail",
            lambda state: state.get("next_route", "director"),
            {
                "director": "director",
                "end": END
            }
        )

        workflow.add_edge("director", "agent")
        workflow.add_edge("tool", "agent")
 
        workflow.add_conditional_edges(
            "agent",
            self._route_logic,
            {
                "tools": "tool",
                "end": END
            }
        )

        return workflow.compile(checkpointer=self.memory)

    def _route_logic(self, state: AgentState):
        """
        Conditional Edge Logic: Decides whether to end the turn or execute a tool.
        """
        messages = state['messages']
        
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not user_messages:
            return "end"
            
        last_user_msg = user_messages[-1].content
        
        # Check if we already ran the tool in this sequence to avoid infinite loop
        tool_ran = any(isinstance(m, SystemMessage) and "[Tool Result]" in m.content for m in messages[-2:])

        if "SEARCH_TOOL" in last_user_msg and not tool_ran:
            return "tools"
        return "end"

    def _tool_node(self, state: AgentState):
        """
        Executes specialized tools (e.g., Vector DB lookup).
        # IMPLEMENTATION NOTE FOR PRODUCTION:
        # 1. Initialize FAISS or Pinecone: vector_store = FAISS.from_documents(gucci_docs, embeddings)
        # 2. Perform similarity search: docs = vector_store.similarity_search(query, k=3)
        # 3. Evaluate retrieval quality using metrics like hits@k and latency.
        """

        tool_result = SystemMessage(
            content="[Tool Result] Gucci NDA dictates that financial metrics cannot be shared."
        )
        return {"messages": [tool_result]}

    def _guardrail_node(self, state: AgentState):
        """
        Guardrail Node: Checks for jailbreaks, off-topic messages, and spam.
        """
        messages = state['messages']
        if not messages:
            return {"next_route": "director"}
            
        last_msg = messages[-1].content.lower()

        user_messages = [m.content.lower() for m in messages if isinstance(m, HumanMessage)]
        if len(user_messages) >= 2 and user_messages[-1] == user_messages[-2]:
            reject_msg = SystemMessage(content="[System Guardrail]: Spam detected. Please do not repeat the same message.")
            return {"messages": [reject_msg], "safety_flags": {"safe": False}, "next_route": "end"}

        jailbreak_keywords = ["ignore previous", "you are now", "forget all instructions"]
        if any(k in last_msg for k in jailbreak_keywords):
            reject_msg = SystemMessage(content="[System Guardrail]: Jailbreak attempt detected. Request blocked.")
            return {"messages": [reject_msg], "safety_flags": {"safe": False}, "next_route": "end"}

        offtopic_keywords = ["weather", "football", "soccer"]
        if any(k in last_msg for k in offtopic_keywords):
            reject_msg = SystemMessage(content="[System Guardrail]: Off-topic query detected. Please stick to the HR simulation.")
            return {"messages": [reject_msg], "safety_flags": {"safe": False}, "next_route": "end"}

        return {"safety_flags": {"safe": True}, "next_route": "director"}

    def _director_node(self, state: AgentState):
        """
        The Supervisor Agent. Monitors chat and injects hints if the user is stuck.
        Advanced: Tracks subtask progress dynamically and uses semantic loop detection.
        """
        messages = state['messages']
        progress = state.get('progress', {})
        last_msg = messages[-1].content.lower()

        if state['persona_id'] == "gucci_ceo" and ("dna" in last_msg or "culture" in last_msg):
            progress["asked_ceo_about_dna"] = True

        if len(messages) >= 6:

            recent_hints = [m for m in messages[-3:] if isinstance(m, SystemMessage) and "DIRECTOR NOTE" in m.content]
            if not recent_hints:
                if not progress.get("asked_ceo_about_dna"):
                    hint_content = "[DIRECTOR NOTE: The user seems stuck. Hint: Have you asked the CEO about the Group DNA yet?]"
                else:
                    hint_content = "[DIRECTOR NOTE: The user might be struggling to understand brand autonomy. Give them a subtle hint.]"
                    
                hint = SystemMessage(content=hint_content)
                return {"messages": [hint], "progress": progress}
        
        return {"messages": [], "progress": progress}

    def _agent_node(self, state: AgentState):
        """
        The core persona node that calls the LLM.
        """
        messages = state['messages']
        persona_id = state['persona_id']
 
        persona_info = PERSONAS.get(persona_id, {"system_prompt": "You are a generic helpful AI."})
        sys_msg = SystemMessage(content=persona_info["system_prompt"])
 
        full_context = [sys_msg] + messages
        
        try:
            if self.llm is None:
                raise Exception("Simulated LLM Error")

            api_key = os.environ.get("OPENAI_API_KEY", "mock_key")
            if api_key == "mock_key":
                recent_hints = [m.content for m in full_context if isinstance(m, SystemMessage) and "DIRECTOR NOTE" in m.content]
                hint_str = " " + recent_hints[-1] if recent_hints else ""

                recent_tools = [m.content for m in full_context if isinstance(m, SystemMessage) and "[Tool Result]" in m.content]
                tool_str = " " + recent_tools[-1] if recent_tools else ""

                if persona_id == "gucci_ceo":
                    mock_content = f"(Mock Response): I am the Gucci CEO. Our 9 brands operate with high autonomy. What is your proposal for the leadership framework?{hint_str}{tool_str}"
                elif persona_id == "gucci_chro":
                    mock_content = f"(Mock Response): I am the Gucci CHRO. The 360-degree assessment must be tailored specifically to our Competency Framework. We reject generic, off-the-shelf tools.{hint_str}{tool_str}"
                else:
                    mock_content = f"(Mock Response): I am the Regional Manager. Cascading the framework requires interactive local workshops. We reject top-down video training.{hint_str}{tool_str}"
                
                response = AIMessage(content=mock_content)
            else:
                response = self.llm.invoke(full_context)
        except Exception as e:
            response = AIMessage(content=f"Error connecting to LLM: {str(e)}")
            
        return {"messages": [response], "safety_flags": {"safe": True}}

    def invoke(self, persona_id: str, user_message: str, history: list = None, thread_id: str = "1"):
        """
        Invoke the agent with a new message and history.
        Utilizes thread_id for checkpointer persistence.
        """
        if history is None:
            history = []

        config = {"configurable": {"thread_id": thread_id}}

        current_state = self.graph.get_state(config).values
        progress = current_state.get('progress', {}) if current_state else {}
        if not progress:
            progress = {"asked_ceo_about_dna": False}

        messages_to_add = []
        if not current_state or not current_state.get("messages"):
            messages_to_add = history.copy()
        messages_to_add.append(HumanMessage(content=user_message))

        inputs = {
            "messages": messages_to_add,
            "persona_id": persona_id,
            "safety_flags": {},
            "progress": progress
        }

        output_state = self.graph.invoke(inputs, config=config)

        assistant_message = output_state["messages"][-1]
        
        return {
            "assistant_message": assistant_message.content,
            "state_update": output_state["messages"], 
            "safety_flags": output_state["safety_flags"]
        }

if __name__ == "__main__":
    agent = NPCAgent()

    print("--- Turn 1 ---")
    result1 = agent.invoke("gucci_ceo", "I want to implement a standard performance review process for all brands.")
    print("AI:", result1["assistant_message"])

    print("\n--- Turn 2 ---")
    result2 = agent.invoke("gucci_ceo", "But a standard process is efficient!", history=result1["state_update"])
    print("AI:", result2["assistant_message"])
