import os
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import json
import re

from src.tools import ALL_TOOLS

# SET YOUR API KEY HERE (or use environment variable)
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"  # Replace with your actual key

# Define State
class AgentState(TypedDict):
    messages: list[BaseMessage]

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Create tool descriptions
tool_descriptions = []
for tool in ALL_TOOLS:
    tool_descriptions.append(f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.args, indent=2)}
""")

tools_text = "\n".join(tool_descriptions)

sys_msg = SystemMessage(content=f"""You are a helpful AI assistant with access to these tools:

{tools_text}

When you need to use a tool, respond ONLY in this exact format:
TOOL_CALL: tool_name
PARAMETERS: {{"param": "value"}}

After getting tool results, provide a natural response to the user.
If no tool is needed, respond naturally.""")

def extract_tool_call(content: str):
    """Extract tool call from response"""
    if "TOOL_CALL:" not in content:
        return None
    
    try:
        tool_match = re.search(r'TOOL_CALL:\s*(\w+)', content)
        params_match = re.search(r'PARAMETERS:\s*(\{.*?\})', content, re.DOTALL)
        
        if tool_match and params_match:
            return {
                "name": tool_match.group(1),
                "parameters": json.loads(params_match.group(1))
            }
    except Exception as e:
        print(f"Parse error: {e}")
    
    return None

def execute_tool(tool_call):
    """Execute tool and return result"""
    for tool in ALL_TOOLS:
        if tool.name == tool_call["name"]:
            try:
                return tool.invoke(tool_call["parameters"])
            except Exception as e:
                return f"Error: {e}"
    return f"Tool {tool_call['name']} not found"

def agent_node(state: AgentState):
    current_messages = state["messages"]
    
    if len(current_messages) > 10:
        current_messages = current_messages[-10:]
    
    messages = [sys_msg] + current_messages
    
    # Get model response
    response = llm.invoke(messages)
    
    print(f"\n📝 Model Response:\n{response.content}\n")
    
    # Check for tool call
    tool_call = extract_tool_call(response.content)
    
    if tool_call:
        print(f"🔧 Tool Call: {tool_call['name']}")
        print(f"📋 Parameters: {tool_call['parameters']}")
        
        # Execute tool
        tool_result = execute_tool(tool_call)
        print(f"✅ Tool Result: {tool_result}\n")
        
        # Get final response with tool result
        tool_message = HumanMessage(
            content=f"