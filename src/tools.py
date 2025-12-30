from langchain_core.tools import tool
from src.rag import query_qdrant
import requests

# --- TOOL 1: Weather ---
@tool
def get_weather(city: str):
    """
    Fetches the current weather for a specific city.
    Call this tool when the user asks about weather, temperature, or rain.
    """
    api_key = ""  
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            weather_desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"The weather in {city} is {weather_desc} with a temperature of {temp}°C."
        else:
            return f"Error fetching weather: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Failed to connect to weather service: {e}"

# --- TOOL 2: RAG (PDF Search) ---
@tool
def search_knowledge_base(query: str):
    """
    Searches the uploaded PDF document for information.
    Call this tool when the user asks questions about the content of the file
    (e.g., "What does the document say about X?", "Summarize the file").
    """
    return query_qdrant(query)

# Export list of tools
ALL_TOOLS = [get_weather, search_knowledge_base]