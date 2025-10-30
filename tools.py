from config import TAVILY_API_KEY, WEATHER_API_KEY

import requests
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

@tool
def get_weather(query: str):
    """Get current weather for a given location using WeatherAPI"""
    endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
    response = requests.get(endpoint)
    data = response.json()
    return data if data.get("location") else {"error": "Weather Data Not Found"}

@tool
def search_web(query: str):
    """Search the web for a given query to provide further information"""
    tavily_search = TavilySearch(
        api_key=TAVILY_API_KEY, max_results=2, search_depth='advanced', max_tokens=1000
    )
    return tavily_search.invoke(query)