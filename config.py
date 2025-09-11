import os
from dotenv import load_dotenv
load_dotenv('.env')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your .env file")

WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')