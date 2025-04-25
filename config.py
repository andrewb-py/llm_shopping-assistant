import os
from dotenv import load_dotenv

load_dotenv() 

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    CSV_PATH = os.getenv("CSV_PATH")