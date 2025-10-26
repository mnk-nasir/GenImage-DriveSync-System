"""
config.py

Environment-backed configuration for the research report companion script.
Create a .env file in the project root or set environment variables directly.

Supported environment variables (see .env.example):
- OPENAI_API_KEY
- OPENAI_MODEL
- NEWSAPI_KEY
- SERPAPI_KEY
- PDFSHIFT_API_KEY
- SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- DEFAULT_RECIPIENT_EMAIL
"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def getenv(k: str, default=None):
    v = os.getenv(k)
    return v if v is not None else default

# LLM
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_MODEL = getenv("OPENAI_MODEL", "gpt-4o-mini")

# News and Scholar
NEWSAPI_KEY = getenv("NEWSAPI_KEY")
SERPAPI_KEY = getenv("SERPAPI_KEY")

# PDF conversion
PDFSHIFT_API_KEY = getenv("PDFSHIFT_API_KEY")

# SMTP (Gmail or other)
SMTP_SERVER = getenv("SMTP_SERVER")
SMTP_PORT = getenv("SMTP_PORT")
SMTP_USER = getenv("SMTP_USER")
SMTP_PASS = getenv("SMTP_PASS")

# Telegram
TELEGRAM_BOT_TOKEN = getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = getenv("TELEGRAM_CHAT_ID")

# App defaults
DEFAULT_RECIPIENT_EMAIL = getenv("DEFAULT_RECIPIENT_EMAIL", "example@example.com")
OUTPUT_DIR = getenv("OUTPUT_DIR", str(ROOT / "outputs"))

def as_dict():
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY and ("***redacted***"),
        "OPENAI_MODEL": OPENAI_MODEL,
        "NEWSAPI_KEY": NEWSAPI_KEY and ("***redacted***"),
        "SERPAPI_KEY": SERPAPI_KEY and ("***redacted***"),
        "PDFSHIFT_API_KEY": PDFSHIFT_API_KEY and ("***redacted***"),
        "SMTP_SERVER": SMTP_SERVER,
        "SMTP_PORT": SMTP_PORT,
        "SMTP_USER": SMTP_USER and ("***redacted***"),
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN and ("***redacted***"),
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "DEFAULT_RECIPIENT_EMAIL": DEFAULT_RECIPIENT_EMAIL
    }