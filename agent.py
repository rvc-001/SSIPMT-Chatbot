from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
import requests
import toons  # Import the TOON library for token optimization
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# --- App Initialization ---
app = FastAPI()

# Mount the 'images' directory to serve static files
app.mount("/images", StaticFiles(directory="images"), name="images")

# --- CORS Configuration ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ChatMessage(BaseModel):
    message: str

# --- Configuration ---
# 1. Google Apps Script URL for live data from .env
try:
    APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL")
    if not APPS_SCRIPT_URL:
        raise ValueError("APPS_SCRIPT_URL environment variable not set.")
except ValueError as e:
    print(f"Error: {e}")
    APPS_SCRIPT_URL = None

# 2. Configure the Gemini API
try:
    # Retrieve key and STRIP whitespace to prevent 'Illegal header value' gRPC errors
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    # .strip() removes hidden newlines/spaces that cause crashes
    genai.configure(api_key=api_key.strip()) 
    
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Helper Function to Fetch College Data ---
def fetch_college_data():
    """
    Fetches the latest college data from the Google Apps Script endpoint.
    """
    if not APPS_SCRIPT_URL:
        return {"error": "Could not fetch live data. APPS_SCRIPT_URL not configured."}
    try:
        response = requests.get(APPS_SCRIPT_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Google Sheet: {e}")
        return {"error": "Could not fetch live data."}

# --- API Endpoint ---
@app.post("/chat")
async def chat(chat_message: ChatMessage):
    if model is None:
        return {'reply': 'Error: The chatbot is not configured correctly. Please check server logs.'}

    user_message = chat_message.message
    if not user_message:
        return {'reply': 'Please provide a message.'}

    # --- Fetch Live Data on Every Request ---
    college_data = fetch_college_data()
    
    # OPTIMIZATION: Use TOON dumps instead of JSON dumps
    # This removes braces, quotes, and commas to save tokens while keeping structure.
    college_data_string = toons.dumps(college_data)

    # --- System Prompt to handle list data ---
    SYSTEM_PROMPT = f"""
    You are "Sankalp", a friendly, multilingual, and helpful AI assistant for SSIPMT.

    Your primary goal is to answer user questions based *only* on the information provided below.
    - Do not make up information or use external knowledge.
    - If the answer is not found in the provided data, politely say that you don't have that information in the user's language.
    - Detect the user's language and respond in the same language.
    - Be robust to spelling and grammatical errors in the user's query. Try to understand the intent.
    - Format your answers clearly, using markdown for bolding and lists when appropriate.

    Here is the college data in TOON (Token-Oriented Object Notation) format:
    ---
    {college_data_string}
    ---
    """

    try:
        chat_session = model.start_chat(history=[
            {'role': 'user', 'parts': [SYSTEM_PROMPT]},
            {'role': 'model', 'parts': ["Hello! I'm Sankalp, your AI assistant for SSIPMT. How can I help you today?"]},
        ])

        response = chat_session.send_message(user_message)

        return {'reply': response.text}

    except Exception as e:
        print(f"Error during chat generation: {e}")
        return {'reply': 'Sorry, something went wrong on my end. Please try again.'}

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Chatbot server is running"}

# --- Health Check Endpoint ---
@app.head("/")
def status_check():
    # Return a response with no body for HEAD requests
    return Response(status_code=200)