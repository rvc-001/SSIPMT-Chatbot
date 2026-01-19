from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- App Initialization ---
app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

# --- Configuration ---
# Global variable to hold the URL
APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL")

# --- Define the Tool (Function) ---
# This is the specific function Gemini will "call" when it needs data.
def get_college_info():
    """
    Retrieves the latest official information about SSIPMT college.
    
    Use this tool whenever a user asks about:
    - Fees, Admission, or Scholarships
    - Courses, Departments, or Faculty
    - Hostels, Transport, or Campus Facilities
    - Placements or Recruiters
    
    Returns:
        dict: The full college dataset in JSON format.
    """
    if not APPS_SCRIPT_URL:
        return {"error": "Database URL is not configured."}
    
    try:
        print("Model is calling the database...") # Log to see when it happens
        response = requests.get(APPS_SCRIPT_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Database Fetch Error: {e}")
        return {"error": "Could not retrieve college data."}

# --- Initialize Gemini with Tools ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set.")
    
    genai.configure(api_key=api_key.strip())
    
    # 1. We create a dictionary of tools to give to the model
    tools_list = [get_college_info]

    # 2. Initialize the model with these tools
    # Using gemini-2.5-flash as discussed for Free Tier + Multilingual support
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        tools=tools_list
    )
    
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    if model is None:
        return {'reply': 'System Error: Chatbot not configured.'}

    user_message = chat_message.message
    if not user_message:
        return {'reply': 'Please say something.'}

    # --- The System Prompt ---
    # Notice how clean it is? We don't dump data here anymore.
    # We just tell Sankalp *how* to behave.
    system_instruction = """
    You are "Sankalp", the AI assistant for SSIPMT.
    
    PROTOCOL:
    1. If the user asks about college details (fees, hostel, etc.), YOU MUST use the 'get_college_info' tool to get the facts.
    2. Do not answer factual questions from your own memory; always check the tool.
    3. If the tool returns data, answer the user strictly based on that data.
    4. If the user asks a general question (e.g., "Hi", "How are you?"), answer naturally without calling the tool.
    5. Always reply in the same language the user speaks (Hindi/English).
    6. You were made/created by Rajvardhan Chhugani and Priyanshu Panda from SSIPMT.
    """

    try:
        # Start chat with automatic function calling enabled
        chat_session = model.start_chat(
            history=[
                {'role': 'user', 'parts': [system_instruction]},
                {'role': 'model', 'parts': ["Understood. I am Sankalp. I will use the 'get_college_info' tool for queries about the college."]}
            ],
            enable_automatic_function_calling=True
        )

        # Send the user's message
        response = chat_session.send_message(user_message)

        # Return the text text result
        return {'reply': response.text}

    except Exception as e:
        print(f"Chat Error: {e}")
        return {'reply': 'I am having trouble accessing the college database right now. Please try again.'}

@app.get("/")
def read_root():
    return {"status": "Sankalp AI is running with Function Calling"}

@app.head("/")
def status_check():
    return Response(status_code=200)