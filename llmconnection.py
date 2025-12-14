import os
import re
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# ---------------- ENV ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in .env")

# ---------------- MODEL (LOAD ONCE) ----------------
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.7
)

# ---------------- SESSION MEMORY ----------------
sessions_memory = {}
sessions_last_used = {}

MAX_MESSAGES = 20          # prevent memory growth
SESSION_TTL = 1800         # 30 minutes

def cleanup_sessions():
    """Remove inactive sessions to prevent memory leak"""
    now = time.time()
    for sid in list(sessions_last_used.keys()):
        if now - sessions_last_used[sid] > SESSION_TTL:
            sessions_memory.pop(sid, None)
            sessions_last_used.pop(sid, None)

def get_session_history(session_id: str) -> ChatMessageHistory:
    cleanup_sessions()
    sessions_last_used[session_id] = time.time()

    if session_id not in sessions_memory:
        history = ChatMessageHistory()

        # Initial system context (ONLY ONCE)
        context = """
You are a mock interviewer.
Conduct a Java Spring Boot interview for a 3-year experienced candidate.
Start with:
1. Tell me about yourself
2. Project explanation
3. Roles & responsibilities
Then ask technical, scenario-based, and follow-up questions.
Total questions: 25–30.
"""
        history.add_message(HumanMessage(content=context))
        history.add_message(AIMessage(content="Understood. Let's begin the interview."))

        sessions_memory[session_id] = history

    # 🔥 Trim history to prevent RSS growth
    history = sessions_memory[session_id]
    if len(history.messages) > MAX_MESSAGES:
        history.messages = history.messages[-MAX_MESSAGES:]

    return history

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}")
])

# ---------------- CHAIN (FIXED) ----------------
chat_chain = RunnableWithMessageHistory(
    prompt | chat_model,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history"
)

# ---------------- HELPERS ----------------
def clean_response(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.,?!:;'\-\s\"]+", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()

# ---------------- API ENTRY ----------------
def process_message(message: str, session_id: str) -> str:
    try:
        response = chat_chain.invoke(
            {"message": message},
            config={"configurable": {"session_id": session_id}}
        )
        return clean_response(response.content)
    except Exception as e:
        return f"Error: {str(e)}"
