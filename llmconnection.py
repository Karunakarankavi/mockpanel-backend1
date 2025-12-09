import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Set it in .env")

# Initialize OpenAI client
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.7
)

# Store memory per session
sessions_memory = {}

def get_session_memory(session_id: str) -> ChatMessageHistory:
    if session_id not in sessions_memory:
        sessions_memory[session_id] = ChatMessageHistory()
        # Preload initial context
        context = """
You are a mock interviewer.
Conduct a Java Spring Boot 3-year experience interview.
Start with:
1. Tell me about yourself
2. Project explanation
3. Roles & responsibilities
Then ask technical + scenario + follow-ups. Total 25-30 questions.
"""
        sessions_memory[session_id].add_message(HumanMessage(content=context))
        sessions_memory[session_id].add_message(AIMessage(content="Understood. Let's begin the interview."))
    return sessions_memory[session_id]

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{message}")
])

chat_chain = RunnableWithMessageHistory(
    prompt | chat_model,
    memory_selector=lambda session_id: get_session_memory(session_id),
    input_messages_key="message",
    history_messages_key="history"
)

def clean_response(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9.,?!:;'\-\s\"]+", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()

def process_message(message: str, session_id: str) -> str:
    try:
        response = chat_chain.invoke({"message": message}, config={"configurable": {"session_id": session_id}})
        return clean_response(response.content)
    except Exception as e:
        return f"Error: {str(e)}"

