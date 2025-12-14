import PyPDF2
import json
import os
import redis
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from patternagent import generate_question_patterns
from dotenv import load_dotenv


# Load environment
load_dotenv(override=True)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# Redis Connection
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# LLM instance
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=OPENAI_KEY,
    temperature=0.7
)

# Prompt template
prompt_template_resume = PromptTemplate(
    input_variables=["resume_text", "jd_text"],
    template="""
You are an expert Resume Analysis AI.
Analyze the candidate resume and extract:

1. Name (if possible)
2. Years of experience
3. List of technical skills found
4. Topics to evaluate candidate under each skill based on their experience level.

Return ONLY valid JSON in format:

{{
  "candidateName": "<name>",
  "experienceYears": <int>,
  "userId": "",
  "skills": ["Java","Spring Boot","SQL"],
  "topicsToEvaluate": {{
      "Java": ["OOP","Collections","JVM internals"],
      "Spring Boot": ["REST","JPA","Security"]
  }}
}}

Resume:
{resume_text}

Job Description:
{jd_text}
"""
)

# PDF → Text
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for pg in pdf.pages:
        text += pg.extract_text() or ""
    return text.strip()


# Main function to call from Flask
def settopics_resume(user_id, job_description, resume_file):
    resume_text = extract_text_from_pdf(resume_file)

    prompt = prompt_template_resume.format(resume_text=resume_text, jd_text=job_description or "N/A")
    messages = [
        SystemMessage(content="You are a Resume Skill Extraction AI."),
        HumanMessage(content=prompt)
    ]

    res = llm.invoke(messages)

    try:
        raw = res.content.strip()
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        parsed["userId"] = user_id
    except Exception:
        return {"error": "Could not parse extracted JSON", "raw": res.content}, 500

    # Generate question patterns
    question_patterns = generate_question_patterns(parsed, llm)

    # Save in Redis
    redis_client.set(user_id, json.dumps({"question": question_patterns}))
    redis_client.expire(user_id, 86400)

    return question_patterns
