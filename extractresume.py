from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage   # << UPDATED
import PyPDF2
import os
import json
from dotenv import load_dotenv
import redis

# Redis Connection
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

from patternagent import generate_question_patterns

# Flask App
app = Flask(__name__)

# Allow cross-domain usage **(Important for frontends hosted remotely)**
CORS(app, supports_credentials=True)

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# ===========================================
#  LLM (UPDATED for new LangChain API syntax)
# ===========================================
llm = ChatOpenAI(
    model="gpt-4.1-mini",    # OpenAI official small-fast reasoning model
    api_key=OPENAI_KEY,
    temperature=0.7
)


# ==============================
# PDF → TEXT Extractor
# ==============================
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for pg in pdf.pages:
        text += pg.extract_text() or ""
    return text.strip()


# ==============================
# 1️⃣ MpSetTopicsFromResume
# ==============================
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


@app.route("/MpSetTopicsFromResume", methods=["POST"])
def settopics_resume():

    if "resume" not in request.files:
        return jsonify({"error": "resume file is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("jd", "")
    user_id = request.form.get("userId", "USR_" + os.urandom(3).hex())

    resume_text = extract_text_from_pdf(resume_file)

    prompt = prompt_template_resume.format(resume_text=resume_text, jd_text=jd_text or "N/A")

    messages = [
        SystemMessage(content="You are a Resume Skill Extraction AI."),
        HumanMessage(content=prompt)
    ]

    res = llm.invoke(messages)  # <—— updated method

    try:
        raw = res.content.strip()
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        parsed["userId"] = user_id

    except:
        return jsonify({"error": "Could not parse extracted JSON", "raw": res.content}), 500

    question_patterns = generate_question_patterns(parsed, llm)

    payload = {
        "question": question_patterns,
        "role": request.form.get("role",""),
        "experience": request.form.get("exp",""),
        "candidateName": parsed.get("candidateName")
    }

    redis_client.set(user_id, json.dumps(payload))
    redis_client.expire(user_id, 86400)

    return question_patterns



# ==============================
# 2️⃣ MpSetTopicsFromInput
# ==============================
prompt_template_input = PromptTemplate(
    input_variables=["skills","experience","candidateName","userId"],
    template="""
You are a Technical Interview Trainer AI.

Based on:
Candidate: {candidateName}
Experience: {experience} years
Key Skills: {skills}

Generate JSON with recommended evaluation topics.

Return ONLY JSON:

{{
 "candidateName":"{candidateName}",
 "experienceYears":{experience},
 "userId":"{userId}",
 "skills":[{skills}],
 "topicsToEvaluate":{{
      "<skill>":["topic1","topic2","topic3"]
 }}
}}
"""
)


@app.route("/MpSetTopicsFromInput", methods=["POST"])
def settopics_input():

    data = request.get_json()
    required = ["skills","experience","candidateName","userId"]

    if not data or not all(k in data for k in required):
        return jsonify({"error":"Missing required keys"}),400

    skills_str = ", ".join(data["skills"]) if isinstance(data["skills"],list) else data["skills"]

    prompt = prompt_template_input.format(
        skills=skills_str,
        experience=data["experience"],
        candidateName=data["candidateName"],
        userId=data["userId"]
    )

    messages = [
        SystemMessage(content="You generate skill-wise interview topic mapping."),
        HumanMessage(content=prompt)
    ]

    res = llm.invoke(messages)

    try:
        js = res.content.strip()
        result = json.loads(js[js.find("{"): js.rfind("}") + 1])

    except:
        return jsonify({"error":"LLM JSON Parse error","raw":res.content}),500

    question_patterns = generate_question_patterns(result,llm)

    payload = {
        "question":question_patterns,
        "role":result["skills"][0] + " Developer",
        "experience":data["experience"],
        "candidateName":result["candidateName"]
    }

    redis_client.set(data["userId"],json.dumps(payload))
    redis_client.expire(data["userId"],86400)

    return question_patterns



# ==============================
# Run Server 🚀
# ==============================
if __name__ == "__main__":
    print("\n🔥 Interview Skill Engine Running on 0.0.0.0:8085")
    app.run(host="0.0.0.0", port=8085, debug=False)

