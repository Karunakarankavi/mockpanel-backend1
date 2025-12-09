import os
import json
import openai
import redis
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from threading import Lock

# ----------- Pinecone v3+ Correct Import ----------- #
from pinecone import Pinecone, ServerlessSpec

from evaluation_agent import EvaluationAgent


# ---------------- Redis Setup ---------------- #
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# ---------------- Flask + OpenAI + Pinecone Setup ---------------- #
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API"))
INDEX_NAME = "topic-summary"
index = pc.Index(INDEX_NAME)   # global index reference

app = Flask(__name__)
agent_lock = Lock()
agents = {}
evaluators = {}   # user_id -> EvaluationAgent instance
question_asked = None



# ---------------- MAIN QUESTION GENERATOR ---------------- #
class QuestionPatternAgent:

    def __init__(self, question_structure, developer_role, experience_level, max_questions_per_topic=2, user_id=None):
        self.structure = question_structure
        self.developer_role = developer_role
        self.experience_level = experience_level
        self.max_questions_per_topic = max_questions_per_topic
        self.user_id = user_id

        self.current_domain = list(self.structure.keys())[0]
        self.current_topic_index = 0
        self.current_pattern_index = 0
        self.question_count = 0
        self.topics = list(self.structure[self.current_domain].keys())

    def _get_current_topic(self):
        return self.topics[self.current_topic_index]

    def _get_current_pattern(self):
        domain, topic = self.current_domain, self._get_current_topic()
        patterns = self.structure[domain][topic]
        return patterns[self.current_pattern_index % len(patterns)]

    def _move_to_next_topic(self):
        self.current_topic_index += 1
        self.current_pattern_index = 0
        self.question_count = 0

        if self.current_topic_index >= len(self.topics):
            domain_keys = list(self.structure.keys())
            current_domain_index = domain_keys.index(self.current_domain)

            if current_domain_index + 1 < len(domain_keys):
                self.current_domain = domain_keys[current_domain_index + 1]
                self.topics = list(self.structure[self.current_domain].keys())
                self.current_topic_index = 0
            else:
                self.current_domain = None


    # ----------------- Redis Storage ----------------- #
    def _get_asked_questions(self, topic):
        redis_key = f"asked_questions:{self.user_id}:{topic}"
        return redis_client.lrange(redis_key, 0, -1) or []

    def _store_asked_question(self, topic, question):
        redis_key = f"asked_questions:{self.user_id}:{topic}"
        redis_client.rpush(redis_key, question)


    # ---------------- Embedding ---------------- #
    def _embed_text(self, text):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            vector = response.data[0].embedding

            # ensure 1024 vector dimension
            expected_dim = 1024
            if len(vector) != expected_dim:
                if len(vector) > expected_dim:
                    vector = vector[:expected_dim]
                else:
                    vector += [0.0] * (expected_dim - len(vector))

            return vector

        except Exception as e:
            print(f"⚠️ Embedding Error: {e}")
            return None


    # ---------------- Question Generation ---------------- #
    def _generate_question_from_llm(self, domain, topic, pattern_type, previous_answer=None):

        # 🔥 Query Pinecone for summary context
        try:
            topic_vector = self._embed_text(topic)

            pinecone_result = index.query(
                vector=topic_vector,
                top_k=1,
                include_metadata=True,
                filter={"type": "summary"}
            )

            topic_summary = ""
            weak_areas = []

            if pinecone_result.matches:
                meta = pinecone_result.matches[0].metadata
                topic_summary = meta.get("summary", "")
                weak_areas = meta.get("weak_areas", [])

        except Exception as e:
            print(f"⚠️ Pinecone retrieval failed: {e}")
            topic_summary = ""
            weak_areas = []


        summary_context = ""
        if topic_summary or weak_areas:
            summary_context = f"""
Candidate performance summary:
"{topic_summary}"
Weak skills to probe deeper: {", ".join(weak_areas) if weak_areas else "None"}
"""

        dynamic_hint = ""
        if previous_answer:
            dynamic_hint = f"\nFollow-up question should relate to previous answer: \"{previous_answer}\".\n"

        prompt = f"""
You are an expert interviewer.
Generate ONE {pattern_type} question for a {self.experience_level} {self.developer_role} candidate.
Topic: {topic} ({domain})
{summary_context}
{dynamic_hint}
Return ***only the question***.
"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict interviewer."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"


    # ---------------- Core Public Method ---------------- #
    def get_question(self, previous_answer=None):

        if not self.current_domain:
            return {"question": "✅ Interview Completed!"}

        domain = self.current_domain
        topic = self._get_current_topic()
        pattern_type = self._get_current_pattern()

        asked = self._get_asked_questions(topic)
        attempt = 0

        while attempt < 3:
            question = self._generate_question_from_llm(domain, topic, pattern_type, previous_answer)
            if question not in asked:
                break
            attempt += 1

        self._store_asked_question(topic, question)
        self.question_count += 1
        self.current_pattern_index += 1

        if self.question_count >= self.max_questions_per_topic:
            self._move_to_next_topic()

        return {"domain": domain, "topic": topic, "pattern": pattern_type, "question": question}



# ---------------- API ENTRY ---------------- #
def get_question_endpoint(user_answer, userid):
    global question_asked

    user_id = userid
    previous_answer = user_answer  

    data = redis_client.get(user_id)
    payload = json.loads(data)

    question_structure = payload.get("question")
    role = payload.get("role")
    exp = payload.get("experience")

    # Initialize Question Agent if not present
    with agent_lock:
        if user_id not in agents:
            agents[user_id] = QuestionPatternAgent(
                question_structure,
                developer_role=role,
                experience_level=exp,
                user_id=user_id
            )

    agent = agents[user_id]
    result = agent.get_question(previous_answer)


    # ---------------- Evaluation Agent ---------------- #
    if user_id not in evaluators:
        evaluators[user_id] = EvaluationAgent(role=role, experience_level=exp)

    if question_asked:
        evaluators[user_id].add_question_answer(question_asked, previous_answer, result.get("topic"), user_id)

    question_asked = result.get("question")

    return result


if __name__ == "__main__":
    app.run(port=5000, debug=True)

