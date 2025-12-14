import asyncio
import threading
import websockets
from flask import Flask, request, jsonify

from extractresume import settopics_resume
from llmconnection import process_message
from speechtotext import send_to_assemblyai, run, send_msg_to_llm
from flask_cors import CORS
import subprocess
import time
import objgraph
import psutil
import os

process = psutil.Process(os.getpid())


# ------------------- WebSocket Handler -------------------
async def handler(websocket):
    print("🔗 Client connected")
    global stopmsgtollm
    try:
        async for message in websocket:
            if not stopmsgtollm:
                if isinstance(message, (bytes, bytearray)):
                    send_to_assemblyai(message, is_binary=True)
                else:
                    send_to_assemblyai(message)
            else:
                print("⚠️ Message to LLM is stopped (stopmsgtollm=True)")
    except websockets.exceptions.ConnectionClosed as e:
        print("❌ Client disconnected:", e)


# ------------------- Flask API -------------------
app = Flask(__name__)
application = app   # AWS reads this
CORS(
    app,
    supports_credentials=True,
    resources={
        r"/api/*": {
            "origins": "http://localhost:3000"
        }
    }
)

stopmsgtollm = False


@app.route("/api/v1/resume/topics", methods=["POST"])
def extract_topics_from_resume():

    # 1. Get form data
    user_id = request.form.get("userId")
    job_description = request.form.get("jobDescription")
    resume_file = request.files.get("resume")

    # 2. Validations
    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    if not job_description:
        return jsonify({"error": "jobDescription is required"}), 400

    if not resume_file:
        return jsonify({"error": "resume PDF file is required"}), 400

    if not resume_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    response = settopics_resume(
        user_id=user_id,
        job_description=job_description,
        resume_file=resume_file
    )

    return response

@app.after_request
def track_memory(response):
    rss_mb = process.memory_info().rss / 1024 / 1024
    print(f"\n📊 RSS Memory: {rss_mb:.2f} MB")
    print("📈 Object growth since last request:")
    objgraph.show_growth(limit=10)
    return response

@app.route("/api/v1/send-msg", methods=["POST"])
def send_msg_api():
    data = request.get_json()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    response = send_msg_to_llm(user_id)
    print(response)
    return response

@app.route("/test", methods=["POST"])
def test():
    return "test success"

@app.route("/api/v1//reconnect", methods=["POST"])
def reconnect():
    global stopmsgtollm
    stopmsgtollm = False
    return jsonify({"success": True, "stopmsgtollm": stopmsgtollm})


def run_flask():
    print("🚀 Flask API started at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


# ------------------- Start Both -------------------
async def main():

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    stt_thread = threading.Thread(target=run, daemon=True)
    stt_thread.start()

    async with websockets.serve(handler, "0.0.0.0", 8001):
        print("✅ WebSocket server started at ws://0.0.0.0:8001")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
