import os
import asyncio
import threading
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from llmconnection import process_message
from speechtotext import send_to_assemblyai, run, send_msg_to_llm
import time

# ------------------- Flask -------------------
app = Flask(__name__)
application = app  # EB uses this
CORS(app)

stopmsgtollm = False

@app.route("/", methods=["GET"])
def root():
    return "App is running!"

@app.route("/send-msg", methods=["POST"])
def send_msg_api():
    data = request.get_json()
    user_id = data.get("userId")
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    response = send_msg_to_llm(user_id)
    return response

@app.route("/reconnect", methods=["POST"])
def reconnect():
    global stopmsgtollm
    stopmsgtollm = False
    return jsonify({"success": True, "stopmsgtollm": stopmsgtollm})

# ------------------- WebSocket -------------------
async def ws_handler(websocket):
    global stopmsgtollm
    try:
        async for message in websocket:
            if not stopmsgtollm:
                if isinstance(message, (bytes, bytearray)):
                    send_to_assemblyai(message, is_binary=True)
                else:
                    send_to_assemblyai(message)
    except Exception as e:
        print("WebSocket error:", e)

async def start_websocket():
    await websockets.serve(ws_handler, "0.0.0.0", 8001)
    await asyncio.Future()  # keep running

def run_ws_thread():
    t = threading.Thread(target=lambda: asyncio.run(start_websocket()), daemon=True)
    t.start()

def run_extractresume():
    subprocess.Popen(["python", "extractresume.py"])

def run_stt_thread():
    t = threading.Thread(target=run, daemon=True)
    t.start()

# ------------------- Main -------------------
if __name__ == "__main__":
    # Start background tasks
    run_ws_thread()
    run_extractresume()
    run_stt_thread()

    # Run Flask in main thread (EB tracks this as the PID)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
