import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
from app.agents.ng12_agent import assess_patient, warmup_agent
from app.agents.chat_agent import chat_with_guidelines, get_chat_history, clear_chat_history
from app.tools.patient_db import get_patient_by_id, get_all_patient_ids, add_new_patients

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to warm up AI services on startup."""
    logger.info("Warming up AI services (Embedding & LLM)...")
    warmup_agent()
    logger.info("AI Services ready.")
    yield

app = FastAPI(title="NG12 Cancer Referral API", version="1.0", description="Clinical decision support system for NICE NG12 guidelines.", lifespan=lifespan)

# --- Pydantic Models ---
class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    smoking_history: str
    symptoms: List[str]
    symptom_duration_days: int

class AssessmentResponse(BaseModel):
    patient_summary: Optional[str] = None
    guideline_analysis: Optional[str] = None
    recommendation: Optional[str] = None
    next_steps: Optional[str] = None
    # Allow extra fields if the LLM returns slightly different JSON
    class Config:
        extra = "allow"

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[dict]

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui_root():
    """Serves a simple HTML UI to interact with the API without using Swagger."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NG12 Assistant</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; background-color: #f4f4f9; }
            .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-top: 0; }
            label { font-weight: bold; display: block; margin-bottom: 0.5rem; color: #34495e; }
            .input-group { display: flex; gap: 10px; margin-bottom: 1rem; }
            input { flex: 1; padding: 0.75rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 0.75rem 2rem; font-size: 1rem; background-color: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #0052a3; }
            #result { margin-top: 1.5rem; background: #fff; padding: 0; border-radius: 4px; display: none; }
            .loading { color: #666; font-style: italic; }
            .result-box { background: #e8f4fd; border-left: 5px solid #0066cc; padding: 1rem; margin-bottom: 1rem; }
            .error-box { background: #fde8e8; border-left: 5px solid #cc0000; padding: 1rem; color: #cc0000; }
            
            /* Floating Chat Styles */
            .chat-widget {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 350px;
                height: 500px;
                background: white;
                border: 1px solid #ccc;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                z-index: 1000;
                display: none; /* Hidden by default */
                overflow: hidden;
            }
            .chat-header {
                background-color: #0066cc;
                color: white;
                padding: 10px 15px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .chat-header button {
                background: none;
                border: none;
                color: white;
                font-size: 1.5rem;
                padding: 0;
                cursor: pointer;
                line-height: 1;
            }
            .chat-header button:hover { background: none; color: #ddd; }

            .chat-toggle-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background-color: #0066cc;
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 1001;
                font-size: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0;
            }
            .chat-toggle-btn:hover { background-color: #0052a3; }

            /* Chat Content Styles */
            #chat-history { flex: 1; overflow-y: auto; padding: 1rem; background: #f9f9f9; }
            .chat-msg { margin-bottom: 10px; padding: 10px; border-radius: 8px; max-width: 85%; font-size: 0.9rem; }
            .user-msg { background-color: #0066cc; color: white; margin-left: auto; text-align: right; }
            .bot-msg { background-color: #e9e9e9; color: #333; margin-right: auto; }
            .citation { font-size: 0.8em; color: #555; margin-top: 5px; border-top: 1px solid #ccc; padding-top: 5px; }
            
            .chat-input-area {
                padding: 10px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 5px;
            }
            .chat-input-area input { margin-bottom: 0; font-size: 0.9rem; }
            .chat-input-area button { padding: 0.5rem 1rem; font-size: 0.9rem; }
            .chat-footer { background: #f1f1f1; padding: 5px; text-align: center; font-size: 0.8rem; }
            .chat-footer button { background: none; border: none; color: #666; padding: 0; font-size: 0.8rem; text-decoration: underline; font-weight: normal; }
            .chat-footer button:hover { background: none; color: #333; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🩺 NG12 Cancer Referral Assistant</h1>
            
            <!-- Assessment Section -->
            <div id="assessment">
                <label for="patient_id">Enter Patient ID:</label>
                <div class="input-group">
                    <input list="patients" id="patient_id" placeholder="Type or select ID (e.g., p001)">
                    <datalist id="patients"></datalist>
                    <button onclick="runAssessment()">Assess Patient</button>
                </div>
                <div id="result"></div>
            </div>
        </div>

        <!-- Floating Chat Button -->
        <button class="chat-toggle-btn" onclick="toggleChat()" title="Chat with Guidelines">💬</button>

        <!-- Floating Chat Widget -->
        <div class="chat-widget" id="chat-widget">
            <div class="chat-header">
                <span>NG12 Assistant</span>
                <button onclick="toggleChat()">×</button>
            </div>
            <div id="chat-history">
                <div class="chat-msg bot-msg">Hello! I can answer questions about the NG12 guidelines. What would you like to know?</div>
            </div>
            <div class="chat-input-area">
                <input type="text" id="chat-input" placeholder="Ask a question...">
                <button onclick="sendChat()">Send</button>
            </div>
            <div class="chat-footer">
                 <button onclick="clearChat()">Clear History</button>
            </div>
        </div>

        <script>
            const SESSION_ID = "session_" + Math.random().toString(36).substr(2, 9);

            function toggleChat() {
                const widget = document.getElementById('chat-widget');
                if (widget.style.display === 'none' || widget.style.display === '') {
                    widget.style.display = 'flex';
                    setTimeout(() => document.getElementById('chat-input').focus(), 100);
                } else {
                    widget.style.display = 'none';
                }
            }

            // Load patient IDs for autocomplete
            fetch('/patients').then(r => r.json()).then(ids => {
                const dl = document.getElementById('patients');
                ids.forEach(id => {
                    const opt = document.createElement('option');
                    opt.value = id;
                    dl.appendChild(opt);
                });
            });

            async function runAssessment() {
                const id = document.getElementById('patient_id').value.trim();
                if (!id) return;
                
                const resDiv = document.getElementById('result');
                resDiv.style.display = 'block';
                resDiv.innerHTML = '<p class="loading">Consulting NICE guidelines...</p>';
                
                try {
                    const response = await fetch(`/assess/${id}`);
                    const data = await response.json();
                    
                    if (!response.ok) {
                        resDiv.innerHTML = `<div class="error-box"><strong>Error:</strong> ${data.detail || 'Unknown error'}</div>`;
                        return;
                    }

                    let html = `<div class="result-box"><h3>Assessment for ${id}</h3>`;
                    if (data.recommendation) html += `<p><strong>Recommendation:</strong> ${data.recommendation}</p>`;
                    if (data.next_steps) html += `<p><strong>Next Steps:</strong> ${data.next_steps}</p></div>`;
                    
                    if (data.guideline_analysis) html += `<p><strong>Guideline Analysis:</strong><br>${data.guideline_analysis}</p>`;
                    if (data.patient_summary) html += `<hr><p><small><strong>Patient Summary:</strong> ${data.patient_summary}</small></p>`;
                    
                    resDiv.innerHTML = html;
                } catch (e) {
                    resDiv.innerHTML = `<div class="error-box"><strong>Network Error:</strong> ${e.message}</div>`;
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('patient_id').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') runAssessment();
            });

            // Chat Functions
            async function sendChat() {
                const input = document.getElementById('chat-input');
                const msg = input.value.trim();
                if (!msg) return;

                const historyDiv = document.getElementById('chat-history');
                historyDiv.innerHTML += `<div class="chat-msg user-msg">${msg}</div>`;
                input.value = '';
                historyDiv.scrollTop = historyDiv.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: SESSION_ID, message: msg })
                    });
                    const data = await response.json();
                    
                    let botHtml = `<div class="chat-msg bot-msg">${data.answer}`;
                    if (data.citations && data.citations.length > 0) {
                        botHtml += `<div class="citation"><strong>Sources:</strong><br>`;
                        data.citations.forEach(c => {
                            botHtml += `[Page ${c.page}] ${c.excerpt}<br>`;
                        });
                        botHtml += `</div>`;
                    }
                    botHtml += `</div>`;
                    
                    historyDiv.innerHTML += botHtml;
                    historyDiv.scrollTop = historyDiv.scrollHeight;
                } catch (e) {
                    historyDiv.innerHTML += `<div class="chat-msg bot-msg" style="color:red">Error: ${e.message}</div>`;
                }
            }

            async function clearChat() {
                await fetch(`/chat/${SESSION_ID}`, { method: 'DELETE' });
                document.getElementById('chat-history').innerHTML = '<div class="chat-msg bot-msg">History cleared.</div>';
            }

            document.getElementById('chat-input').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') sendChat();
            });
        </script>
    </body>
    </html>
    """

@app.get("/patients", response_model=List[str], tags=["Patients"])
def list_patients():
    """Get a list of all available patient IDs."""
    return get_all_patient_ids()

@app.get("/patients/{patient_id}", response_model=Patient, tags=["Patients"])
def get_patient(patient_id: str):
    """Get details for a specific patient."""
    try:
        return get_patient_by_id(patient_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Patient not found")

@app.post("/patients", status_code=status.HTTP_201_CREATED, tags=["Patients"])
def create_patients(patients: List[Patient]):
    """Add one or more new patients to the database."""
    # Convert Pydantic models to dicts
    patients_data = [p.model_dump() for p in patients]
    count = add_new_patients(patients_data)
    return {"message": f"Added {count} new patients.", "added_count": count}

@app.get("/assess/{patient_id}", response_model=AssessmentResponse, tags=["Assessment"])
def assess(patient_id: str):
    """Run an AI assessment against NG12 guidelines for a specific patient."""
    try:
        return assess_patient(patient_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Patient not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """Chat with the NG12 guidelines using RAG."""
    return chat_with_guidelines(request.session_id, request.message, request.top_k)

@app.get("/chat/{session_id}/history", tags=["Chat"])
def get_history_endpoint(session_id: str):
    """Get conversation history for a session."""
    return get_chat_history(session_id)

@app.delete("/chat/{session_id}", tags=["Chat"])
def delete_history_endpoint(session_id: str):
    """Clear conversation history."""
    clear_chat_history(session_id)
    return {"message": "History cleared"}
