import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from app.config import PROJECT_ID, LOCATION, LLM_MODEL
from app.tools.guideline_rag import search_guidelines
import json

# Simple in-memory session store
# Structure: { session_id: [ {"role": "user", "content": "..."}, {"role": "model", "content": "..."} ] }
sessions = {}

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(LLM_MODEL)

SYSTEM_PROMPT = """You are an expert assistant for the NICE NG12 guidelines (Suspected Cancer: Recognition and Referral).
Your goal is to answer user questions ACCURATELY based ONLY on the provided context chunks.

RULES:
1. GROUNDING: Answer using ONLY the provided "Context Guidelines". Do not use outside knowledge.
2. REFUSAL: If the provided context does not contain the answer, state clearly: "I couldn't find support in the NG12 text for that."
3. CITATIONS: You must cite the specific page number for every claim you make.
4. FORMAT: Return your response in valid JSON format.

JSON SCHEMA:
{
  "answer": "Your natural language answer here...",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": 12,
      "chunk_id": "chunk_1",
      "excerpt": "Direct quote or summary of the specific rule..."
    }
  ]
}
"""

def get_chat_history(session_id: str):
    """Retrieves the conversation history for a given session ID."""
    return sessions.get(session_id, [])

def clear_chat_history(session_id: str):
    """Clears the conversation history for a specific session ID."""
    if session_id in sessions:
        del sessions[session_id]

def chat_with_guidelines(session_id: str, user_message: str, top_k: int = 5):
    """
    Processes a user message using RAG against NG12 guidelines.

    Args:
        session_id: Unique identifier for the chat session.
        user_message: The user's query.
        top_k: Number of context chunks to retrieve.

    Returns:
        dict: Contains session_id, answer, and citations.
    """
    # 1. Retrieve Context
    # We search based on the user message.
    docs = search_guidelines(user_message, k=top_k)

    # Deduplicate docs based on content to prevent repetitive context
    unique_docs = []
    seen_content = set()
    for doc in docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)

    context_text = ""
    for i, doc in enumerate(unique_docs):
        # PyPDFLoader usually uses 0-indexed pages. We add 1 for display.
        page = doc.metadata.get("page", 0)
        display_page = page + 1 if isinstance(page, int) else page
        chunk_id = f"chunk_{i+1}"
        context_text += f"[ID: {chunk_id} | Page {display_page}]\n{doc.page_content}\n\n"

    # 2. Manage History
    if session_id not in sessions:
        sessions[session_id] = []
    
    history = sessions[session_id]
    
    # 3. Construct Prompt
    # We construct the full prompt manually to ensure RAG context is prioritized
    messages_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        messages_str += f"{role}: {msg['content']}\n\n"
    
    final_prompt = f"""{SYSTEM_PROMPT}

CONTEXT GUIDELINES:
{context_text}

CONVERSATION HISTORY:
{messages_str}
User: {user_message}
Assistant:"""

    # 4. Generate Response
    try:
        response = model.generate_content(
            final_prompt,
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        response_text = response.text
        response_json = json.loads(response_text)
    except Exception as e:
        # Fallback if model fails or returns invalid JSON
        print(f"Error generating chat response: {e}")
        response_json = {
            "answer": "I encountered an error processing your request. Please try again.",
            "citations": []
        }

    # 5. Update History
    sessions[session_id].append({"role": "user", "content": user_message})
    sessions[session_id].append({"role": "model", "content": response_json.get("answer", "")})

    # Deduplicate citations in the response
    raw_citations = response_json.get("citations", [])
    unique_citations = []
    seen_citations = set()
    for cit in raw_citations:
        # Use page and excerpt as a unique signature
        sig = (cit.get("page"), cit.get("excerpt"))
        if sig not in seen_citations:
            unique_citations.append(cit)
            seen_citations.add(sig)

    return {
        "session_id": session_id,
        "answer": response_json.get("answer", ""),
        "citations": unique_citations
    }