
import json
from functools import lru_cache
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from app.tools.patient_db import get_patient_by_id
from app.tools.guideline_rag import search_guidelines
from app.config import PROJECT_ID, LOCATION, LLM_MODEL

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel(LLM_MODEL)

SYSTEM_PROMPT = """You are a clinical decision support assistant.
Analyze the patient's symptoms against the provided NICE guidelines.

Return the output in valid JSON format with the following keys:
- patient_summary: A brief summary of age, symptoms, and risk factors.
- guideline_analysis: How the guidelines apply to this specific patient (cite specific criteria and page numbers).
- recommendation: The clinical recommendation (e.g., Urgent Referral, Routine Referral, Safety Netting, or No Referral).
- next_steps: Specific actions for the GP (e.g., "Order Chest X-Ray", "Refer via 2WW").
"""

def warmup_agent():
    """Performs a dummy call to warm up the model and vector store connection."""
    try:
        # Warmup embedding
        search_guidelines("warmup", k=1)
        # Warmup LLM
        model.generate_content("warmup")
    except Exception as e:
        print(f"Warmup warning: {e}")

@lru_cache(maxsize=32)
def assess_patient(patient_id: str):
    """
    Retrieves patient data and generates a clinical assessment using NG12 guidelines.
    """
    patient = get_patient_by_id(patient_id)
    chunks = search_guidelines(", ".join(patient["symptoms"]), k=2)
    
    # Format context with page numbers for better citations
    context_list = []
    for c in chunks:
        page = c.metadata.get("page", 0)
        display_page = page + 1 if isinstance(page, int) else page
        context_list.append(f"[Page {display_page}]\n{c.page_content}")
    context = "\n\n".join(context_list)

    user_prompt = f"""Patient: {json.dumps(patient)}
Guidelines:
{context}
"""
    
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        generation_config=GenerationConfig(response_mime_type="application/json")
    )
    
    try:
        text = response.text
        data = json.loads(text)

        # Ensure patient_summary is a string to satisfy Pydantic model
        if "patient_summary" in data and not isinstance(data["patient_summary"], str):
            data["patient_summary"] = str(data["patient_summary"])

        return data
    except ValueError:
        # Handles cases where response.text raises "Content has no parts" (Safety/Recitation)
        return {
            "recommendation": "Unable to assess",
            "guideline_analysis": "The model refused to generate a response due to safety filters.",
            "next_steps": "Please review the patient data manually."
        }
    except json.JSONDecodeError:
        return {"assessment": response.text}
