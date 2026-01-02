# Prompt Engineering Strategy

Here is a detailed breakdown of the prompt engineering strategy I employed for the NG12 Clinical Decision Support System. My approach focuses on **safety, groundedness, and structured reasoning**.

## 1. Core Philosophy: "Grounded Clinical Assistant"

I designed the system around three non-negotiable pillars to ensure clinical safety:

### A. The Persona
I explicitly defined the system prompt with: *"You are a clinical decision support assistant."*
*   **Why?** I avoided calling it a "Doctor" to prevent the model from assuming too much authority. The "Assistant" persona primes the model to be objective and deferential to the guidelines, acting as a tool for a GP rather than a replacement.

### B. Strict Grounding (RAG)
In medical AI, hallucination is the biggest risk.
*   **Strategy:** I implemented a strict **RAG** workflow. The model is forbidden from using its internal training data for medical facts.
*   **The Instruction:** *"Answer using ONLY the provided 'Context Guidelines'."*
*   **The Mechanism:** I inject retrieved chunks from the NG12 PDF directly into the prompt context. If the retrieval is empty or irrelevant, the model is instructed to refuse to answer.

### C. Structured Output (JSON Enforcement)
Natural language is hard to parse programmatically.
*   **Strategy:** I force the model to output valid JSON.
*   **Benefit:** This allows the frontend to reliably render specific UI elements (like a red "Urgent Referral" badge or a list of citations) without using regex or fragile text parsing.

---

## 2. Agent 1: The Clinical Assessment Tool (`ng12_agent.py`)

This agent is designed for **deterministic reasoning** based on specific patient data.

### The "Chain of Thought" Strategy
I structured the JSON output to force the model to "think" before it decides.
1.  **`patient_summary`**: First, the model must summarize the patient. This ensures it hasn't missed key details like "smoker" or "age 55".
2.  **`guideline_analysis`**: Second, it must map the symptoms to the specific NG12 criteria *before* making a recommendation. This explicit step reduces logic errors.
3.  **`recommendation`**: Only after the analysis does it output the final decision (e.g., "Urgent Referral").

### Handling Input Data
Instead of converting patient data into a sentence (e.g., "The patient is 40..."), I dump the raw JSON into the prompt.
*   **Why?** LLMs are excellent at parsing JSON. Providing the structured data prevents ambiguity that might arise from natural language descriptions of the patient.

---

## 3. Agent 2: The Conversational Chatbot (`chat_agent.py`)

This agent handles open-ended queries, which requires a more flexible but heavily guarded approach.

### Dynamic Context Injection
To handle citations accurately, I don't just dump text. I format the context like this:
```text
[Chunk 1 | Page 12]
(Text content...)
```
*   **Strategy:** By embedding the page number directly into the context block, the model can "see" the source metadata and include it in the citation field.

### The "Refusal" Guardrail
I added a specific rule: *If the provided context does not contain the answer, state clearly: "I couldn't find support in the NG12 text for that."*
*   **Why?** It is better for a clinical tool to say "I don't know" than to invent a threshold (e.g., "refer if cough lasts 2 weeks" when the guideline says 3).

### Multi-Turn Context
I append the conversation history to the prompt.
*   **Strategy:** `User: ... Assistant: ...`
*   **Why?** This allows the user to ask "What about if they are younger?" without restating the symptoms. The model resolves the coreference ("they") using the history.

---

## 4. Iterative Refinement

During development, I refined the prompts based on failure cases:
*   **Issue:** Initially, the model would give general medical advice (e.g., "Stop smoking") even if not in NG12.
*   **Fix:** I added the constraint *"Do not use outside knowledge"* to the System Prompt.
*   **Issue:** Citations were vague (e.g., "See NG12 guidelines").
*   **Fix:** I enforced a JSON schema for citations requiring specific `page` and `excerpt` fields, forcing the model to pinpoint the evidence.