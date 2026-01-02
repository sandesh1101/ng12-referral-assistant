# NG12 Cancer Referral Assistant

This project implements a clinical decision support system based on the NICE NG12 guidelines. It includes a structured patient assessment tool and a conversational AI agent (Chat Mode) grounded in the guideline text.

## Features
*   **Assessment Mode:** Analyzes specific patient data to recommend referrals based on NG12 criteria.
*   **Chat Mode:** A conversational interface to ask general questions about the guidelines (e.g., "What are the red flags for lung cancer?").
*   **RAG Pipeline:** Uses Retrieval-Augmented Generation to ensure answers are cited from the NG12 PDF.

## Performance Optimizations
*   **Latency Reduction:** Implemented LRU caching, reduced RAG context size, and added a **server startup warmup** routine to handle the initial connection overhead (cold start) before the first user request.
*   **Model Efficiency:** Selected `gemini-2.5-flash` for its balance of speed and reasoning capability, ensuring sub-second inference times for real-time interactions.
*   **Asynchronous Design:** Leveraged FastAPI's async capabilities to handle concurrent user requests without blocking, improving throughput under load.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables:**
    Create a `.env` file in the root directory:
    ```Add your GCP project ID here:
    ```env
    GCP_PROJECT_ID=your-project-id
    GCP_LOCATION=us-central1
    GOOGLE_API_KEY=your-api-key
    ```
3.  **Authentication (Important):**
    You must be authenticated with Google Cloud to use Vertex AI.
    ```bash
    gcloud auth application-default login
    ```
4.  **Ingest Data: (important)**
    If running for the first time, build the vector database:
    ```bash
    python scripts/ingest_ng12_pdf.py
    ```

## Running the Application

### 1. Local Execution
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
*   **Access the UI:** Open `http://127.0.0.1:8000/` in your browser.
*   **Use Chat:** Click the floating 💬 icon in the bottom-right corner to open the chat widget.
*   **API Docs:** Swagger UI is available at `http://127.0.0.1:8000/docs`.

### 2. Docker Execution(Please complete the setup 4 steps to run the docker container)

Build the container:
```bash
docker build -t ng12-app .
```

Run the container (ensure you pass Google Cloud credentials):

For Windows:
```bash
docker run -p 8000:8000 `
  -e GCP_PROJECT_ID=your-project-id `
  -e GCP_LOCATION=GCP_LOCATION `
  -e GOOGLE_API_KEY=Your-API-key `
  -v "$env:APPDATA\gcloud:/root/.config/gcloud" `
  ng12-app
```
Open http://localhost:8000/ in your browser. You can enter the patient_ID and also chat with the chatbot.


For Mac/Linux:
```bash
# Assuming you have Application Default Credentials (ADC) set up locally
docker run -p 8000:8000 \
  -e GCP_PROJECT_ID=your-project-id \
  -e GCP_LOCATION=GCP_LOCATION \
  -e GOOGLE_API_KEY=your-api-key \
  -v ~/.config/gcloud:/root/.config/gcloud \
  ng12-app
```