

# 📘 Business License Advisor

## Project Overview

This project is an **AI-powered compliance assistant** designed to help restaurant owners in Israel understand the business licensing requirements relevant to them.

The system takes a **short questionnaire** (business size, seating capacity, free-text features such as “gas,” “meat,” or “delivery”), and produces a **personalized compliance report** in plain Hebrew, summarizing the applicable regulations.

<p align="center">
  <img src="photos\projectphoto.png" alt="Screenshot 1" width="45%"/>
  <img src="photos\LLMrespond.png" alt="Screenshot 2" width="45%"/>
</p>
---

## Key Points

* The entire project was **built solely with ChatGPT (GPT-5)** and the **OpenAI API**.
* The **data extraction** itself (PDF → JSON) was also performed with the OpenAI API, using the `data_builder` script.
* The final application integrates **FastAPI backend** + **React frontend** + **OpenAI models** for report generation.
* Future plan: migrate logic to **LangGraph** and use **RAG (Retriever-Augmented Generation)** for more precise rule matching.

---

## Architecture

* **Frontend**: React (Vite + TypeScript) – questionnaire + report display
* **Backend**: FastAPI (Python) – APIs for:

  * Rule loading & filtering
  * Free-text feature matching (semantic embeddings)
  * LLM integration for report generation
* **Data Pipeline**:

  * `scripts/data_builder_langchain.py` – converts the official PDF (regulations) into structured JSON Using LLM (`rules_flat.json`)
  * `scripts/build_feature_index.py` – builds embeddings index (`terms.json` + `embeddings.npz`)
* **AI**:

  * `text-embedding-3-small` for semantic similarity
  * `gpt-5` for report generation

---

## Installation & Setup

### Requirements

* Python 3.11+
* Node.js 18+
* An **OpenAI API key** (`OPENAI_API_KEY`)

### Backend Setup

```bash
cd business-license-advisor
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Generate the rules JSON (`rules_flat.json`)

Run the **data builder** (requires OpenAI key):

```bash
$env:OPENAI_API_KEY = <YOU API KEY>
python scripts/data_builder_langchain.py \
  --pdf "data/raw/18-07-2022_4.2A.pdf" \
  --out_dir "data/curated" \
  --model "gpt-5-mini"
```

This will parse the regulation PDF and produce:

* `data/curated/rules_flat.json` – all structured rules
* `data/curated/debug/` – optional intermediate debug outputs

---

### 2. Build the feature embeddings index

```bash
$env:OPENAI_API_KEY = <YOU API KEY>
python scripts/build_feature_index.py \
  --rules "data/curated/rules_flat.json" \
  --out_dir "data/index"
```

This produces:

* `data/index/terms.json` – extracted features (terms)
* `data/index/embeddings.npz` – vector embeddings for semantic matching

---

## Running the Application

### Start Backend

```bash
$env:OPENAI_API_KEY = <YOU API KEY>
uvicorn server.server:app --reload --port 8000
```

### Start Frontend

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to view the UI.

---

## API Usage

### `POST /report`

Example request:

```json
{
  "size_m2": 120,
  "seats": 80,
  "features_text": "gas, CCTV cameras, frying, no alcohol",
  "top_n_rules": 60,
  "debug": true
}
```

Response (truncated):

```json
{
  "selected_count": 60,
  "matches": [...],
  "sample_rules": [...],
  "report": "### Business Summary ... (Hebrew report)"
}
```

---

## 🧠 AI Usage & Key Prompts

This project was fully developed using ChatGPT (GPT-5) and the OpenAI API, both for data preparation and for building the application logic.

AI Roles

Data Builder

We used the OpenAI API to read the official PDF (18-07-2022_4.2A.pdf) and convert it into a structured JSON (rules_flat.json).

Embeddings (text-embedding-3-small) were created for semantic search of business features.

Compliance Report Generation

A GPT model (gpt-4o-mini) takes the filtered rules + user input (size, seats, features) and produces a custom compliance report in plain Hebrew.

System Development

ChatGPT was used iteratively to design, refactor, and simplify the FastAPI server, the React frontend, and the data processing scripts.

## 🔑 Key Prompts Used

Below are the most significant prompts that guided the AI-driven development:

1. Data Extraction

“Build me a script that takes the regulation PDF and outputs structured JSON rules (id, category, conditions, features, text). Add debugging so I can see the raw LLM output every 10 pages.”

2. Feature Matching

“Because features are free-text, let’s generate a set of canonical terms from the dataset, build embeddings for them, and then match user input semantically (even in Hebrew).”

3. Compliance Report Prompt

System prompt for the LLM:

You are a helpful assistant that writes plain Hebrew compliance summaries 
for restaurant licensing. Use only the provided rules context. 
Be concise, structured, and clear. Avoid legalese; explain in business language. 
Group by category and add actionable next steps.


User prompt template:

Business data:
- Size: {size_m2} m²
- Seats: {seats}
- Free-text features: {features_text}

Relevant rules:
{rules_ctx}

Task: Write a personalized compliance report in Hebrew with:
1) Business summary
2) Mandatory requirements grouped by category
3) Practical action recommendations (immediate / medium / nice-to-have)
4) References to rule numbers

4. Development Assistance

“Please rewrite this server script in the simplest, cleanest way possible.
Keep only the necessary functions, and ensure stability.”

“Now generate a minimal React UI that collects user input (size, seats, features) and calls the FastAPI endpoint.”

---

## Future Improvements

* Switch to **LangGraph** for deterministic flow orchestration.
* Add **RAG pipeline** for more accurate multi-rule retrieval.
* Improve semantic feature extraction for Hebrew.
* Add **PDF export** of compliance reports.

---

## Credits

* **Development**: Built entirely with **ChatGPT (GPT-5)**.
* **Data Extraction & LLM Reports**: Powered by **OpenAI API**.
* **PDF Regulations Source**: *18-07-2022\_4.2A.pdf* (Business Licensing for Restaurants).


