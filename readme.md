# Agentic AI Approach for Online Fraud Detection

This repository presents an **Agentic AI** system for online credit‑card fraud detection, developed as part of a Master's thesis. It implements a retrieval‑augmented, multi‑agent architecture (retriever, fraud analyst, report generator) orchestrated with CrewAI: semantic search over historical transactions (FAISS) supplies contextual evidence to LLM‑based risk reasoning, an optional human‑in‑the‑loop (HITL) step supports review and feedback, and the pipeline produces audit‑ready reports plus rigorous evaluation (precision, recall, F1, AUC‑PR) and per‑request latency analysis — can be ran without docker, but also reproducible and containerized via Docker/Docker Compose.

---

## 📂 Repository Structure

```
.
├── agents/
│   ├── retriever_agent.py       # Crafts semantic queries to FAISS
│   ├── fraud_analyst.py         # Main fraud analysis logic
│   └── report_generator.py      # Generates alert & audit-ready report
├── tasks/
│   ├── retrieval_task.py        # Defines retrieval task & prompt
│   ├── analysis_task.py         # Analyzes transactions for fraud
│   └── report_task.py           # Generates alert and fraud report
├── tools/
│   ├── vector_search_tool.py    # FAISS-based search tool
├── data/
│   └── customer_transaction_history.csv
├── reports/
│   └── all_results.json         # Model predictions with latency
├── evaluation/
│   ├── evaluator.py             # Evaluation metrics & visualization
│   ├── ground_truth1.json       # Ground truth labels
│   ├── evaluation_metrics.json  # Computed metrics (precision, recall, F1, AUC-PR, latency)
│   ├── evaluation_metrics.md    # Markdown report
│   ├── evaluation_confusion_matrix.png
│   └── evaluation_aucpr.png
├── transaction/
│   └── sample.py                # Sample transaction for testing
├── main.py                      # Orchestrates CrewAI pipeline
├── Dockerfile                   # Container setup
├── docker-compose.yml           # Services + interactive HITL
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment containing all dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

---

## 🛠️ Technology Stack

- **CrewAI** for multi-agent orchestration
- **LangChain** + **FAISS** for vector retrieval
- **OpenAI API** for LLM reasoning
- **Python 3.10+**, Pydantic for schemas
- **Scikit-learn, NumPy** for evaluation metrics
- **Matplotlib, Seaborn** for visualization
- **Docker/Docker Compose** for containerized, interactive deployment

---

## 📦 Dependencies

The project requires the following key dependencies (see `requirements.txt` for the full list):

- Python 3.10+
- CrewAI
- LangChain and langchain-community components
- FAISS for vector search
- OpenAI API client (`langchain_openai`)
- Pydantic (v2)
- NumPy, Scikit-learn (for evaluation)
- Matplotlib, Seaborn (for visualization)
- Docker and Docker Compose (for containerized deployment)

To install locally:

```bash
pip install -r requirements.txt
```

Or using conda (if you have environment.yml):

```bash
conda env create -f environment.yml
conda activate fraud-det
```

---

## 🚀 Quick Start

1. **Set your OpenAI API key**\
   Edit the existing `.env` file (already in the project) and add:

   ```bash
   OPENAI_API_KEY=your_real_key_here
   ```

2. **Run without Docker (simple method)**\
   Install dependencies and run the app directly:

   ```bash
   pip install -r requirements.txt
   python main.py
   ```

   This will run locally, using your environment variables from `.env`.

3. **Run with Docker (recommended interactive mode)**\
   For human-in-the-loop (HITL) review and interactive feedback:

   ```bash
   docker-compose run --rm --service-ports -it fraud_detection_ai python main.py
   ```

   - This runs the `fraud_detection_ai` container interactively, with ports exposed.
   - Press **Enter** to approve or type feedback when prompted.

4. **Alternative: Run as a service (non-interactive)**\
   If you prefer to run the entire stack as defined in `docker-compose.yml` without interactive input:

   ```bash
   docker-compose up --build
   ```

   - This uses the default command from the compose file and launches all services.
   - Useful for automated or non-interactive runs.

5. **Run without human input (auto-approve mode)**\
   Edit `tasks/analysis_task.py` and set `human_input=False`, then run with either method above.

6. **View Results**

   - Customer alert appears in console (a brief notification message).
   - Both the short user-facing alert and the detailed fraud team report are currently saved together in `reports/fraud_report.md` and accumulated in `reports/all_results.md`.
   - Model predictions (with latency) are saved to `reports/all_results.json`.

---

## 📊 Evaluation & Metrics

The repository includes a comprehensive evaluation pipeline (`evaluation/evaluator.py`) to compute and visualize model performance metrics.

### Usage

**Evaluate with manually using your TN, FP, FN and TP** (computes latency from `all_results.json` if present)
```bash
python evaluation/evaluator.py --manual-conf 32 1 3 64 --results reports/all_results.json --ground-truth evaluation/ground_truth1.json
```

- Arguments: `--manual-conf TN FP FN TP` (True Negatives, False Positives, False Negatives, True Positives)
- When using `--manual-conf`, the evaluator computes metrics from confusion counts and optionally extracts AUC-PR and latency statistics from the results file.

### Outputs

The evaluator generates:
- **evaluation_metrics.json** – Precision, Recall, F1-Score, Confusion Matrix, AUC-PR (single point), Latency Stats
- **plot_pr_curve.py** - Precision–Recall Curve with AUCPR = 0.9889
- **evaluation_metrics.md** – Human-readable markdown report
- **evaluation_confusion_matrix.png** – Heatmap visualization of confusion matrix
- **evaluation_aucpr.png** – Precision-Recall curve (full curve if scores available, single point otherwise)

### Example Output

```json
{
  "precision": 0.9846,
  "recall": 0.9552,
  "f1_score": 0.9697,
  "confusion_matrix": [[32, 1], [3, 64]],
  "counts": {
    "total_evaluated": 100,
    "total_positive": 67,
    "total_predicted_positive": 65
  },
  "aucpr": 0.9889,
  "latency": {
    "mean": 50.40,
    "median": 48.84,
    "min": 30.81,
    "max": 86.74,
    "std": 11.72
  }
}
```

### Latency Field Support

The evaluator extracts latency from result objects using these field names (in order of precedence):
- `latency_seconds` – latency in seconds
- `latency_ms` – latency in milliseconds (auto-converted to seconds)
- `latency` – generic latency field

Ensure your `all_results.json` includes per-request latency for accurate latency statistics.

---

## 🏗️ Architecture

This project uses an agentic, retrieval‑augmented architecture designed for accurate, auditable, and reproducible online fraud decisions. Responsibilities are separated into lightweight components so each part can be developed, tested, and scaled independently.

- High-level flow: Client/User → Ingest → Retrieval → LLM-based analysis → Optional human review (HITL) → Reporting → Evaluation.
- Design goals: interpretability (evidence + audit report), rigorous, end-to-end evaluation (precision/recall/F1, AUC‑PR, per-request latency), and reproducibility (Docker/Docker Compose).

### Components & Responsibilities
- **Client / Ingest** — Entrypoint (`main.py`): accepts a transaction and submits it to the pipeline.
- **Orchestration (CrewAI)** — Coordinates agent flow, retries, error handling and task distribution between agents.
- **Retriever Agent** — Builds semantic queries and retrieves similar historical transactions (via FAISS embeddings) to provide contextual evidence.
- **Vector DB (FAISS)** — Stores transaction embeddings and metadata for fast semantic lookup.
- **Fraud Analyst Agent** — Combines context + heuristics based analysis, risk score and explanation, and outputs a classification plus supporting evidence and recommendation.
- **Human-in-the-Loop (HITL)** — Optional approval/feedback step that can overwrite or confirm analyst decisions.
- **Report Generator** — Produces the customer alert and an audit-ready report, writing per-request results to `reports/all_results.json`.
- **Evaluation Module** — `evaluation/evaluator.py` computes confusion matrix, precision/recall/F1, AUC‑PR, and latency statistics; generates visualizations (PR curve, confusion matrix heatmap).
- **Deployment** — `Docker/Docker Compose` enables reproducible, interactive runs (HITL), while the system can also be run locally without containers.

### Data & Artifacts
- Input schema: `transaction` objects (see `transaction/sample.py`).
- Runtime outputs: `reports/all_results.json`, `reports/fraud_report.md`.
- Evaluation outputs: `evaluation/evaluation_metrics.json`, `evaluation/evaluation_metrics.md`, `evaluation_aucpr.png`, `evaluation_confusion_matrix.png`.

### Diagrams
Below are the high/low-level architecture of the system.

#### Low-level Architecture diagram
![Architecture diagram](images/solution-concept.png)

#### High-level Architecture diagram
![Request sequence](images/solution-concept2.png)

---

## ⚙️ Configuration

```python
CSV_PATH = "data/..."
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

- **Environment Variables**
  - `OPENAI_API_KEY` (required)
  - `HITL_MODE` (optional) – if set to `false`, skips human_input.

---

## 🧪 Testing Sample Transactions

The application includes a predefined `sample_transaction` (in `transaction/sample.py`) that is automatically analyzed on each run, so you can immediately test the system. Currently, the transaction tested is:

```python
sample_transaction = {
    "transaction_id": "b40c083614de1a1c8c8835d4bb01b380",
    "amount": 60.00,
    "category": "gas_transport",
    "location": (35.5494, -80.4226),
    "merchant": "Raynor Feest and Miller",
    "timestamp": "2025-04-15T15:47:12Z",
    "user_id": "4065133387262473",
    "merch_lat": 35.219240,
    "merch_long": -80.402563,
    "trans_time": "15:47:12"
}
```

This transaction is processed by the Retriever Agent (for context retrieval), the Fraud Analyst Agent (for fraud analysis, risk scoring and classification), and the Report Generator Agent (for alerts and final reports generation). The app retrieves any historical context available, analyzes this sample together with the input transaction, and produces both a console alert and a detailed markdown report.

To run the test with this sample transaction:

```bash
python main.py
```

or using Docker (interactive HITL):

```bash
docker-compose run --rm --service-ports -it fraud_detection_ai python main.py
```

A brief alert will be shown in the console, and both the alert and detailed fraud report will be saved to `reports/fraud_report.md`.

---

## 📄 Sample Output

Both the short user-facing alert and the detailed fraud team report are currently saved together in `reports/fraud_report.md`, so you can find both outputs in one file.

**Sample Alert:**

   ```text
   ALERT: Suspicious transaction detected for User 4192832764832. Risk Score: 85 (HIGH). Transaction blocked pending review.
   ```

**Sample Fraud Report:**

```markdown
# Fraud Report

**Transaction ID:** f82dfd045c91110964bcedd1dc0df84e  
**Risk Score:** 85 (HIGH)  
**Key Evidence:**
- Amount (320.99) ≫ user average (45.60)
- Merchant location ≠ home city
**Recommendation:** Block & Escalate  
**Reasoning:** Rapid location deviation and high amount.
```

**Sample Evaluation Metrics (from evaluator.py):**

See `evaluation/evaluation_metrics.md` for formatted metrics report, and `evaluation/evaluation_metrics.json` for raw metrics data.

---

## 🔄 Workflow

1. **Prepare data** – Ensure `data/customer_transaction_history.csv` is populated.
2. **Run fraud detection** – Execute `python main.py` (or via Docker).
3. **Review results** – Check console output and `reports/fraud_report.md`.
4. **Evaluate model** – Run `python evaluation/evaluator.py` with your results.
5. **Analyze metrics** – Review `evaluation/evaluation_metrics.json` and visualizations.

---

## 📝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/...`)
3. Commit changes (`git commit -m "feat: ..."`)
4. Push (`git push origin feature/...`)
5. Open a Pull Request

---

## 📚 References

1. Lewis *et al.* (2020). *Retrieval-Augmented Generation for Knowledge‑Intensive NLP Tasks*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
2. Carcillo, Le Borgne, Caelen & Kessaci (2019). *Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection*. [Wikipedia Data Analysis for Fraud Detection](https://en.wikipedia.org/wiki/Data_analysis_for_fraud_detection)
3. Akca *et al.* (2023). *A Systematic Review of Intelligent Systems and Analytic Applications in Credit Card Fraud Detection*. [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/15/3/1356)
4. Bonkoungou, Roy & Ako (2024). *Credit Card Fraud Detection Using ML Techniques*. [SpringerLink](https://link.springer.com/chapter/10.1007/978-981-99-9811-1_2)
5. Business Insider (2025). *At Mastercard, AI is helping to power fraud‑detection systems*. [Business Insider](https://www.businessinsider.com/mastercard-ai-credit-card-fraud-detection-protects-consumers-2025-5)

---

Welcome any feedback!

