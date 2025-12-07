'''
from crewai import Crew
from tasks.retrieval_task import retrieval_task
from tasks.analysis_task import analysis_task
from tasks.report_task import report_task
from agents.retriever_agent import retriever_agent
from agents.fraud_analyst import fraud_analyst
from agents.report_generator import report_generator

def run_pipeline():
    fraud_detection_crew = Crew(
        agents=[retriever_agent, fraud_analyst, report_generator],
        tasks=[retrieval_task, analysis_task, report_task],
        verbose=True
    )
    result = fraud_detection_crew.kickoff()
    print("\n--- Final Output ---\n")
    print(result)

if __name__ == "__main__":
    run_pipeline()

    


import os
import json
import time
from datetime import datetime
from crewai import Crew
from tasks.retrieval_task import retrieval_task
from tasks.analysis_task import analysis_task
from tasks.report_task import report_task
from agents.retriever_agent import retriever_agent
from agents.fraud_analyst import fraud_analyst
from agents.report_generator import report_generator

from transaction.sample import sample_transaction

# Output directory
OUTPUT_PATH = "reports/all_results.json"

def load_existing_results():
    """Load previous analysis results if file exists, else start a new list."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_results(all_results):
    """Save the updated results list back to JSON."""
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

def run_pipeline():
    """Run the agentic fraud detection pipeline for a single transaction."""
    #start_time = time.time()

    # Create crew for this transaction
    fraud_detection_crew = Crew(
        agents=[retriever_agent, fraud_analyst, report_generator],
        tasks=[retrieval_task, analysis_task, report_task],
        verbose=True
    )

    # Inject the current transaction into the context if your agents expect it
    #fraud_detection_crew.context = {"sample_transaction": transaction_data}

    # Run the pipeline
    result = fraud_detection_crew.kickoff()

    #end_time = time.time()
    #latency = round(end_time - start_time, 3)

    # Wrap with metadata for traceability
    #report_entry = {
    #    "transaction_id": sample_transaction.get("transaction_id"),
    #    "timestamp": datetime.utcnow().isoformat(),
    #    "latency_seconds": latency,
    #    "result": result
    #}

    # Load previous results, append new one, and save
    #all_results = load_existing_results()
    #all_results.append(report_entry)
    #save_results(all_results)

    #print(f"\n✅ Transaction {sample_transaction.get('transaction_id')} processed in {latency}s")
    #return result

    print("\n--- Final Output ---\n")
    print(result)


if __name__ == "__main__":
    # Run and log results
    run_pipeline()



import os
import json
import time
from datetime import datetime, UTC
from crewai import Crew
from tasks.retrieval_task import retrieval_task
from tasks.analysis_task import analysis_task
from tasks.report_task import report_task
from agents.retriever_agent import retriever_agent
from agents.fraud_analyst import fraud_analyst
from agents.report_generator import report_generator
from transaction.sample import sample_transaction  # your single transaction

OUTPUT_PATH = "reports/all_results.json"


def load_existing_results():
    """Load previous pipeline results if available."""
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_results(all_results):
    """Save results as human-readable Markdown-friendly JSON."""
    def preserve_newlines(obj):
        if isinstance(obj, str):
            # Convert escaped newlines (\n) into actual newlines
            return obj.encode('utf-8').decode('unicode_escape')
        elif isinstance(obj, list):
            return [preserve_newlines(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: preserve_newlines(v) for k, v in obj.items()}
        return obj

    readable_results = preserve_newlines(all_results)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(readable_results, f, indent=2, ensure_ascii=False)


def save_results(all_results):
    """Save cumulative results back to JSON."""
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


def serialize_result(result):
    """Safely convert CrewAI result to a JSON-safe format."""
    try:
        if hasattr(result, "to_dict"):
            return result.to_dict()
        elif hasattr(result, "output"):
            return result.output
        else:
            return str(result)
    except Exception as e:
        return f"Serialization error: {e}"


def serialize_result(result):
    """Safely extract and serialize CrewAI results for JSON logging."""
    try:
        # Most modern CrewAI versions return a list or dict in .output
        if hasattr(result, "output") and result.output:
            return result.output
        # Some versions have .raw or .final_output
        elif hasattr(result, "raw") and result.raw:
            return result.raw
        elif hasattr(result, "final_output") and result.final_output:
            return result.final_output
        # Fallback to string conversion (works for print-like output)
        else:
            result_str = str(result)
            # Try to load JSON if it's a JSON string
            try:
                return json.loads(result_str)
            except json.JSONDecodeError:
                return {"text": result_str}
    except Exception as e:
        return {"error": f"Serialization failed: {e}"}


def run_pipeline():
    """Run the complete Agentic Fraud Detection pipeline for a single transaction."""
    start_time = time.time()

    # Create the Crew with agents and tasks
    fraud_detection_crew = Crew(
        agents=[retriever_agent, fraud_analyst, report_generator],
        tasks=[retrieval_task, analysis_task, report_task],
        verbose=True
    )

    # Execute pipeline — all agents & tasks coordinate automatically
    result = fraud_detection_crew.kickoff()

    end_time = time.time()
    latency = round(end_time - start_time, 3)

    # Prepare the structured report entry
    report_entry = {
        "transaction_id": sample_transaction.get("transaction_id"),
        "timestamp": datetime.now(UTC).isoformat(),
        "latency_seconds": latency,
        "result": serialize_result(result)
    }

    # Persist results incrementally
    all_results = load_existing_results()
    all_results.append(report_entry)
    save_results(all_results)

    print("\n✅ Transaction processed successfully!")
    print(f"⏱ Latency: {latency}s")
    #print("\n--- Final Output ---\n")
    #print(json.dumps(report_entry["result"], indent=2))

    pretty_output = json.dumps(report_entry["result"], indent=2, ensure_ascii=False)
    print("\n--- Final Output ---\n")
    print(pretty_output.encode('utf-8').decode('unicode_escape'))

    return report_entry


if __name__ == "__main__":
    run_pipeline()

'''

import os
import json
import time
from datetime import datetime, UTC
from crewai import Crew

# Agents and tasks
from tasks.retrieval_task import retrieval_task
from tasks.analysis_task import analysis_task
from tasks.report_task import report_task
from agents.retriever_agent import retriever_agent
from agents.fraud_analyst import fraud_analyst
from agents.report_generator import report_generator

# Input transaction
from transaction.sample import sample_transaction

# Output paths
JSON_OUTPUT_PATH = "reports/all_results.json"
MARKDOWN_LOG_PATH = "reports/all_results_log.md"


# ==========================================================
#  Utility Functions
# ==========================================================

def load_existing_results():
    """Load previous analysis results if JSON file exists."""
    if os.path.exists(JSON_OUTPUT_PATH):
        try:
            with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_results(all_results):
    """Save structured machine-readable results."""
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

'''
def append_markdown_log(transaction_id, markdown_text):
    """Append formatted multi-line markdown report to a single log."""
    os.makedirs("reports", exist_ok=True)

    with open(MARKDOWN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n# Transaction Report: {transaction_id}\n\n")
        f.write(markdown_text)
        f.write("\n\n" + ("=" * 80) + "\n\n")

    print(f"📄 Markdown log updated → {MARKDOWN_LOG_PATH}")
'''

def append_markdown_log(transaction_id, markdown_text, latency_seconds):
    """Append formatted multi-line markdown report to a single log, including latency."""
    os.makedirs("reports", exist_ok=True)

    with open(MARKDOWN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n# Transaction Report: {transaction_id}\n")
        f.write(f"**Processed At:** {datetime.now(UTC).isoformat()}\n")
        f.write(f"**Latency:** {latency_seconds} seconds\n\n")
        f.write(markdown_text)
        f.write("\n\n" + ("=" * 80) + "\n\n")

    print(f"📄 Markdown log updated → {MARKDOWN_LOG_PATH}")

def serialize_result(result):
    """Extract usable content from CrewAI output."""
    try:
        # Newer CrewAI versions store results as list/dict under .output
        if hasattr(result, "output") and result.output:
            return result.output

        # Some versions use .raw or .final_output
        if hasattr(result, "raw") and result.raw:
            return result.raw
        if hasattr(result, "final_output") and result.final_output:
            return result.final_output

        # Fallback: convert to string, attempt to parse as JSON
        raw_text = str(result)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    except Exception as e:
        return {"error": f"Serialization failure: {e}"}


# ==========================================================
#  Fraud Detection Pipeline
# ==========================================================

def run_pipeline():
    """Run the agentic fraud detection pipeline on one transaction."""
    start_time = time.time()

    # Build the Crew
    fraud_detection_crew = Crew(
        agents=[retriever_agent, fraud_analyst, report_generator],
        tasks=[retrieval_task, analysis_task, report_task],
        verbose=True
    )

    # Execute the Crew pipeline
    result = fraud_detection_crew.kickoff()
    latency = round(time.time() - start_time, 3)

    # Extract structured result from CrewAI output
    structured_result = serialize_result(result)

    # Prepare entry for JSON storage
    report_entry = {
        "transaction_id": sample_transaction.get("transaction_id"),
        "timestamp": datetime.now(UTC).isoformat(),
        "latency_seconds": latency,
        "result": structured_result
    }

    # Save JSON structured data
    results = load_existing_results()
    results.append(report_entry)
    save_results(results)

    # ----------------------------------------------------------
    # Build human-readable markdown (customer + internal report)
    # ----------------------------------------------------------
    if isinstance(structured_result, str):
        markdown_block = structured_result

    elif isinstance(structured_result, list) and len(structured_result) > 0:
        entry = structured_result[0]
        customer_note = entry.get("customer_notification", "").strip()
        internal_report = entry.get("internal_fraud_report", "").strip()

        if customer_note:
            markdown_block = f"{customer_note}\n\n---\n\n{internal_report}".strip()
        else:
            markdown_block = internal_report

    else:
        markdown_block = "⚠️ No report content available."

    # Save markdown log
    #append_markdown_log(report_entry["transaction_id"], markdown_block)
    append_markdown_log(
    report_entry["transaction_id"],
    markdown_block,
    report_entry["latency_seconds"]
    )



    # Console output
    print(f"\n✅ Transaction {report_entry['transaction_id']} processed successfully!")
    print(f"⏱ Latency: {latency}s")
    print("\n--- Final Output ---\n")
    print(markdown_block.encode("utf-8").decode("unicode_escape"))

    return report_entry


# ==========================================================
#  Entry Point
# ==========================================================

if __name__ == "__main__":
    run_pipeline()

