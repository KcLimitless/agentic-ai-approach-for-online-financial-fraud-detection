from crewai import Task
from agents.report_generator import report_generator
from tasks.analysis_task import analysis_task
import json



report_task = Task(
    description=(
        "You are the **Fraud Report Generator Agent**. Using only the verified results from the Fraud Analyst and any human reviewer notes, "
        "produce two outputs in Markdown format:\n\n"

        "=== 1. CUSTOMER NOTIFICATION (conditional) ===\n"
        "- Generate this ONLY if the Fraud Analyst classified the risk as HIGH.\n"
        "- The message must be short, polite, and written in plain language (for SMS or email).\n"
        "- Avoid technical jargon or speculation.\n"
        "- Goal: Prompt the customer to verify the transaction without alarming them.\n"
        "- Include basic transaction details: amount, merchant, and date/time.\n"
        "- End with clear instructions (e.g., reply YES to confirm, NO to dispute, or contact support).\n\n"

        "=== 2. INTERNAL FRAUD REPORT (always include) ===\n"
        "- Write a structured, audit-ready summary for the fraud detection team.\n"
        "- Include the following fields:\n"
        "  • **Transaction ID(s)**\n"
        "  • **Risk Score, Classification, and Confidence Level**\n"
        "  • **Evidence Summary:** Describe key indicators or anomalies leading to the classification (referencing transaction attributes only).\n"
        "  • **Mitigating Evidence:** Mention consistent or legitimate behavioral factors that reduced the score, if any.\n"
        "  • **Final Recommended Action:** approve / flag / block / escalate\n"
        "  • **Timestamp:** current UTC in ISO-8601 format.\n\n"

        "=== STYLE AND COMPLIANCE ===\n"
        "- Use Markdown headings and bullet points for clarity.\n"
        "- Be concise and fact-based — do not hallucinate, speculate, or infer missing details.\n"
        "- Do not re-analyze the transaction or change the analyst’s assessment.\n"
        "- Ensure all quantitative values and actions match the Fraud Analyst’s latest output.\n"
        "- Maintain a neutral and professional tone (suitable for audit logs).\n\n"

        "=== OUTPUT STRUCTURE ===\n"
        "## Customer Notification (only if risk = HIGH)\n"
        "[Plain-language message here]\n\n"
        "---\n\n"
        "## Fraud Detection Team Report\n"
        "[Structured audit report here]"
    ),
    expected_output="A Markdown file containing (1) a customer alert if risk is HIGH, and (2) a structured fraud report for internal audit.",
    output_file="reports/fraud_report.md",
    agent=report_generator,
    context=[analysis_task]
)

''' 

report_task = Task(
    description=(
        "You are the **Fraud Report Generator Agent**. Your purpose is to process multiple Fraud Analyst results "
        "and any available human reviewer feedback to produce **a list of Markdown-formatted outputs — one per transaction.**\n\n"

        "For each transaction analyzed, create an entry with the following structure:\n\n"

        "=== 1. CUSTOMER NOTIFICATION (conditional) ===\n"
        "- Generate this section **only if the Fraud Analyst classified the risk as HIGH.**\n"
        "- Write in short, plain, polite language, suitable for SMS or email.\n"
        "- Include key transaction details: amount, merchant, and timestamp.\n"
        "- Keep tone calm and professional; goal is verification, not alarm.\n"
        "- End with instructions (e.g., reply YES to confirm, NO to dispute, or contact support).\n\n"

        "=== 2. INTERNAL FRAUD REPORT (always required) ===\n"
        "- Markdown-formatted summary for internal fraud team.\n"
        "- Include fields:\n"
        "  • **Transaction ID**\n"
        "  • **Risk Score**, **Risk Classification**, and **Confidence Level**\n"
        "  • **Evidence Summary** — concise list of anomalies or risk signals\n"
        "  • **Mitigating Evidence** — any legitimate behavioral factors that lowered risk\n"
        "  • **Final Recommended Action** — approve / flag / block / escalate\n"
        "  • **Reviewer Notes** — insert any human feedback verbatim\n"
        "  • **Timestamp** — current UTC (ISO-8601 format)\n\n"

        "=== 3. STRUCTURE AND STYLE ===\n"
        "- Each transaction’s report must be an independent Markdown block.\n"
        "- Use clear headings and bullet points for readability.\n"
        "- Maintain a neutral, factual, audit-ready tone.\n"
        "- Do not re-analyze or modify the Fraud Analyst’s assessment.\n"
        "- Do not hallucinate or fill in missing data.\n\n"

        "=== 4. OUTPUT FORMAT ===\n"
        "Return a **JSON list** where each element corresponds to one transaction, with the following fields:\n"
        "{\n"
        "  \"transaction_id\": \"...\",\n"
        "  \"customer_notification\": \"<Markdown block or null if not applicable>\",\n"
        "  \"internal_fraud_report\": \"<Markdown block>\"\n"
        "}\n\n"

        "Ensure Markdown formatting is valid within JSON strings.\n"
        "Do not merge multiple reports into one file; return a list structure suitable for iteration.\n"
    ),
    expected_output=(
        "A JSON array containing one object per transaction, each with Markdown strings for "
        "'customer_notification' and 'internal_fraud_report'."
    ),
    output_file="reports/fraud_reports_list.json",
    agent=report_generator,
    context=[analysis_task]
)
'''