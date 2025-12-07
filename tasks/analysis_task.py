from crewai import Task
from agents.fraud_analyst import fraud_analyst
from tasks.retrieval_task import retrieval_task
from transaction.sample import sample_transaction


'''
analysis_task = Task(
    description=(
        f"You are analyzing this **specific target transaction**, you must not hallucinate or make assumptions:\n"
        f"{sample_transaction}\n\n"
        "You will also be given **retrieved historical transactions as context** from the Query Strategist if available, "
        "but DO NOT classify or score the retrieved transactions individually. Instead:\n\n"
        "Steps:\n"
        "1. Use the retrieved transactions and heuristics to understand the user's behavior, anomalies, or repeating fraud patterns.\n"
        "2. Focus your fraud analysis solely on the target transaction above.\n"
        "3. If retrieved context is sparse or absent, fall back to intrinsic checks (amount, geo, time, merchant risk, etc.) and reasoning.\n"
        "   - Make it clear that no context was available if that is the case.\n"
        "4. Produce a single fraud risk assessment for the target transaction, including:\n"
        "   - Risk Score (0–100)\n"
        "   - Fraud Classification (HIGH/LOW)\n"
        "   - Confidence Level (High/Medium/Low)\n"
        "   - Clear and comprehensive reason for fraud risk assessment results\n"
        "   - Recommended Action (approve, flag, block, escalate)\n"
        "   - Clear, step-by-step reasoning.\n\n"
        "5. Ignore the fraud label attached to the merchant names from the sample transaction and retrieved context in your evaluation, it does not mean that the merchant is fraudulent.\n"
        "**Do not output classifications for each retrieved transaction. They are supporting evidence only.**"
    ),
    expected_output="A single comprehensive but concise fraud assessment report for the target transaction.",
    agent=fraud_analyst,
    context=[retrieval_task],
    human_input=True
)
'''

analysis_task = Task(
    description=(
        f"You are analyzing the **specific target transaction** below. Your objective is to provide a *balanced, evidence-driven* fraud risk assessment.\n"
        f"Transaction under analysis:\n{sample_transaction}\n\n"
        "You may be given **retrieved historical transactions as context** from the Query Strategist. These represent past behavior for comparison only.\n"
        "Do NOT classify or score the retrieved transactions individually; use them only to establish behavioral baselines.\n\n"

        "=== ANALYSIS INSTRUCTIONS ===\n"
        "1. Begin by identifying the user's *normal behavioral patterns* from retrieved history: spending frequency, merchant types, transaction times, locations, and amounts.\n"
        "   - If retrieved data are sparse or empty, explicitly state that and rely on general heuristics (e.g., typical consumer patterns).\n"
        "2. Compare the target transaction to those baselines:\n"
        "   - Determine whether differences are *within reasonable variation* or *truly anomalous*.\n"
        "   - Minor deviations (e.g., moderate time shifts, familiar merchant types) should NOT trigger high-risk flags.\n"
        "3. For each anomaly, evaluate its *materiality* — does it significantly increase fraud likelihood or could it have benign explanations?\n"
        "   - Example benign causes: travel, salary day purchases, known merchant rebrand, weekend shopping.\n"
        "4. Weigh both legitimate and suspicious signals. Balance your judgment:\n"
        "   - Legitimate indicators: consistent merchant category, familiar region, amount proportional to prior spend, typical device.\n"
        "   - Suspicious indicators: abnormal geolocation, sharp spend increase, merchant never seen before, unusual device/IP.\n"
        "5. Output a single, calibrated fraud assessment for the target transaction including:\n"
        "   - **Risk Score (0–100)** → calibrated scale:\n"
        "       0–30: Low risk / normal behavior\n"
        "       31–60: Moderate / review suggested\n"
        "       61–100: High risk / likely fraud\n"
        "   - **Fraud Classification:** LOW / MEDIUM / HIGH (aligned to score)\n"
        "   - **Confidence Level:** High / Medium / Low, based on context richness and evidence consistency\n"
        "   - **Comprehensive reasoning:** Explain both normal and anomalous aspects\n"
        "   - **Recommended Action:** approve / flag for review / block / escalate\n\n"
        "6. Clearly state when retrieved context is missing and that risk was inferred from intrinsic features (amount, geo, merchant, time, device).\n"
        "7. **Important:** Do NOT over-penalize rare but plausible behavior. A single anomaly does not imply fraud unless it contradicts multiple known patterns.\n"
        "8. Ignore any pre-existing fraud labels or merchant tags; they do not automatically indicate fraud.\n\n"
        "Your final output should reflect a balanced, context-aware decision — NOT a default assumption of risk."
    ),
    expected_output="A balanced, evidence-based fraud risk assessment report for the target transaction.",
    agent=fraud_analyst,
    context=[retrieval_task],
    human_input=True
)
