from crewai import Task
from agents.retriever_agent import retriever_agent
from transaction.sample import sample_transaction


retrieval_task = Task(
    description=(
        f"Given this transaction:\n{sample_transaction}\n\n"
        "Your task is to craft 2–4 diverse semantic search queries (covering spending, location, merchant, and velocity anomalies) "
        "to retrieve the most relevant historical transactions. Use the TransactionVectorSearch tool for each query, "
        "aggregate and de-duplicate results, and annotate why each was retrieved."
        "You must not hallucinate or make assumptions."
        "You MUST call TransactionVectorSearch ONLY with a plain string for the query. "
        "Never wrap it in JSON, dictionaries, or keys like 'description' or 'type'."
    ),
    expected_output="A structured, annotated list of relevant past transactions for the Fraud Analyst.",
    agent=retriever_agent,
    human_input=False
)