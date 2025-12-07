import os
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain_community.document_loaders import CSVLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import CSV_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from typing import Any


vector_store = None  # Global fallback

# --- Build FAISS Index (with fallback) ---
if os.path.exists(CSV_PATH):
    try:
        loader = CSVLoader(file_path=CSV_PATH)
        docs = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"[WARN] Failed to load vector store: {e}")
else:
    print(f"[WARN] No transaction history CSV found at {CSV_PATH}. Retrieval will be skipped.")
    vector_store = None

# --- Schema for the tool arguments ---
class VectorSearchToolSchema(BaseModel):
    query: Any = Field(..., description="Search query. Will be coerced to string by the tool.")

# --- Tool Definition ---
class VectorSearchTool(BaseTool):
    name: str = "TransactionVectorSearch"
    description: str = "Search and retrieve 5 most semantically similar historical transactions."
    args_schema = VectorSearchToolSchema

    def _run(self, query) -> str:
        # Handle CrewAI's nested dicts
        if isinstance(query, dict):
            query = query.get("description") or str(query)
        elif not isinstance(query, str):
            query = str(query)

        if vector_store:
            results = vector_store.similarity_search(query, k=5)
            return "\n".join([r.page_content for r in results])
        else:
            # Fallback behavior if no data is available
            return "[INFO] No historical transactions available to search."

    async def _arun(self, query) -> str:
        return self._run(query)
