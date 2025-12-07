# tools/vector_search_tool.py
'''
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain_text_splitters import CharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import CSV_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from preprocessing.schema_mapper import map_sparkov_csv_to_transactions
#from langchain.docstore.document import Document


# --- Load and Preprocess Sparkov CSV ---
try:
    transactions = map_sparkov_csv_to_transactions(CSV_PATH)
    if not transactions:
        print(f"⚠️ No transactions found in {CSV_PATH}. Running without historical data...")
        vector_store = None
    else:
        docs = [
            Document(
                page_content=(
                    f"Transaction ID: {t['transaction_id']}, "
                    f"Amount: {t['amount']}, "
                    f"Category: {t['category']}, "
                    f"Location: {t['location']}, "
                    f"Merchant: {t['merchant']}, "
                    f"Timestamp: {t['timestamp']}, "
                    f"User ID: {t['user_id']}, "
                    f"Merchant Lat: {t['merch_lat']}, "
                    f"Merchant Long: {t['merch_long']}, "
                    f"Time: {t['trans_time']}, "
                )
            )
            for t in transactions
        ]
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        if not chunks:
            print("⚠️ No document chunks to index. Running without historical data...")
            vector_store = None
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
except FileNotFoundError:
    print(f"⚠️ No dataset found at {CSV_PATH}. Running without historical data...")
    vector_store = None
except Exception as e:
    print(f"⚠️ Error loading transactions: {e}. Running without historical data...")
    vector_store = None


class VectorSearchToolSchema(BaseModel):
    query: str = Field(..., description="Search query as a plain string.")

class VectorSearchTool(BaseTool):
    name: str = "TransactionVectorSearch"
    description: str = "Search and retrieve 5 most semantically similar historical transactions."
    args_schema = VectorSearchToolSchema

    def _run(self, query) -> str:
        if vector_store is None:
            return "⚠️ No historical transactions available (empty dataset)."
        
        results = vector_store.similarity_search(query, k=5)
        return "\n".join([r.page_content for r in results])

    async def _arun(self, query) -> str:
        return self._run(query)

'''

import os
import re
from pydantic import BaseModel, Field
from typing import Any, Type
try:
    from crewai.tools import BaseTool
except Exception:
    class BaseTool:
        name: str = "BaseTool"
        description: str = ""
        args_schema = None
        def _run(self, *a, **kw): raise NotImplementedError
        async def _arun(self, *a, **kw): return self._run(*a, **kw)

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import CSV_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from preprocessing.schema_mapper import map_sparkov_csv_to_transactions
from langchain_core.documents import Document

# --- Load and Preprocess Sparkov CSV ---
vector_store = None
try:
    transactions = map_sparkov_csv_to_transactions(CSV_PATH)
    if not transactions:
        print(f"⚠️ No transactions found in {CSV_PATH}. Running without historical data...")
        vector_store = None
    else:
        docs = [
            Document(
                page_content=(
                    f"Transaction ID: {t['transaction_id']}, "
                    f"Amount: {t['amount']}, "
                    f"Category: {t['category']}, "
                    f"Location: {t['location']}, "
                    f"Merchant: {t['merchant']}, "
                    f"Timestamp: {t['timestamp']}, "
                    f"User ID: {t['user_id']}, "
                    f"Merchant Lat: {t['merch_lat']}, "
                    f"Merchant Long: {t['merch_long']}, "
                    f"Time: {t['trans_time']}, "
                )
            )
            for t in transactions
        ]
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        if not chunks:
            print("⚠️ No document chunks to index. Running without historical data...")
            vector_store = None
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
except FileNotFoundError:
    print(f"⚠️ No dataset found at {CSV_PATH}. Running without historical data...")
    vector_store = None
except Exception as e:
    print(f"⚠️ Error loading transactions: {e}. Running without historical data...")
    vector_store = None

class VectorSearchToolSchema(BaseModel):
    query: str = Field(..., description="Search query as a plain string.")

class VectorSearchTool(BaseTool):
    name: str = "TransactionVectorSearch"
    description: str = "Search and retrieve 5 most semantically similar historical transactions."
    args_schema: Type[BaseModel] | Any = VectorSearchToolSchema

    def _run(self, query) -> str:
        if vector_store is None:
            return "⚠️ No historical transactions available (empty dataset)."
        results = vector_store.similarity_search(query, k=5)
        return "\n".join([r.page_content for r in results])

    async def _arun(self, query) -> str:
        return self._run(query)