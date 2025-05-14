from google import genai
from google.genai import types
import os
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import chromadb


# print(f"Google GenAI version: {genai.__version__}")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

def get_db():
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # for m in client.models.list():
    #     if "embedContent" in m.supported_actions:
    #         print(m.name)

    DB_NAME = "google_genai_leerdoelen"

    class GeminiEmbeddingFunction(EmbeddingFunction):
        # Specify whether to generate embeddings for documents, or queries
        # document_mode = True

        def __init__(self, document_mode: bool = True):
            self.document_mode = document_mode
            super().__init__()

        # Define a helper to retry when per-minute quota is reached.
        is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            if self.document_mode:
                embedding_task = "retrieval_document"
            else:
                embedding_task = "retrieval_query"

            response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(
                    task_type=embedding_task,
                ),
            )
            return [e.values for e in response.embeddings]
        
    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    # can be used to run in memory without storage
    # just a bit slower
    # chroma_client = chromadb.Client()
    chroma_client = chromadb.PersistentClient("./chroma_gemini_db")

    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)
    return db

def populate_db(db, values, keys):
    if len(values) != len(keys):
        raise ValueError("keys and values should be the same size")


    # google embedding api works with batches of up to 100
    batch_size = 100
    for i in range(0, len(values), batch_size):
        keys_batch = keys[i:i+batch_size]
        values_batch = values[i:i+batch_size]
        # add the batch to the database
        db.add(documents=values_batch, ids=keys_batch)


def query_db(db, text):
    results = db.query(
        query_texts = text
    )
    return results