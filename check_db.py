# check_db.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import VECTOR_STORE_DIR, EMBEDDING_MODEL

vector_store = Chroma(
    persist_directory=VECTOR_STORE_DIR,
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
)

print(f"Total chunks: {vector_store._collection.count()}")

# Get all unique author_year values
all_data = vector_store.get()
metadatas = all_data['metadatas']

unique_authors = sorted(set(m.get('author_year', 'NO_AUTHOR') for m in metadatas))
print(f"\nAll unique author_year values ({len(unique_authors)} studies):")
for a in unique_authors:
    count = sum(1 for m in metadatas if m.get('author_year') == a)
    print(f"  {a}  ({count} chunks)")