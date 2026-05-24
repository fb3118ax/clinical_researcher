import pdfplumber
import os
from langchain_core.documents import Document
from config import DATA_FOLDER

def loader_doc():
    document = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            file_name = file.replace(".pdf", "")
            parts = file_name.split("_", 1)
            study_id = parts[0]
            author_year = parts[1]
            print(f"file: {file}, study_id: {study_id}, author_year: {author_year}")
            with pdfplumber.open(f"{DATA_FOLDER}/{file}") as pdf:
                for page in pdf.pages:
                    if not page:
                        continue
                    else:
                        text = page.extract_text()
                        if text and sum(len(w) for w in text.split()) / len(text.split()) >= 5.0:
                            document.append(Document(
                                page_content=text,
                                metadata={
                                    "study_id": study_id,
                                    "author_year": author_year,
                                    "source": file_name,
                                    "page_number": page.page_number
                                }
                            ))
    return document
