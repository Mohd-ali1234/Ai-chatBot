from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
from google import genai
from dotenv import load_dotenv
import os
import chromadb
import numpy as np
import uvicorn

load_dotenv()

app = FastAPI()

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Gemini API client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# ------------------------------
# GEMINI EMBEDDING FUNCTION
# ------------------------------
import numpy as np

def get_embedding(text):
    emb_response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=[text]
    )
    return np.array(emb_response.embeddings[0].values)



# ------------------------------
# CHROMADB COLLECTION
# ------------------------------
chroma_client = chromadb.PersistentClient(path="/data/chroma")

collection = chroma_client.get_or_create_collection(
    name="pdf_collection_gemini",
    metadata={"hnsw:space": "cosine", "embedding_dim": 768}
)

PDF_PATH = "test.pdf"


# -------------------------------------------------------
# STEP 1 → Store PDF embeddings only if not already there
# -------------------------------------------------------
def store_pdf_embeddings():
    if collection.count() > 0:
        print("📌 Embeddings already exist. Skipping...")
        return

    print("⏳ Reading PDF & creating embeddings...")
    reader = PdfReader(PDF_PATH)
    pdf_text = "\n".join(
        filter(None, (page.extract_text() for page in reader.pages))
    )

    sentences = [s.strip() for s in pdf_text.splitlines() if s.strip()]

    embeddings = [get_embedding(s) for s in sentences]

    ids = [f"id_{i}" for i in range(len(sentences))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=sentences
    )

    print("✅ PDF stored successfully in ChromaDB!")


# Run once at startup
store_pdf_embeddings()


# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# -------------------------------------------------------
# STEP 2 → Query ChromaDB Instead of Recomputing Embeddings
# -------------------------------------------------------
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(query: str = Form(...)):
    
    query_emb = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    retrieved_chunks = "\n\n".join(results["documents"][0])
    print("📌 Retrieved:", retrieved_chunks)

    # Prepare Gemini prompt
    prompt = f"""
You are an Islamic scholar. Answer the question strictly according to the information given below.
Do NOT add your own opinions or outside info.

--- Retrieved Information ---
{retrieved_chunks}
-----------------------------

Question: {query}

Important:
- Write the final answer ONLY in Gujarati.
- Do NOT translate or modify the retrieved text.

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = response.text

    return f"""
    <h2>Question:</h2>
    <p>{query}</p>

    <h2>Answer (Gujarati):</h2>
    <p>{answer}</p>

    <a href="/">Ask another question</a>
    """


# -------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
