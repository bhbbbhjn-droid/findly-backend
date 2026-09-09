from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Findly API is running"}

@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    # يمكن دمج API سحابي مجاني هنا لاحقاً
    return {"message": "Image received successfully", "filename": file.filename}

@app.get("/search-text")
def search_text(q: str):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=5):
            results.append(r)
    return {"results": results}

