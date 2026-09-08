from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import re

app = FastAPI(title="Visual Product Search API")

# إعدادات CORS للربط مع Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل نموذج BLIP الخفيف للتعرف البصري بدلاً من CLIP الثقيل
MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

def extract_price(text: str):
    """دالة خفيفة لاستخراج الأرقام والأسعار من النصوص"""
    match = re.search(r'(\$|€|£|USD|SAR|EGP)?\s?(\d+[\.,]?\d*)', text)
    if match:
        try:
            return float(match.group(2).replace(',', ''))
        except ValueError:
            return None
    return None

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف المرفوع ليس صورة")

    try:
        # 1. قراءة الصورة وتحليلها بصرياً
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # 2. البحث الحي عبر الإنترنت بالكلمات المفتاحية المستخرجة
        search_query = f"{caption} buy price store"
        results = []
        
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(search_query, max_results=15))
            
            for item in ddg_results:
                title = item.get("title", "")
                snippet = item.get("body", "")
                url = item.get("href", "")
                
                price = extract_price(title) or extract_price(snippet)
                
                if price:
                    results.append({
                        "title": title,
                        "price": price,
                        "currency": "USD",
                        "product_url": url,
                        "store_name": url.split("/")[2].replace("www.", ""),
                        "image_url": "https://via.placeholder.com/150"  # صورة افتراضية عند عدم توفر الرابط المباشر
                    })

        # 3. ترتيب النتائج تلقائياً من الأرخص للأغلى
        sorted_results = sorted(results, key=lambda x: x["price"])
        
        return {
            "query_detected": caption,
            "total_products": len(sorted_results),
            "products": sorted_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
