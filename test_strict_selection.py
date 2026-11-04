import asyncio
import io
import re
import sys
import httpx
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR

sys.stdout.reconfigure(encoding='utf-8')
ocr_engine = RapidOCR()

url = 'http://localhost:8888/search'
query = '24 Mantra Organic 7 Grain Multigrain Atta'

resp = httpx.get(url, params={
    'q': query,
    'category_images': 1,
    'format': 'json',
    'engines': 'google images,bing images,duckduckgo images'
})
results = resp.json().get('results', [])[:20]

def has_any_size_ocr_or_title(words: list[str], title: str) -> bool:
    full_text = " ".join(words) + " " + title
    # Detect patterns like 1kg, 500g, 2.2lb, 5kg, 10kg, 1 kg, 500 gm, 500g, 100g, etc.
    if re.search(r'\b\d+(?:\.\d+)?\s*(kg|g|gm|ml|l|ltr|lb|lbs)\b', full_text, re.I):
        return True
    clean = full_text.replace(" ", "").lower()
    if re.search(r'\d+(?:kg|gm|ml|ltr|lb)', clean):
        return True
    return False

def has_exact_size_ocr_or_title(words: list[str], title: str, target_size: str) -> bool:
    m = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gm|ml|l|ltr|lb)?', target_size.lower())
    if not m:
        return False
    num = m.group(1)
    unit = m.group(2) or ""
    full_text = " ".join(words) + " " + title
    clean = full_text.replace(" ", "").lower()
    targets = [f"{num}{unit}", f"{num}gm", f"{num}g", f"{num}kg", f"{num}ml", f"{num}l", f"{num}lb"]
    return any(t in clean for t in targets)

def has_brand(words: list[str], title: str, brand: str) -> bool:
    full_text = (" ".join(words) + " " + title).lower()
    tokens = [t.lower() for t in re.split(r'[\s\-_]+', brand) if len(t) >= 2]
    return any(bt in full_text for bt in tokens)

print(f"Testing 20 Candidates for Product (Without Size) and Variant (With Size 1kg):")

best_product_idx = -1
best_variant_idx = -1

for idx, r in enumerate(results):
    img_url = r.get('img_src') or r.get('thumbnail_src') or ''
    title = r.get('title') or ''
    
    # Fast OCR
    try:
        im_resp = httpx.get(img_url, timeout=3.0)
        im = Image.open(io.BytesIO(im_resp.content)).convert('RGB')
        im.thumbnail((640, 640))
        res, _ = ocr_engine(np.array(im))
        words = [w.lower() for _, w, _ in res] if res else []
    except Exception:
        words = []

    brand_ok = has_brand(words, title, "24 Mantra Organic")
    has_size = has_any_size_ocr_or_title(words, title)
    has_1kg = has_exact_size_ocr_or_title(words, title, "1kg")

    print(f"[{idx:02d}] Brand: {brand_ok} | AnySize: {has_size} | Has1kg: {has_1kg} | Title: {title[:40]}")

    # Check for Product (Without Size): Brand must match AND must NOT have size
    if brand_ok and not has_size and best_product_idx == -1:
        best_product_idx = idx

    # Check for Variant (With Size 1kg): Brand must match AND must have exact 1kg
    if brand_ok and has_1kg and best_variant_idx == -1:
        best_variant_idx = idx

print("\n================ FINAL SELECTION ================")
print(f"BEST PRODUCT (WITHOUT SIZE) -> Candidate [{best_product_idx}]: {results[best_product_idx]['title']}")
print(f"    URL: {results[best_product_idx].get('img_src')}")
print(f"\nBEST VARIANT (WITH SIZE 1kg) -> Candidate [{best_variant_idx}]: {results[best_variant_idx]['title']}")
print(f"    URL: {results[best_variant_idx].get('img_src')}")
