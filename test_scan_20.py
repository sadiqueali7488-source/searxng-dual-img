import httpx
import io
import re
import sys
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
results = resp.json().get('results', [])[:15]

def extract_ocr_words(img_url: str):
    try:
        r = httpx.get(img_url, timeout=4.0)
        if r.status_code != 200 or len(r.content) < 800:
            return []
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((640, 640))
        res, _ = ocr_engine(np.array(im))
        if not res:
            return []
        return [text.lower().strip() for _, text, conf in res if conf >= 0.4]
    except Exception as e:
        return []

def has_any_size_callout(words: list[str]) -> bool:
    """Returns True if the image text has any weight/size mention (e.g. 1kg, 500g, 5kg)."""
    text = " ".join(words).replace(" ", "")
    # Check for sizes like 1kg, 500g, 2kg, 5kg, 10kg, 200g, 250g, 100g, etc.
    if re.search(r'\b\d+\s*(kg|g|gm|ml|l|ltr|lb)\b', " ".join(words)):
        return True
    if re.search(r'\d+(kg|gm|ml|ltr|lb)', text):
        return True
    return False

def has_exact_size(words: list[str], target_size: str) -> bool:
    """Returns True if the image text has the EXACT requested size."""
    m = re.search(r'(\d+)\s*(kg|g|gm|ml|l|ltr)?', target_size.lower())
    if not m:
        return False
    num = m.group(1)
    unit = m.group(2) or ""
    text = " ".join(words).replace(" ", "")
    targets = [f"{num}{unit}", f"{num}gm", f"{num}g", f"{num}kg", f"{num}ml", f"{num}l"]
    return any(t in text for t in targets)

print(f"Inspecting {len(results)} images with Vision/OCR...")
for i, r in enumerate(results):
    img_url = r.get('img_src') or r.get('thumbnail_src') or ''
    title = r.get('title') or ''
    words = extract_ocr_words(img_url)
    any_size = has_any_size_callout(words)
    has_1kg = has_exact_size(words, "1kg")
    has_brand = any("mantra" in w or "24" in w for w in words)
    print(f"\n[{i:02d}] Title: {title[:50]}")
    print(f"     Words: {words[:8]}")
    print(f"     Has Brand: {has_brand} | Any Size: {any_size} | Has 1kg: {has_1kg}")
