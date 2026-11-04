import httpx
import io
import re
import time
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ocr_engine = RapidOCR()

def extract_image_text(img_url: str, timeout: float = 4.0) -> list[str]:
    try:
        resp = httpx.get(img_url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return []
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Resize to max 640px for super fast OCR
        img.thumbnail((640, 640))
        img_np = np.array(img)
        res, _ = ocr_engine(img_np)
        if not res:
            return []
        return [text.lower().strip() for _, text, conf in res if conf >= 0.5]
    except Exception as e:
        return []

def verify_variant_size(detected_words: list[str], variant_label: str) -> bool:
    if not variant_label or not detected_words:
        return False
    combined = " ".join(detected_words).replace(" ", "")
    
    # Parse number and unit: e.g., 500g, 1kg, 250ml
    m = re.search(r'(\d+)\s*(kg|g|gm|ml|l|ltr)?', variant_label.lower())
    if not m:
        return False
    num = m.group(1)
    unit = m.group(2) or ""

    targets = [
        f"{num}{unit}",
        f"{num}gm",
        f"{num}g",
        f"{num}kg",
        f"{num}ml",
        f"{num}l"
    ]
    for w in detected_words:
        w_clean = w.replace(" ", "")
        if any(t in w_clean for t in targets):
            return True
    return any(t in combined for t in targets)

def verify_brand(detected_words: list[str], brand: str) -> bool:
    if not brand or not detected_words:
        return False
    combined = " ".join(detected_words)
    brand_tokens = [t.lower() for t in re.split(r'[\s\-_]+', brand) if len(t) >= 2]
    return any(bt in combined for bt in brand_tokens)

# Test on 3 images of varying brands/sizes
test_urls = [
    ("https://5.imimg.com/data5/SELLER/Default/2023/9/348529450/VN/EB/RG/153971840/500g-chana-sattu-1000x1000.jpg", "Narayani Sattu 500g"),
    ("https://m.media-amazon.com/images/I/41stinW0a3L._SX800_.jpg", "Tata Sampann 500g"),
    ("https://m.media-amazon.com/images/I/7103jX7C01L.jpg", "Tata Sampann 1kg")
]

print("Testing PaddleOCR Verification...")
for url, label in test_urls:
    t0 = time.time()
    words = extract_image_text(url)
    elapsed = time.time() - t0
    print(f"\nImage: {label} (Time: {elapsed:.2f}s)")
    print(f"  Detected words: {words}")
    has_tata = verify_brand(words, "Tata Sampann")
    has_500g = verify_variant_size(words, "500g")
    has_1kg = verify_variant_size(words, "1kg")
    print(f"  Brand 'Tata Sampann': {has_tata}")
    print(f"  Size '500g':          {has_500g}")
    print(f"  Size '1kg':           {has_1kg}")

