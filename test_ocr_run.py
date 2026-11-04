import httpx
import io
import time
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

print("Initializing PaddleOCR...")
t0 = time.time()
# use lightweight English OCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')
print(f"PaddleOCR initialized in {time.time() - t0:.2f}s")

# Let's test on a real product image URL from Tata Sampann
img_url = "https://m.media-amazon.com/images/I/41stinW0a3L._SX800_.jpg"

print(f"Downloading test image from {img_url}...")
resp = httpx.get(img_url, timeout=10.0)
img = Image.open(io.BytesIO(resp.content)).convert("RGB")
img_np = np.array(img)

print(f"Image size: {img.size}")
t1 = time.time()
result = ocr.ocr(img_np)
print(f"OCR Inference completed in {time.time() - t1:.2f}s")

detected_text = []
for line in result:
    for word_info in line:
        text, conf = word_info[1]
        detected_text.append(f"{text} ({conf:.2f})")

print("\n--- DETECTED TEXT ON PACKET ---")
for dt in detected_text:
    print(dt)
