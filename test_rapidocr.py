import httpx
import io
import time
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR

print("Initializing RapidOCR (PaddleOCR ONNX Engine)...")
t0 = time.time()
engine = RapidOCR()
print(f"Initialized in {time.time() - t0:.2f}s")

# Test image: Tata Sampann
img_url = "https://m.media-amazon.com/images/I/41stinW0a3L._SX800_.jpg"
resp = httpx.get(img_url, timeout=10.0)
img = Image.open(io.BytesIO(resp.content)).convert("RGB")
img_np = np.array(img)

print("Running OCR on packet image...")
t1 = time.time()
result, elapse = engine(img_np)
print(f"OCR inference completed in {time.time() - t1:.2f}s (Engine elapse: {elapse})")

print("\n--- DETECTED TEXT ON PRODUCT PACKET ---")
if result:
    for box, text, score in result:
        print(f"[{score:.2f}] {text}")
else:
    print("No text detected!")

