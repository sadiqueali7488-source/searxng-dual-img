import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import paddle
paddle.framework.set_flags({'FLAGS_enable_pir_api': False, 'FLAGS_use_onednn': False})

import httpx
import io
import time
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=False, lang='en')

# Test image: Tata Sampann
img_url = "https://m.media-amazon.com/images/I/41stinW0a3L._SX800_.jpg"
resp = httpx.get(img_url, timeout=10.0)
img = Image.open(io.BytesIO(resp.content)).convert("RGB")
img_np = np.array(img)

print("Running OCR predict...")
t0 = time.time()
res = ocr.predict(img_np)
print(f"Prediction done in {time.time() - t0:.2f}s")
for r in res:
    print("RES:", r)
