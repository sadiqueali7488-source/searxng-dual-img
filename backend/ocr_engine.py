import asyncio
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("ocr_engine")

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Each worker thread owns its own OCR engine instance (engines are not shared across threads).
OCR_WORKERS = max(1, int(os.environ.get("OCR_WORKERS", "4")))
MIN_CONFIDENCE = 0.5

_executor = ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr")
_local = threading.local()
_create_lock = threading.Lock()
backend_name: Optional[str] = None


def _create_engine():
    """PP-OCRv6 Medium on ONNX Runtime (~0.5s/image on CPU), falling back to RapidOCR."""
    global backend_name
    with _create_lock:
        try:
            from paddleocr import PaddleOCR
            engine = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="onnxruntime",
                text_det_limit_side_len=960,
                text_det_limit_type="max",
            )
            backend_name = "PP-OCRv6_medium"
            return "ppocr", engine
        except Exception as e:
            logger.warning(f"PP-OCRv6 unavailable, falling back to RapidOCR: {e}")
        try:
            from rapidocr_onnxruntime import RapidOCR
            backend_name = "RapidOCR"
            return "rapid", RapidOCR()
        except Exception as e:
            logger.warning(f"RapidOCR unavailable, OCR disabled: {e}")
        return None, None


def _get_engine():
    if not hasattr(_local, "engine"):
        _local.kind, _local.engine = _create_engine()
    return _local.kind, _local.engine


def _prepare(content: bytes) -> Optional[np.ndarray]:
    img = Image.open(io.BytesIO(content))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGBA")
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < 40:
        return None
    longest = max(w, h)
    if longest > 1280:
        img.thumbnail((1280, 1280))
    elif longest < 600:
        # Small search thumbnails: upscale so small net-weight text is readable
        scale = 900 / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # OCR engines expect BGR (OpenCV channel order)
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def _ocr_sync(content: bytes) -> List[str]:
    kind, engine = _get_engine()
    if engine is None:
        return []
    try:
        img = _prepare(content)
        if img is None:
            return []
        if kind == "ppocr":
            out = engine.predict(img)[0]
            pairs = zip(out["rec_texts"], out["rec_scores"])
        else:
            res, _ = engine(img)
            pairs = ((text, conf) for _, text, conf in (res or []))
        return [str(t).lower().strip() for t, s in pairs if s >= MIN_CONFIDENCE and str(t).strip()]
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return []


async def run_ocr(content: bytes) -> List[str]:
    """Reads packaging text from image bytes. Returns lowercase text lines."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _ocr_sync, content)


def warmup():
    """Loads one engine per worker thread in the background so the first rows are not slow."""
    blank = Image.new("RGB", (64, 64), "white")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    for _ in range(OCR_WORKERS):
        _executor.submit(_ocr_sync, buf.getvalue())
