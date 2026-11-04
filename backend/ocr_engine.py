import asyncio
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("ocr_engine")

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Two-stage OCR:
#   "fast"   - PP-OCRv6 small screens every candidate (~0.3s for a front pack)
#   "strict" - PP-OCRv6 medium re-reads only the image about to be accepted (more accurate on
#              small / stylised text), so a wrong image must fool both models to get through.
#
# Each worker thread owns its own engine. ONNX Runtime grabs EVERY core per session by default and
# the engines then fight over the CPU (measured 3.4s/image); capping threads per engine and running
# several engines side by side gives ~0.26s/image throughput on the same machine.
OCR_THREADS_PER_ENGINE = max(1, int(os.environ.get("OCR_THREADS_PER_ENGINE", "4")))
_cores = os.cpu_count() or 4
OCR_WORKERS = max(1, int(os.environ.get("OCR_WORKERS", str(min(12, max(1, _cores // OCR_THREADS_PER_ENGINE))))))
STRICT_OCR_WORKERS = max(1, int(os.environ.get("STRICT_OCR_WORKERS", str(min(6, max(1, _cores // (2 * OCR_THREADS_PER_ENGINE)))))))
MIN_CONFIDENCE = 0.5

# (text detection model, text recognition model) per stage. Reading accuracy comes from the recognition
# model; detection only finds the text boxes, so the strict stage pairs small detection with medium
# recognition (~2x faster than medium+medium, same strict reading).
def _models(det_env: str, rec_env: str, det_default: str, rec_default: str):
    det = os.environ.get(det_env, det_default).strip().lower()
    rec = os.environ.get(rec_env, rec_default).strip().lower()
    ok = ("tiny", "small", "medium")
    return (det if det in ok else det_default, rec if rec in ok else rec_default)

MODELS = {
    "fast": _models("OCR_DET_MODEL", "OCR_MODEL", "small", "small"),
    "strict": _models("STRICT_OCR_DET_MODEL", "STRICT_OCR_MODEL", "small", "medium"),
}

_executors = {
    "fast": ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr-fast"),
    "strict": ThreadPoolExecutor(max_workers=STRICT_OCR_WORKERS, thread_name_prefix="ocr-strict"),
}
_local = threading.local()
_create_lock = threading.Lock()
backend_name: Optional[str] = None


def _create_engine(models: tuple):
    """PP-OCRv6 on ONNX Runtime, falling back to RapidOCR."""
    global backend_name
    det, rec = models
    with _create_lock:
        try:
            from paddleocr import PaddleOCR
            engine = PaddleOCR(
                text_detection_model_name=f"PP-OCRv6_{det}_det",
                text_recognition_model_name=f"PP-OCRv6_{rec}_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="onnxruntime",
                engine_config={"intra_op_num_threads": OCR_THREADS_PER_ENGINE, "inter_op_num_threads": 1},
                text_det_limit_side_len=960,
                text_det_limit_type="max",
            )
            backend_name = f"PP-OCRv6 {MODELS['fast']} + {MODELS['strict']}"
            return "ppocr", engine
        except Exception as e:
            logger.warning(f"PP-OCRv6 {det}/{rec} unavailable, falling back to RapidOCR: {e}")
        try:
            from rapidocr_onnxruntime import RapidOCR
            backend_name = "RapidOCR"
            return "rapid", RapidOCR()
        except Exception as e:
            logger.warning(f"RapidOCR unavailable, OCR disabled: {e}")
        return None, None


def _get_engine(stage: str):
    engines = getattr(_local, "engines", None)
    if engines is None:
        engines = _local.engines = {}
    if stage not in engines:
        engines[stage] = _create_engine(MODELS[stage])
    return engines[stage]


def _load(content: bytes) -> Optional[Image.Image]:
    img = Image.open(io.BytesIO(content))
    if img.mode in ("RGBA", "LA", "P"):
        # Transparent PNG packshots: put them on white instead of black
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    if min(img.size) < 40:
        return None
    return img


def _prepare(img: Image.Image) -> np.ndarray:
    w, h = img.size
    longest = max(w, h)
    if longest > 1280:
        img = img.copy()
        img.thumbnail((1280, 1280))
    elif longest < 600:
        # Small search thumbnails: upscale so small net-weight text is readable
        scale = 900 / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # OCR engines expect BGR (OpenCV channel order)
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def packshot_quality(img: Image.Image) -> float:
    """
    0..1 score of how much the image looks like a clean e-commerce packshot.
    A packshot sits on a plain background, so all four borders share one colour.
    Lifestyle photos (pack on a kitchen table) and cropped / half-cut packs fail on at least one side.
    Returns the worst side's share of background-coloured border pixels.
    """
    a = np.asarray(img.resize((96, 96))).astype(np.int32)
    band = 4
    sides = [a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3), a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)]
    ref = np.median(np.concatenate(sides), axis=0)
    shares = [float(np.mean(np.abs(s - ref).sum(axis=1) < 60)) for s in sides]
    return round(min(shares), 3)


def repeated_pack_score(img: Image.Image) -> float:
    """
    How much the image looks like the SAME pack repeated side by side (2-pack / 3-pack photos).
    The image is compared with itself shifted 28-72% of its width; only pixels where both copies show
    product (not background) count. Measured: identical 2-pack 0.07, single packs 0.39-0.80.
    Lower = more repeated.
    """
    g = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32)
    border = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
    bg = float(np.median(border))
    contrast = float(np.abs(g - bg).mean()) + 1e-6
    best = 9.0
    for k in range(int(128 * 0.28), int(128 * 0.72)):
        a, b = g[:, :128 - k], g[:, k:]
        fg = (np.abs(a - bg) > 35) & (np.abs(b - bg) > 35)
        if fg.mean() < 0.12:
            continue
        best = min(best, float(np.abs(a - b)[fg].mean()) / contrast)
    return round(best, 3)


def _ocr_sync(content: bytes, stage: str) -> Dict[str, Any]:
    kind, engine = _get_engine(stage)
    try:
        img = _load(content)
        if img is None:
            return {"words": [], "quality": 0.0, "repeat": 9.0}
        quality = packshot_quality(img)
        repeat = repeated_pack_score(img)
        if engine is None:
            return {"words": [], "quality": quality, "repeat": repeat}
        arr = _prepare(img)
        if kind == "ppocr":
            out = engine.predict(arr)[0]
            pairs = zip(out["rec_texts"], out["rec_scores"])
        else:
            res, _ = engine(arr)
            pairs = ((text, conf) for _, text, conf in (res or []))
        words = [str(t).lower().strip() for t, s in pairs if s >= MIN_CONFIDENCE and str(t).strip()]
        return {"words": words, "quality": quality, "repeat": repeat}
    except Exception as e:
        logger.debug(f"OCR failed: {e}")
        return {"words": [], "quality": 0.0, "repeat": 9.0}


async def run_ocr_full(content: bytes, stage: str = "fast") -> Dict[str, Any]:
    """Reads packaging text. stage='fast' (screening) or 'strict' (final verification)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executors[stage], _ocr_sync, content, stage)


async def run_ocr(content: bytes) -> List[str]:
    """Fast-stage OCR text lines (lowercase)."""
    return (await run_ocr_full(content, "fast"))["words"]


def _warm_one(stage: str, barrier: threading.Barrier):
    _get_engine(stage)
    try:
        # Hold every pool thread until all have started, so each thread loads its own engine
        barrier.wait(timeout=240)
    except threading.BrokenBarrierError:
        pass


def warmup():
    """Loads every engine of both stages in the background so the first rows are not slow."""
    for stage, n in (("fast", OCR_WORKERS), ("strict", STRICT_OCR_WORKERS)):
        logger.info(f"Warming up {n} {stage} OCR engines ({MODELS[stage]}) x {OCR_THREADS_PER_ENGINE} threads")
        barrier = threading.Barrier(n)
        for _ in range(n):
            _executors[stage].submit(_warm_one, stage, barrier)
