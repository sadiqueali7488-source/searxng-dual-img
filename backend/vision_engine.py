import asyncio
import base64
import io
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("vision_engine")

# OpenCLIP (MIT license) "looks" at each candidate image, the way a person would:
#   - what kind of photo is it (single front pack / back panel / several packs / lifestyle / banner / crop)
#   - how well does it match the product name
#   - an image embedding, used to find the same official photo on several stores (consensus) and to
#     check that the product image and the variant image show the same pack
# Runs on CPU (~0.1s per image). Set VISION_ENABLED=0 to switch it off.
VISION_ENABLED = os.environ.get("VISION_ENABLED", "1") not in ("0", "false", "no")
VISION_MODEL = os.environ.get("VISION_MODEL", "ViT-B-32")
VISION_PRETRAINED = os.environ.get("VISION_PRETRAINED", "laion2b_s34b_b79k")
VISION_THREADS = max(1, int(os.environ.get("VISION_THREADS", "8")))
VISION_WORKERS = max(1, int(os.environ.get("VISION_WORKERS", "2")))

# Photo kinds. The first one is what we want; the rest are reasons to reject.
SCENE_PROMPTS = {
    "front_single": [
        "a product photo of one single package, front view, on a plain white background",
        "the front of a single retail product pack",
    ],
    "back_panel": [
        "the back of a food package with nutrition facts, ingredients and barcode",
        "a printed label with a table of nutrition information",
    ],
    "multiple": [
        "several identical product packages placed side by side",
        "a group of many different product packages",
    ],
    "lifestyle": [
        "a lifestyle photo of food on a kitchen table",
        "a cooked dish served in a bowl",
    ],
    "banner": [
        "an advertisement banner with large promotional text",
        "a screenshot of a shopping website",
    ],
    "crop": [
        "an extreme close-up crop showing only part of a package",
    ],
    # Shop-counter display strips / hanger cards with many small sachets (e.g. a lollipop strip):
    # CLIP calls them "one front pack" otherwise. Rejected only when very sure (real bags of candy score <= 0.56)
    "display": [
        "a cardboard display strip with many identical small packets hanging in a column",
    ],
}

_executor = ThreadPoolExecutor(max_workers=VISION_WORKERS, thread_name_prefix="vision")
_lock = threading.Lock()
_state: Dict[str, Any] = {}


def _load():
    """Loads the model once (thread-safe). Returns None when OpenCLIP isn't installed or fails."""
    if "model" in _state:
        return _state["model"]
    with _lock:
        if "model" in _state:
            return _state["model"]
        if not VISION_ENABLED:
            _state["model"] = None
            return None
        try:
            import torch
            import open_clip
            torch.set_num_threads(VISION_THREADS)
            model, _, preprocess = open_clip.create_model_and_transforms(VISION_MODEL, pretrained=VISION_PRETRAINED)
            model.eval()
            tokenizer = open_clip.get_tokenizer(VISION_MODEL)
            labels, prompts = [], []
            for label, texts in SCENE_PROMPTS.items():
                for t in texts:
                    labels.append(label)
                    prompts.append(t)
            with torch.no_grad():
                scene = model.encode_text(tokenizer(prompts))
                scene = scene / scene.norm(dim=-1, keepdim=True)
            _state.update(torch=torch, preprocess=preprocess, tokenizer=tokenizer,
                          scene=scene, scene_labels=labels, text_cache={})
            _state["model"] = model
            logger.info(f"Vision model {VISION_MODEL}/{VISION_PRETRAINED} loaded")
        except Exception as e:
            logger.warning(f"Vision model unavailable, image understanding disabled: {e}")
            _state["model"] = None
        return _state["model"]


def _text_embedding(text: str):
    cache = _state["text_cache"]
    if text not in cache:
        torch = _state["torch"]
        with torch.no_grad():
            t = _state["model"].encode_text(_state["tokenizer"]([text]))
            cache[text] = t / t.norm(dim=-1, keepdim=True)
        if len(cache) > 5000:
            cache.clear()
    return cache[text]


def _analyze_sync(content: bytes, product_text: str) -> Optional[Dict[str, Any]]:
    model = _load()
    if model is None:
        return None
    torch = _state["torch"]
    try:
        img = Image.open(io.BytesIO(content))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        with torch.no_grad():
            emb = model.encode_image(_state["preprocess"](img).unsqueeze(0))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            # Scene: softmax over all prompts, summed per kind
            logits = (100.0 * emb @ _state["scene"].T).softmax(dim=-1)[0].tolist()
            kinds: Dict[str, float] = {}
            for label, p in zip(_state["scene_labels"], logits):
                kinds[label] = kinds.get(label, 0.0) + p
            match = None
            if product_text:
                match = float((emb @ _text_embedding(f"a photo of {product_text} product package").T)[0, 0])
        return {
            "kinds": {k: round(v, 4) for k, v in kinds.items()},
            "kind": max(kinds, key=kinds.get),
            "match": round(match, 4) if match is not None else None,
            # float16 + base64: ~1.4 KB per image instead of ~10 KB of JSON floats (1 lakh rows stay small)
            "embedding": base64.b64encode(emb[0].numpy().astype(np.float16).tobytes()).decode("ascii"),
        }
    except Exception as e:
        logger.debug(f"Vision analysis failed: {e}")
        return None


async def analyze(content: bytes, product_text: str = "") -> Optional[Dict[str, Any]]:
    """Image understanding for one candidate, or None when the vision model is unavailable."""
    if not VISION_ENABLED:
        return None
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _analyze_sync, content, product_text)


def _decode(e) -> Optional[np.ndarray]:
    if not e:
        return None
    try:
        return np.frombuffer(base64.b64decode(e), dtype=np.float16).astype(np.float32)
    except Exception:
        return None


def similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Cosine similarity of two image embeddings (1.0 = same photo, ~0.8+ = same pack design)."""
    va, vb = _decode(a), _decode(b)
    if va is None or vb is None or va.shape != vb.shape:
        return None
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def warmup():
    if VISION_ENABLED:
        _executor.submit(_load)
