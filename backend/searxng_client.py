import asyncio
import io
import logging
import re
import time
from typing import Dict, Any, List, Optional
import httpx
import numpy as np
from PIL import Image
from models import ImageResult

logger = logging.getLogger("searxng_client")

# Initialize PaddleOCR Engine (RapidOCR ONNX backend)
ocr_engine = None
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    logger.info("PaddleOCR engine initialized successfully.")
except Exception as e:
    logger.warning(f"PaddleOCR failed to initialize: {e}")

# In-memory OCR cache: url -> list of lowercase text words
ocr_cache: Dict[str, List[str]] = {}

def normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    return url

def sanitize_search_query(text: str) -> str:
    """Removes special characters that break or confuse image search engines."""
    s = re.sub(r'[%#*+=\\\[\]{}()<>"~`]', ' ', text or "")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

async def extract_image_ocr_text(client: httpx.AsyncClient, img_url: str, timeout: float = 3.5) -> List[str]:
    """
    Downloads image and runs PaddleOCR to read text printed on packaging.
    Results are cached in memory.
    """
    if not ocr_engine or not img_url:
        return []

    if img_url in ocr_cache:
        return ocr_cache[img_url]

    try:
        resp = await client.get(img_url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 800:
            ocr_cache[img_url] = []
            return []

        def _do_ocr(content_bytes: bytes) -> List[str]:
            img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
            # Resize image to max 640px for fast inference
            img.thumbnail((640, 640))
            img_np = np.array(img)
            res, _ = ocr_engine(img_np)
            if not res:
                return []
            return [text.lower().strip() for _, text, conf in res if conf >= 0.45]

        words = await asyncio.to_thread(_do_ocr, resp.content)
        ocr_cache[img_url] = words
        return words
    except Exception as e:
        ocr_cache[img_url] = []
        return []

def verify_variant_size(detected_words: List[str], variant_label: str) -> bool:
    """Checks if the package text actually contains the requested size (e.g., 500g, 1kg)."""
    if not variant_label or not detected_words:
        return False
    combined = " ".join(detected_words).replace(" ", "")

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

def verify_brand_ocr(detected_words: List[str], brand: str) -> Optional[bool]:
    """
    Returns True if brand found on package.
    Returns False if brand words exist and none matched.
    Returns None if uncertain or no brand given.
    """
    if not brand or not detected_words:
        return None
    combined = " ".join(detected_words)
    brand_tokens = [t.lower() for t in re.split(r'[\s\-_]+', brand) if len(t) >= 2]
    if not brand_tokens:
        return None

    matches = sum(1 for bt in brand_tokens if bt in combined)
    if matches > 0:
        return True
    return False

def score_and_rank_images(
    results: List[Dict[str, Any]],
    brand: str = "",
    product_name: str = "",
    variant_label: str = "",
    is_variant: bool = False
) -> List[Dict[str, Any]]:
    """
    Scores and ranks image results by metadata to eliminate wrong/competitor brands.
    """
    if not results:
        return []

    brand_clean = (brand or "").strip().lower()
    brand_tokens = [t for t in re.split(r'[\s\-_]+', brand_clean) if len(t) >= 2]

    stop_words = {'100', 'pure', 'pack', 'pouch', 'packet', 'the', 'and', 'for', 'with', 'original', 'best', 'organic', 'grade'}
    prod_clean = (product_name or "").strip().lower()
    prod_tokens = [t for t in re.split(r'[\s\-_]+', prod_clean) if len(t) >= 3 and t not in stop_words]

    scored = []
    has_any_brand_match = False

    for item in results:
        title = (item.get("title") or "").lower()
        img_url = (item.get("img_src") or "").lower()
        page_url = (item.get("url") or "").lower()
        combined = f"{title} {img_url} {page_url}"

        score = 0.0

        # 1. STRICT BRAND MATCHING
        brand_match_count = 0
        if brand_tokens:
            for bt in brand_tokens:
                if bt in combined:
                    brand_match_count += 1

            if brand_match_count == len(brand_tokens):
                score += 100.0
                has_any_brand_match = True
            elif brand_match_count > 0:
                score += 50.0
                has_any_brand_match = True
            else:
                score -= 80.0

        # 2. CORE PRODUCT NAME KEYWORD MATCHING
        for pt in prod_tokens:
            if pt in brand_tokens:
                continue
            if pt in combined:
                if pt in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 'masala', 'cornflour', 'poha', 'dalia']:
                    score += 45.0
                else:
                    score += 15.0

        # 3. VARIANT SIZE METADATA MATCHING
        if is_variant and variant_label:
            v_match = re.search(r'(\d+)\s*(kg|g|gm|ml|l|ltr)?', variant_label.lower())
            if v_match:
                num = v_match.group(1)
                unit = v_match.group(2) or ""
                patterns = [
                    rf'\b{num}\s*{unit}\b',
                    rf'\b{num}{unit}\b',
                    rf'\b{num}\s*gm\b',
                    rf'\b{num}gm\b',
                    rf'\b{num}\s*kg\b',
                    rf'\b{num}kg\b'
                ]
                if any(re.search(p, combined) for p in patterns):
                    score += 50.0

        # 4. TRUSTED RETAILER BOOST
        trusted = [
            'amazon', 'flipkart', 'bigbasket', 'bbassets', 'blinkit', 'grofers', 
            'zepto', 'jiomart', 'swiggy', 'tataneu', 'dailymart4u', 'f.nooncdn', 
            'exportersindia', 'indiamart', 'dmart'
        ]
        if any(t in combined for t in trusted):
            score += 20.0

        # 5. SPAM / STOCK PENALTY
        spam = ['pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 'cartoon', 'recipe', 'meme']
        if any(s in combined for s in spam):
            score -= 60.0

        scored.append((score, item, brand_match_count))

    if has_any_brand_match and brand_tokens:
        scored = [x for x in scored if x[2] > 0 or x[0] > 0]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]

async def test_searxng_connection(base_url: str = "http://localhost:8888") -> Dict[str, Any]:
    base_url = base_url.rstrip("/")
    test_endpoint = f"{base_url}/search"
    params = {
        "q": "ping",
        "category_images": 1,
        "format": "json"
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(test_endpoint, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results_count = len(data.get("results", []))
                return {
                    "ok": True,
                    "status_code": resp.status_code,
                    "base_url": base_url,
                    "results_count": results_count,
                    "message": f"Connected successfully! Returned {results_count} test items."
                }
            else:
                return {
                    "ok": False,
                    "status_code": resp.status_code,
                    "base_url": base_url,
                    "message": f"Server responded with HTTP {resp.status_code}: {resp.text[:200]}"
                }
    except Exception as e:
        return {
            "ok": False,
            "status_code": 0,
            "base_url": base_url,
            "message": f"Connection error: {str(e)}"
        }

async def search_searxng_image(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    brand: str = "",
    product_name: str = "",
    variant_label: str = "",
    is_variant: bool = False,
    safesearch: int = 0,
    time_range: str = "",
    language: str = "auto",
    top_limit: int = 6
) -> Dict[str, Any]:
    """
    Search SearXNG + PaddleOCR Verification Pipeline:
    1. Query SearXNG for candidate images.
    2. Rank candidates by metadata relevance.
    3. Run PaddleOCR to verify text directly on product packaging:
       - For Variant (with size): Verifies the requested size is printed on the pack.
         If candidate #1 does not have the size, moves to candidate #2, #3...
       - For Product (without size): Verifies brand is on the pack and rejects competitor brands.
    4. Selects the verified matching image.
    """
    base_url = base_url.rstrip("/")
    search_url = f"{base_url}/search"

    clean_q = sanitize_search_query(query)

    params = {
        "q": clean_q,
        "category_images": 1,
        "engines": "google images,bing images,duckduckgo images",
        "pageno": 1,
        "language": language,
        "time_range": time_range,
        "safesearch": safesearch,
        "format": "json"
    }

    try:
        response = await client.get(search_url, params=params)
        if response.status_code != 200:
            return {
                "status": "ERROR",
                "best_image": None,
                "alternatives": [],
                "error_message": f"HTTP {response.status_code}: {response.text[:120]}"
            }

        data = response.json()
        raw_results = data.get("results", [])

        if not raw_results:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": [],
                "error_message": "No images found for this query"
            }

        # Step 1: Pre-rank results by metadata
        ranked_results = score_and_rank_images(
            results=raw_results,
            brand=brand,
            product_name=product_name,
            variant_label=variant_label,
            is_variant=is_variant
        )
        if not ranked_results:
            ranked_results = raw_results

        # Convert to ImageResult objects
        candidate_items: List[ImageResult] = []
        for item in ranked_results[:top_limit]:
            img_src = normalize_url(item.get("img_src"))
            thumbnail_src = normalize_url(item.get("thumbnail_src"))
            page_url = normalize_url(item.get("url"))
            title = (item.get("title") or "").strip()
            resolution = item.get("resolution") or ""
            engine = item.get("engine") or ""

            direct_img = img_src or thumbnail_src or page_url
            thumb = thumbnail_src or img_src or direct_img

            if direct_img:
                candidate_items.append(ImageResult(
                    title=title,
                    img_src=direct_img,
                    thumbnail_src=thumb,
                    url=page_url,
                    resolution=str(resolution),
                    engine=engine
                ))

        if not candidate_items:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": [],
                "error_message": "No valid image links found"
            }

        # Step 2: PaddleOCR Inspection Loop ("agr ni h uska agla img me jaye")
        selected_index = 0
        ocr_matched_index = -1

        if ocr_engine:
            # Check up to top 4 candidates with OCR
            for idx, candidate in enumerate(candidate_items[:4]):
                img_to_ocr = candidate.thumbnail_src or candidate.img_src
                detected_words = await extract_image_ocr_text(client, img_to_ocr)

                if is_variant and variant_label:
                    has_size = verify_variant_size(detected_words, variant_label)
                    brand_check = verify_brand_ocr(detected_words, brand)

                    # If packet has the size and didn't fail brand check: EXACT MATCH!
                    if has_size and brand_check is not False:
                        ocr_matched_index = idx
                        break
                else:
                    # For Product Image (without size): verify brand is present
                    brand_check = verify_brand_ocr(detected_words, brand)
                    if brand_check is True:
                        ocr_matched_index = idx
                        break

            if ocr_matched_index != -1:
                selected_index = ocr_matched_index

        best_image = candidate_items[selected_index]
        alternatives = [img for i, img in enumerate(candidate_items) if i != selected_index]

        return {
            "status": "SUCCESS",
            "best_image": best_image,
            "alternatives": alternatives,
            "error_message": None
        }

    except httpx.TimeoutException:
        return {
            "status": "ERROR",
            "best_image": None,
            "alternatives": [],
            "error_message": "SearXNG request timed out"
        }
    except Exception as e:
        logger.exception("Error searching SearXNG with OCR")
        return {
            "status": "ERROR",
            "best_image": None,
            "alternatives": [],
            "error_message": str(e)
        }
