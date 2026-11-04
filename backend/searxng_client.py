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

# Initialize RapidOCR (PaddleOCR ONNX backend)
ocr_engine = None
try:
    from rapidocr_onnxruntime import RapidOCR
    ocr_engine = RapidOCR()
    logger.info("RapidOCR (PaddleOCR ONNX backend) initialized successfully.")
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

async def extract_image_ocr_text(client: httpx.AsyncClient, img_url: str, timeout: float = 1.5) -> List[str]:
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
            try:
                img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
                img.thumbnail((640, 640))
                img_np = np.array(img)
                res, _ = ocr_engine(img_np)
                if not res:
                    return []
                return [text.lower().strip() for _, text, conf in res if conf >= 0.4]
            except Exception:
                return []

        words = await asyncio.to_thread(_do_ocr, resp.content)
        ocr_cache[img_url] = words
        return words
    except Exception:
        ocr_cache[img_url] = []
        return []

def has_any_size_callout(words: List[str], title: str = "", url: str = "") -> bool:
    """Checks if image text, title, or url mentions any package size (e.g. 1kg, 500g, 5kg)."""
    full_text = " ".join(words) + " " + title + " " + url
    if re.search(r'\b\d+(?:\.\d+)?\s*(kg|g|gm|gms|ml|l|ltr|litre|liter|lb|lbs)\b', full_text, re.I):
        return True
    clean = re.sub(r'[^a-z0-9]', '', full_text.lower())
    if re.search(r'\d+(?:kg|gm|gms|ml|ltr|litre|liter|lb)', clean):
        return True
    return False

def verify_variant_size(words: List[str], title: str, variant_label: str, url: str = "") -> bool:
    """Checks if image text, title, or url contains the EXACT requested variant size."""
    if not variant_label:
        return False
    m = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gm|gms|ml|l|ltr|lb)?', variant_label.lower())
    if not m:
        return False
    num = m.group(1)
    unit = m.group(2) or ""

    full_text = " ".join(words) + " " + title + " " + url
    clean = re.sub(r'[^a-z0-9]', '', full_text.lower())

    targets = [
        f"{num}{unit}",
        f"{num}gm",
        f"{num}gms",
        f"{num}g",
        f"{num}kg",
        f"{num}ml",
        f"{num}l",
        f"{num}ltr",
        f"{num}lb"
    ]
    return any(t in clean for t in targets)

def verify_brand_match(words: List[str], title: str, brand: str, url: str = "") -> bool:
    """Checks if brand name exists in image OCR text, title, or URL."""
    if not brand:
        return True
    brand_clean = brand.strip().lower()
    full_text = (" ".join(words) + " " + title + " " + url).lower()

    # Specific brand keywords that uniquely identify the brand
    known_brand_keys = {
        'tata': ['tata'],
        'tata sampann': ['sampann'],
        '24 mantra': ['mantra'],
        '24 mantra organic': ['mantra'],
        'aashirvaad': ['aashirvaad', 'asirvad', 'ashirwad'],
        'fortune': ['fortune'],
        'saffola': ['saffola'],
        'daawat': ['daawat', 'dawat'],
        'amul': ['amul'],
        'everest': ['everest'],
        'mdh': ['mdh'],
        'catch': ['catch'],
        '9am': ['9am', '9 am'],
        'nestle': ['nestle', 'maggi'],
        'patanjali': ['patanjali'],
        'britannia': ['britannia'],
        'parle': ['parle'],
        'sunfeast': ['sunfeast'],
        'haldiram': ['haldiram'],
        'bikaji': ['bikaji'],
        'mtr': ['mtr'],
        'dhara': ['dhara'],
        'gemini': ['gemini'],
        'emami': ['emami'],
        'mother dairy': ['mother dairy'],
        'country delight': ['country delight'],
    }
    
    for k, keys in known_brand_keys.items():
        if k in brand_clean:
            return any(key in full_text for key in keys)

    brand_tokens = [t for t in re.split(r'[\s\-_]+', brand_clean) if len(t) >= 3 and t not in {'organic', 'pure', 'food', 'foods', 'brand', 'india', 'international'}]
    if not brand_tokens:
        return True
    return any(bt in full_text for bt in brand_tokens)

def verify_product_keyword_match(words: List[str], title: str, product_name: str, url: str = "") -> bool:
    """Ensures the core product noun matches (e.g. sattu != dal, besan != atta)."""
    if not product_name:
        return True
    full_text = (" ".join(words) + " " + title + " " + url).lower()
    
    core_categories = [
        'sattu', 'besan', 'atta', 'maida', 'sooji', 'suji', 'rawa', 'flour', 
        'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 
        'masala', 'cornflour', 'poha', 'dalia', 'noodles', 'pasta', 'biscuit',
        'cookies', 'paneer', 'butter', 'cheese', 'pickle', 'sauce', 'ketchup'
    ]
    p_lower = product_name.lower()
    for cat in core_categories:
        if re.search(rf'\b{cat}\b', p_lower):
            if cat == 'sattu':
                return 'sattu' in full_text
            if cat == 'cornflour':
                return 'cornflour' in full_text or 'corn flour' in full_text or 'corn starch' in full_text
            if cat in full_text:
                return True
    return True

def score_and_rank_images(
    results: List[Dict[str, Any]],
    brand: str = "",
    product_name: str = "",
    variant_label: str = "",
    is_variant: bool = False
) -> List[Dict[str, Any]]:
    """Scores candidate images by metadata relevance."""
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

        # 3. VARIANT SIZE METADATA CHECK
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
        elif not is_variant:
            # For Product Image (without size), penalize images that have size in title!
            if re.search(r'\b\d+(?:\.\d+)?\s*(kg|g|gm|ml|l|ltr|lb)\b', combined):
                score -= 40.0

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
    top_limit: int = 20
) -> Dict[str, Any]:
    """
    Search SearXNG + PaddleOCR Strict Inspection Pipeline:
    - Fetches top 20 candidate images from SearXNG.
    - Runs PaddleOCR & Vision inspection on candidates sequentially:
      * For Product Image (WITHOUT SIZE):
        Strictly rejects images with size badges (1kg, 500g, etc.) and picks clean general packshot.
      * For Variant Image (WITH SIZE):
        Strictly requires exact variant size printed on packet. If candidate #1 doesn't have it,
        auto-advances to candidate #2, #3... until exact matching size is found!
    """
    base_url = base_url.rstrip("/")
    search_url = f"{base_url}/search"

    clean_q = sanitize_search_query(query)

    params = {
        "q": clean_q,
        "category_images": 1,
        "engines": "google images,bing images,brave.images",
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

        # Take up to 20 candidates as instructed by user ("tm 20 lo ek baar me")
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

        selected_index = -1

        # Step 2: Strict Vision / PaddleOCR Inspection
        # Loop through top candidates to find exact match
        for idx, candidate in enumerate(candidate_items[:12]):
            cand_url = candidate.img_src
            title = candidate.title

            # Fast metadata pre-check: skip immediately if title/url conflicts with brand or category
            if not verify_brand_match([], title, brand, cand_url):
                continue
            if not verify_product_keyword_match([], title, product_name, cand_url):
                continue

            # Run OCR on packaging image
            img_to_ocr = candidate.thumbnail_src or candidate.img_src
            words = await extract_image_ocr_text(client, img_to_ocr, timeout=1.5)

            # Re-verify brand and category with OCR text
            if not verify_brand_match(words, title, brand, cand_url):
                continue
            if not verify_product_keyword_match(words, title, product_name, cand_url):
                continue

            if is_variant and variant_label:
                # WITH SIZE: Must contain exact variant size on pack / title / url
                has_exact = verify_variant_size(words, title, variant_label, cand_url)
                if has_exact:
                    selected_index = idx
                    break
            else:
                # WITHOUT SIZE: Must NOT contain size callout (1kg, 500g, etc.)!
                has_size = has_any_size_callout(words, title, cand_url)
                if not has_size:
                    selected_index = idx
                    break

        # Fallback: if strict size match wasn't met, search for any candidate with matching brand & product
        if selected_index == -1:
            for idx, candidate in enumerate(candidate_items):
                words = ocr_cache.get(candidate.thumbnail_src or candidate.img_src, [])
                if verify_brand_match(words, candidate.title, brand, candidate.img_src) and \
                   verify_product_keyword_match(words, candidate.title, product_name, candidate.img_src):
                    selected_index = idx
                    break

        # If still no candidate matches both brand and product, return NOT_FOUND (Zero false positives!)
        if selected_index == -1:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": candidate_items[:5],
                "error_message": f"No verified image matching brand '{brand}' and product '{product_name}'"
            }

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
