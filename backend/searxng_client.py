import httpx
import logging
import re
from typing import Dict, Any, List, Optional
from models import ImageResult

logger = logging.getLogger("searxng_client")

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

def score_and_rank_images(
    results: List[Dict[str, Any]],
    brand: str = "",
    product_name: str = "",
    variant_label: str = "",
    is_variant: bool = False
) -> List[Dict[str, Any]]:
    """
    Intelligently ranks image search results to eliminate wrong/competitor brands
    and select the single highest-accuracy product packshot.
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
                # Brand was requested but not found in this image
                score -= 80.0

        # 2. CORE PRODUCT NAME KEYWORD MATCHING
        for pt in prod_tokens:
            if pt in brand_tokens:
                continue
            if pt in combined:
                # Grocery/Product category nouns
                if pt in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 'masala', 'cornflour', 'poha', 'dalia']:
                    score += 45.0
                else:
                    score += 15.0

        # 3. VARIANT SIZE MATCHING (For variant search)
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
                    rf'\b{num}kg\b',
                    rf'\b{num}\s*ml\b',
                    rf'\b{num}ml\b'
                ]
                if any(re.search(p, combined) for p in patterns):
                    score += 50.0

        # 4. TRUSTED RETAILER BOOST (Clean e-commerce packshots)
        trusted = [
            'amazon', 'flipkart', 'bigbasket', 'bbassets', 'blinkit', 'grofers', 
            'zepto', 'jiomart', 'swiggy', 'tataneu', 'dailymart4u', 'f.nooncdn', 
            'exportersindia', 'indiamart', 'dmart'
        ]
        if any(t in combined for t in trusted):
            score += 20.0

        # 5. SPAM / STOCK / JUNK PENALTY
        spam = ['pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 'cartoon', 'recipe', 'meme']
        if any(s in combined for s in spam):
            score -= 60.0

        scored.append((score, item, brand_match_count))

    # If any image matched the brand, strictly exclude candidate images with zero brand match
    if has_any_brand_match and brand_tokens:
        scored = [x for x in scored if x[2] > 0 or x[0] > 0]

    # Sort descending by highest relevance score
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
    Search SearXNG for image results, scored and ranked for zero errors.
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

        # Score and rank results by accuracy to brand, product name, and size
        ranked_results = score_and_rank_images(
            results=raw_results,
            brand=brand,
            product_name=product_name,
            variant_label=variant_label,
            is_variant=is_variant
        )

        if not ranked_results:
            ranked_results = raw_results

        image_items: List[ImageResult] = []
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
                image_items.append(ImageResult(
                    title=title,
                    img_src=direct_img,
                    thumbnail_src=thumb,
                    url=page_url,
                    resolution=str(resolution),
                    engine=engine
                ))

        if not image_items:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": [],
                "error_message": "No valid image links found in results"
            }

        # The highest-scored image is the true best match
        best_image = image_items[0]
        alternatives = image_items[1:]

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
        logger.exception("Error searching SearXNG")
        return {
            "status": "ERROR",
            "best_image": None,
            "alternatives": [],
            "error_message": str(e)
        }
