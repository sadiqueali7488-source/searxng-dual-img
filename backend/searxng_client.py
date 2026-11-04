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

def is_query_multipack(query: str, variant_label: str = "") -> bool:
    """Detects if user query or variant specifies a multi-pack, combo, or bundle."""
    text = (query + " " + variant_label).lower()
    patterns = [
        r'\bpack\s*of\s*[2-9]\b',
        r'\bpack-[2-9]\b',
        r'\b(?:combo|bundle|multipack|twin\s*pack|dual\s*pack|set\s*of\s*[2-9])\b',
        r'\bbuy\s*[1-9]\s*get\s*[1-9]\b',
        r'\bbogo\b',
        r'\b\d+(?:\.\d+)?\s*(?:kg|g|gm|l|ml)\s*\+\s*\d+(?:\.\d+)?\s*(?:kg|g|gm|l|ml)\b',
        r'\b[2-9]\s*x\s*\d+\s*(?:kg|g|gm|l|ml)\b'
    ]
    return any(re.search(p, text) for p in patterns)

def is_candidate_multipack(title: str = "", url: str = "", words: Optional[List[str]] = None) -> bool:
    """Detects if candidate image shows or describes a multi-pack / twin pack / combo."""
    ocr_str = " ".join(words or []).lower()
    combined = f"{title.lower()} {url.lower()} {ocr_str}"
    
    # 1. URL signatures
    if re.search(r'pack[-_]?of[-_]?[2-9]', url.lower()) or 'pack-of-' in url.lower():
        return True
    if any(k in url.lower() for k in ['combo', 'bundle', 'twin-pack', 'twin_pack', 'multipack']):
        return True

    # 2. Text / OCR signatures
    patterns = [
        r'\bpack\s*of\s*[2-9]\b',
        r'\b[2-9]\s*pack\b',
        r'\btwin\s*pack\b',
        r'\bcombo\s*pack\b',
        r'\bcombo\b',
        r'\bbundle\b',
        r'\bset\s*of\s*[2-9]\b',
        r'\bbuy\s*[1-9]\s*get\s*[1-9]\b',
        r'\bbogo\b',
        r'\b[2-9]\s*x\s*\d+\s*(?:kg|g|gm|l|ml)\b',
        r'\b\d+(?:\.\d+)?\s*(?:kg|g|gm|l|ml)\s*\+\s*\d+(?:\.\d+)?\s*(?:kg|g|gm|l|ml)\b',
        r'\bmultipack\b',
        r'\bmulti-pack\b'
    ]
    return any(re.search(p, combined) for p in patterns)

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

COMPETITOR_BRANDS = [
    'rajbhog', 'narayani', 'patanjali', 'fortune', 'saffola', 'catch', 'everest',
    'mdh', 'dhara', 'gemini', 'emami', 'quaker', 'gm foods', 'swadeshi', 'jayesh',
    'kashi', 'mahatma', 'shri hans', 'tata', 'aashirvaad', '24 mantra', 'amul',
    'britannia', 'parle', 'sunfeast', 'haldiram', 'bikaji', 'mtr', 'shakti bhog',
    'aachi', 'priya', 'suhana', 'badshah', 'ramdev', 'annapurna', 'bawarchi',
    'sarveshwari', 'goldiee', 'goldie', 'suruchi', 'saras', 'devaiah', '9am'
]

def contains_conflicting_brand(text: str, target_brand: str) -> bool:
    """Checks if text contains a competitor brand that is not the target brand."""
    if not text or not target_brand:
        return False
    t_brand = target_brand.lower().strip()
    clean_text = text.lower()
    t_tokens = set(re.split(r'[\s\-_]+', t_brand))
    
    for cb in COMPETITOR_BRANDS:
        cb_tokens = set(re.split(r'[\s\-_]+', cb))
        if cb_tokens.issubset(t_tokens) or t_tokens.issubset(cb_tokens):
            continue
        if re.search(rf'\b{re.escape(cb)}\b', clean_text):
            return True
    return False

def verify_brand_match(words: List[str], title: str, brand: str, url: str = "") -> bool:
    """Checks if brand name exists in image OCR text, title, or URL, and strictly rejects competitor brands."""
    if not brand:
        return True
    brand_clean = brand.strip().lower()
    full_text = (" ".join(words) + " " + title + " " + url).lower()

    # 1. STRICT COMPETITOR BRAND CHECK:
    if contains_conflicting_brand(full_text, brand):
        return False

    # 2. IF OCR WORDS EXIST, VERIFY BRAND PRINTED ON PACKAGING:
    if words and len(words) >= 2:
        ocr_text = " ".join(words).lower()
        if 'aashirvaad' in brand_clean:
            if not ('aashirvaad' in ocr_text or 'asirvad' in ocr_text or 'ashirwad' in ocr_text):
                return False
        elif 'sampann' in brand_clean:
            if not ('sampann' in ocr_text or 'tata' in ocr_text):
                return False
        elif 'mantra' in brand_clean:
            if not ('mantra' in ocr_text or '24' in ocr_text):
                return False
        elif 'amul' in brand_clean:
            if not ('amul' in ocr_text):
                return False
        elif 'fortune' in brand_clean:
            if not ('fortune' in ocr_text):
                return False
        elif 'saffola' in brand_clean:
            if not ('saffola' in ocr_text):
                return False
        elif '9am' in brand_clean:
            if not ('9am' in ocr_text or '9 am' in ocr_text):
                return False

    # 3. Known brand keys
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
    """Ensures the core product and all distinct sub-types / flavors match (e.g. beetroot, methi, sattu)."""
    if not product_name:
        return True
    full_text = (" ".join(words) + " " + title + " " + url).lower()
    p_lower = product_name.lower()

    # 1. MILLET SPECIES LOCK:
    # Strictly isolate Foxtail, Bajra, Ragi, Jowar, Kodo, Little, Barnyard
    millet_species = {
        'foxtail': ['foxtail', 'kangni'],
        'bajra': ['bajra', 'pearl millet', 'pearlmillet'],
        'pearl': ['bajra', 'pearl millet', 'pearlmillet'],
        'ragi': ['ragi', 'finger millet', 'fingermillet', 'mandua'],
        'jowar': ['jowar', 'sorghum'],
        'kodo': ['kodo', 'kodra'],
        'little': ['little millet', 'kutki', 'sama'],
        'barnyard': ['barnyard', 'sanwa', 'samak']
    }
    for spec, aliases in millet_species.items():
        if spec in p_lower:
            if not any(a in full_text for a in aliases):
                return False
            for other_spec, other_aliases in millet_species.items():
                if other_spec == spec or other_spec in p_lower:
                    continue
                if any(re.search(rf'\b{re.escape(oa)}\b', full_text) for oa in other_aliases):
                    return False

    # 2. FORM LOCK: Flour / Atta vs Whole Grain
    has_flour = any(f in p_lower for f in ['flour', 'atta', 'peeth', 'powder'])
    if has_flour:
        cand_has_flour = any(f in full_text for f in ['flour', 'atta', 'peeth', 'powder'])
        # If image or text specifically states it is unhulled grain / pearled grain without flour:
        if ('pearled' in full_text or 'parboiled' in full_text) and not cand_has_flour:
            return False

    # 3. DISTINCTIVE FLAVOR / SUB-TYPE CHECK:
    distinctive_subtypes = [
        'beetroot', 'methi', 'khapli', 'multigrain', '7 grain', '7grain', 'sharbati',
        'ragi', 'sooji', 'suji', 'maida', 'besan', 'sattu', 'cornflour', 'corn flour',
        'rock salt', 'black salt', 'kala namak', 'iodised', 'iodized', 'poha', 'dalia'
    ]
    for sub in distinctive_subtypes:
        if sub in p_lower:
            if sub in ['7 grain', '7grain']:
                if not any(sg in full_text for sg in ['7 grain', '7grain', '7-grain']):
                    return False
            elif sub in ['cornflour', 'corn flour']:
                if not any(cf in full_text for cf in ['cornflour', 'corn flour', 'corn starch']):
                    return False
            elif sub not in full_text:
                return False

    # 4. CORE CATEGORIES LOCK (e.g. sattu != dal, salt != atta, oil != ghee)
    if 'sattu' in p_lower and 'sattu' not in full_text:
        return False
    if 'salt' in p_lower and any(c in full_text for c in ['atta', 'oil', 'flour']):
        return False
    if 'oil' in p_lower and any(c in full_text for c in ['salt', 'ghee', 'atta']):
        return False
    if 'ghee' in p_lower and 'oil' in full_text:
        return False

    return True

def score_and_rank_images(
    results: List[Dict[str, Any]],
    brand: str = "",
    product_name: str = "",
    variant_label: str = "",
    is_variant: bool = False
) -> List[Dict[str, Any]]:
    """Scores candidate images by metadata relevance and strict brand/combo filters."""
    if not results:
        return []

    brand_clean = (brand or "").strip().lower()
    brand_tokens = [t for t in re.split(r'[\s\-_]+', brand_clean) if len(t) >= 2]

    stop_words = {'100', 'pure', 'pack', 'pouch', 'packet', 'the', 'and', 'for', 'with', 'original', 'best', 'organic', 'grade'}
    prod_clean = (product_name or "").strip().lower()
    prod_tokens = [t for t in re.split(r'[\s\-_]+', prod_clean) if len(t) >= 3 and t not in stop_words]

    query_is_combo = is_query_multipack(product_name, variant_label)

    scored = []
    has_any_brand_match = False

    for item in results:
        title = (item.get("title") or "").lower()
        img_url = (item.get("img_src") or "").lower()
        page_url = (item.get("url") or "").lower()
        combined = f"{title} {img_url} {page_url}"

        # 0. STRICT BLOCKED DOMAINS (e.g. JioMart)
        blocked_domains = ['jiomart', 'jiomart.com']
        if any(b in combined for b in blocked_domains):
            continue

        # 1. STRICT COMPETITOR BRAND FILTER:
        # If competitor brand is found, completely drop candidate!
        if contains_conflicting_brand(combined, brand):
            continue

        # 2. MULTI-PACK VS SINGLE PACK FILTER:
        cand_is_combo = is_candidate_multipack(title, f"{img_url} {page_url}")
        if not query_is_combo and cand_is_combo:
            # Query is SINGLE pack (500g, 1kg), candidate is multi-pack/combo -> REJECT!
            continue

        # 3. MILLET SPECIES CONFLICT FILTER:
        if 'foxtail' in prod_clean:
            if any(m in combined for m in ['bajra', 'pearl millet', 'ragi', 'jowar', 'kodo']):
                continue
        if 'bajra' in prod_clean and any(m in combined for m in ['foxtail', 'ragi', 'jowar']):
            continue

        # 4. DISTINCTIVE FLAVOR / SUB-TYPE FILTER:
        if 'beetroot' in prod_clean and 'beetroot' not in combined:
            continue
        if 'sattu' in prod_clean and 'sattu' not in combined:
            continue

        score = 0.0

        # Combo preference
        if query_is_combo:
            if cand_is_combo:
                score += 100.0
            else:
                score -= 50.0

        # Brand matching
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

        # Product name keywords
        for pt in prod_tokens:
            if pt in brand_tokens:
                continue
            if pt in combined:
                if pt in ['beetroot', 'methi', 'khapli', 'ragi', 'sharbati', 'multigrain', '7grain']:
                    score += 80.0
                elif pt in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 'masala', 'cornflour', 'poha', 'dalia']:
                    score += 45.0
                else:
                    score += 25.0

        # Variant size metadata check
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
                    score += 60.0
        elif not is_variant:
            # For Product Image (without size), penalize images that have size in title!
            if re.search(r'\b\d+(?:\.\d+)?\s*(kg|g|gm|ml|l|ltr|lb)\b', combined):
                score -= 40.0

        # Trusted retailer boost
        trusted = [
            '24mantraorganic', 'aashirvaad', 'bigbasket', 'bbassets', 'blinkit', 'grofers', 
            'zepto', 'swiggy', 'tataneu', 'dailymart4u', 'f.nooncdn', 
            'exportersindia', 'indiamart', 'dmart', 'veehafoods', 'amazon', 'flipkart'
        ]
        if any(t in combined for t in trusted):
            score += 30.0

        # Spam / stock penalty
        spam = [
            'pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 
            'cartoon', 'recipe', 'meme', 'fishbioclear', 'alamy', 'shutterstock',
            'dreamstime', 'freepik', 'vecteezy', 'istockphoto'
        ]
        if any(s in combined for s in spam):
            score -= 100.0

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
      * Strictly rejects multi-packs/twin-packs when query asks for single pack.
      * Strictly rejects competitor brands (Rajbhog, Narayani, etc.).
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
                if any(b in direct_img.lower() or b in page_url.lower() for b in ['jiomart']):
                    continue

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

        query_is_combo = is_query_multipack(product_name, variant_label)
        selected_index = -1

        # Step 2: Strict Vision / PaddleOCR Inspection
        for idx, candidate in enumerate(candidate_items[:15]):
            cand_url = candidate.img_src
            page_url = candidate.url
            title = candidate.title
            cand_context = f"{cand_url} {page_url}"

            # Multi-pack check
            cand_multi = is_candidate_multipack(title, cand_context)
            if not query_is_combo and cand_multi:
                continue
            if query_is_combo and not cand_multi:
                continue

            # Fast metadata pre-check
            if not verify_brand_match([], title, brand, cand_context):
                continue
            if not verify_product_keyword_match([], title, product_name, cand_context):
                continue

            # Run OCR on packaging image
            img_to_ocr = candidate.thumbnail_src or candidate.img_src
            words = await extract_image_ocr_text(client, img_to_ocr, timeout=1.5)

            # Re-verify multi-pack with OCR text
            if not query_is_combo and is_candidate_multipack(title, cand_context, words):
                continue

            # Re-verify brand and category with OCR text
            if not verify_brand_match(words, title, brand, cand_context):
                continue
            if not verify_product_keyword_match(words, title, product_name, cand_context):
                continue

            if is_variant and variant_label:
                # WITH SIZE: Must contain exact variant size on pack / title / url
                has_exact = verify_variant_size(words, title, variant_label, cand_context)
                if has_exact:
                    selected_index = idx
                    break
            else:
                # WITHOUT SIZE: Must NOT contain size callout (1kg, 500g, etc.)!
                has_size = has_any_size_callout(words, title, cand_url)
                if not has_size:
                    selected_index = idx
                    break

        # Fallback: if strict size match wasn't met, search for any candidate with matching brand & product & combo consistency
        if selected_index == -1:
            for idx, candidate in enumerate(candidate_items):
                cand_context = f"{candidate.img_src} {candidate.url}"
                cand_multi = is_candidate_multipack(candidate.title, cand_context)
                if not query_is_combo and cand_multi:
                    continue

                words = ocr_cache.get(candidate.thumbnail_src or candidate.img_src, [])
                if verify_brand_match(words, candidate.title, brand, cand_context) and \
                   verify_product_keyword_match(words, candidate.title, product_name, cand_context):
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
