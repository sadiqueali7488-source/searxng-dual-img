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

# Initialize PP-OCRv6 Medium (PaddleOCR ONNX backend)
try:
    from ocr_engine import run_ocr
    logger.info("PP-OCRv6 Medium (PaddleOCR ONNX backend) initialized successfully.")
except Exception as e:
    logger.warning(f"PP-OCRv6 Medium failed to initialize: {e}")
    run_ocr = None

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

async def extract_image_ocr_text(client: httpx.AsyncClient, img_url: str, timeout: float = 2.0) -> List[str]:
    """
    Downloads image and runs PP-OCRv6 Medium to read text printed on packaging.
    Results are cached in memory.
    """
    if not img_url or run_ocr is None:
        return []

    if img_url in ocr_cache:
        return ocr_cache[img_url]

    try:
        resp = await client.get(img_url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 800:
            ocr_cache[img_url] = []
            return []

        words = await run_ocr(resp.content)
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

def verify_size_strict(words: List[str], title: str, variant_label: str, url: str = "") -> bool:
    """
    Universal size verifier supporting:
    1. FMCG Weights/Volumes: 1kg, 500g, 200ml, 143g, 1L, etc.
    2. Counts: 16 pcs, 4 finger, 12 bars, pack of 2
    3. Apparel Sizes: XS, S, M, L, XL, XXL, Free Size
    """
    if not variant_label:
        return False
    v_clean = variant_label.strip().lower()
    full_text = (" ".join(words) + " " + title + " " + url).lower()
    clean = re.sub(r'[^a-z0-9]', '', full_text)

    # 1. Standard FMCG Weight/Volume/Count: e.g. 1kg, 500g, 200ml, 143g, 16 pcs
    m = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|gm|gms|ml|l|ltr|lb|oz|pcs|pieces)?', v_clean)
    if m and m.group(1):
        num = m.group(1)
        unit = m.group(2) or ""
        targets = [
            f"{num}{unit}", f"{num}gm", f"{num}gms", f"{num}g",
            f"{num}kg", f"{num}ml", f"{num}l", f"{num}ltr", f"{num}lb",
            f"{num}pcs", f"{num}pieces", f"{num}oz"
        ]
        has_target = any(t in clean for t in targets)
        if not has_target:
            return False

        # Check if OCR words on the packet clearly show a conflicting weight
        if words and len(words) >= 2:
            ocr_str = " ".join(words).lower()
            other_sizes = re.findall(r'\b(\d+(?:\.\d+)?)\s*(kg|g|gm|gms|ml|l|ltr)\b', ocr_str)
            target_unit_norm = 'g' if unit in ['g', 'gm', 'gms'] else ('l' if unit in ['l', 'ltr', 'litre', 'liter'] else unit)
            for o_num, o_unit in other_sizes:
                o_unit_norm = 'g' if o_unit in ['g', 'gm', 'gms'] else ('l' if o_unit in ['l', 'ltr', 'litre', 'liter'] else o_unit)
                if o_num != num and o_unit_norm == target_unit_norm:
                    return False
        return True

    # 2. Apparel / Clothing Sizes: e.g. S, M, L, XL, XXL, Free Size
    apparel_sizes = {'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl', 'free size', 'freesize'}
    clean_v = re.sub(r'[^a-z0-9]', '', v_clean)
    if clean_v in apparel_sizes or v_clean in apparel_sizes:
        patterns = [
            rf'\bsize\s*{re.escape(v_clean)}\b',
            rf'\b{re.escape(v_clean)}\s*size\b',
            rf'\b{re.escape(v_clean)}\b'
        ]
        return any(re.search(p, full_text) for p in patterns) or clean_v in clean

    return clean_v in clean

COMPETITOR_BRANDS = [
    'rajbhog', 'narayani', 'patanjali', 'fortune', 'saffola', 'catch', 'everest',
    'mdh', 'dhara', 'gemini', 'emami', 'quaker', 'gm foods', 'swadeshi', 'jayesh',
    'kashi', 'mahatma', 'shri hans', 'tata', 'aashirvaad', '24 mantra', 'amul',
    'britannia', 'parle', 'sunfeast', 'haldiram', 'bikaji', 'mtr', 'shakti bhog',
    'aachi', 'priya', 'suhana', 'badshah', 'ramdev', 'annapurna', 'bawarchi',
    'sarveshwari', 'goldiee', 'goldie', 'suruchi', 'saras', 'devaiah', '9am'
]

GENERIC_BRAND_WORDS = {
    'super', 'royal', 'popular', 'premium', 'classic', 'gold', 'special', 
    'fresh', 'best', 'select', 'choice', 'daily', 'smart', 'star', 'plus',
    'saver', 'delight', 'lite', 'rich', 'shudh', 'original', 'natural',
    'organic', 'pure', 'food', 'foods', 'brand', 'brands', 'india', 
    'international', 'products', 'pvts', 'ltd', 'enterprises', 'enterprise'
}

DERIVATIVE_DISQUALIFIERS = [
    'noodles', 'noodle', 'biscuit', 'biscuits', 'cookie', 'cookies', 'cake',
    'ladoo', 'laddu', 'barfi', 'halwa', 'sev', 'bhujia', 'namkeen', 'chips',
    'snack', 'snacks', 'drink', 'pasta', 'vermicelli', 'papad'
]

SIBLING_FLOUR_CATEGORIES = {
    'maida': ['besan', 'sooji', 'suji', 'rava', 'dalia', 'poha'],
    'besan': ['maida', 'sooji', 'suji', 'rava', 'dalia', 'poha'],
    'sooji': ['maida', 'besan', 'atta'],
    'atta': ['maida', 'besan', 'sooji'],
    'salt': ['atta', 'oil', 'flour', 'besan', 'sooji'],
    'oil': ['salt', 'ghee', 'atta', 'flour'],
    'ghee': ['oil', 'butter milk']
}

def contains_conflicting_brand(text: str, target_brand: str) -> bool:
    """Checks if text contains a competitor brand that is not the target brand."""
    if not text or not target_brand:
        return False
    t_brand = target_brand.lower().strip()
    clean_text = text.lower()
    t_tokens = set(re.split(r'[\s\-_]+', t_brand))
    
    # Ignore corporate endorsement "tata enterprise" on BigBasket / bb packages
    if any(b in t_brand for b in ['bb', 'bigbasket']):
        clean_text = re.sub(r'\btata\s+enterprise[s]?\b', '', clean_text)

    for cb in COMPETITOR_BRANDS:
        cb_tokens = set(re.split(r'[\s\-_]+', cb))
        if cb_tokens.issubset(t_tokens) or t_tokens.issubset(cb_tokens):
            continue
        if re.search(rf'\b{re.escape(cb)}\b', clean_text):
            return True
    return False

def is_dynamic_brand_match(full_text: str, target_brand: str, ocr_words: Optional[List[str]] = None) -> bool:
    """
    Universal, strict dynamic brand matcher for ANY FMCG brand in ANY Excel sheet.
    Rejects partial generic words (e.g. 'super' matching 'bb Super Saver', or 'bb Royal' matching 'bb Super Saver').
    """
    if not target_brand:
        return True

    b_clean = target_brand.strip().lower()
    text_clean = full_text.lower()
    ocr_str = " ".join(ocr_words or []).lower()
    combined = f"{text_clean} {ocr_str}"

    # 1. Competitor conflict check
    if contains_conflicting_brand(combined, target_brand):
        return False

    # 2. Exact full brand phrase match (e.g. "bb super saver" in "bb-super-saver" or "bb supersaver")
    clean_b_str = re.sub(r'[^a-z0-9]', '', b_clean)
    clean_comb_str = re.sub(r'[^a-z0-9]', '', combined)

    if clean_b_str in clean_comb_str:
        return True

    # 3. Tokenize brand into non-filler words
    filler_words = {'organic', 'pure', 'food', 'foods', 'brand', 'brands', 'india', 'international', 'products', 'pvts', 'ltd'}
    tokens = [t for t in re.split(r'[\s\-_]+', b_clean) if len(t) >= 2 and t not in filler_words]
    if not tokens:
        return True

    # Single-word brand (e.g. "Aashirvaad", "Patanjali", "Amul", "Fortune", "9AM", "Catch", "Everest")
    if len(tokens) == 1:
        t = tokens[0]
        return bool(re.search(rf'\b{re.escape(t)}\b', combined) or t in clean_comb_str)

    # Multi-word brand (e.g. "bb Super Saver", "bb Royal", "Tata Sampann", "Tata Salt"):
    matched_count = 0
    for t in tokens:
        if re.search(rf'\b{re.escape(t)}\b', combined) or t in clean_comb_str:
            matched_count += 1

    match_ratio = matched_count / len(tokens)

    # For 2-word brands (e.g. "Tata Sampann", "bb Royal"): both words must match (100%)
    if len(tokens) == 2:
        return match_ratio >= 1.0
    else:
        # For 3+ words (e.g. 'bb super saver'): at least 67% must match AND primary first token must match
        first_token = tokens[0]
        first_matches = bool(re.search(rf'\b{re.escape(first_token)}\b', combined) or first_token in clean_comb_str)
        return match_ratio >= 0.67 and first_matches

def verify_brand_match(words: List[str], title: str, brand: str, url: str = "") -> bool:
    """Combines dynamic brand matching with OCR packaging verification."""
    full_context = f"{title} {url}"
    return is_dynamic_brand_match(full_context, brand, words)

def verify_product_keyword_match(words: List[str], title: str, product_name: str, brand: str = "", url: str = "") -> bool:
    """
    Dynamic, order-independent product matcher.
    - If product name is short (<= 2 words), matches exact product nouns and rejects conflicting derivatives.
    - Matches tokens in ANY order/permutation in packaging OCR text or title.
    - Rejects derivative food products (e.g. noodles, ladoo, biscuits, cake, drinks) when raw staple is requested.
    - Universal for FMCG, Chocolates, Groceries, and Apparel.
    """
    if not product_name:
        return True
    ocr_str = " ".join(words or []).lower()
    full_text = f"{ocr_str} {title} {url}".lower()
    p_lower = product_name.lower().strip()
    b_lower = (brand or "").lower().strip()

    # 1. UNIVERSAL PRODUCT TOKEN MATCHING:
    # Extracts clean product tokens (excluding brand words, stop words, and unit tokens)
    b_tokens = set(re.split(r'[\s\-_]+', b_lower))
    raw_p_tokens = re.split(r'[\s\-_,()]+', p_lower)
    
    stop_words = {
        'the', 'and', 'for', 'with', 'in', 'of', 'at', 'by', 'from', 'to', 'a', 'an',
        'pure', 'original', 'best', 'organic', 'grade', 'premium', 'new', 'pack', 
        'pouch', 'packet', 'box', 'bottle', 'piece', 'pieces', 'pcs', 'product', 
        'item', 'quality', 'genuine', 'online'
    }
    prod_tokens = []
    for t in raw_p_tokens:
        t_clean = t.strip()
        if len(t_clean) < 3 or t_clean in b_tokens or t_clean in stop_words:
            continue
        if re.match(r'^\d+(?:kg|g|gm|ml|l|ltr|pcs|piece|oz|lb)?$', t_clean):
            continue
        prod_tokens.append(t_clean)

    clean_comb_str = re.sub(r'[^a-z0-9]', '', full_text)

    # 2. Dynamic Order-Independent Product Token Check:
    if prod_tokens:
        matched = 0
        for pt in prod_tokens:
            if re.search(rf'\b{re.escape(pt)}\b', full_text) or pt in clean_comb_str:
                matched += 1
        
        # If product name has 1-2 words (e.g. "Maida", "Besan", "Chana Sattu", "Salt", "Crop Top"):
        # ALL tokens must be present! Zero dilution allowed.
        if len(prod_tokens) <= 2:
            if matched < len(prod_tokens):
                return False
        else:
            # Multi-word product (e.g. "Meri Chakki Beetroot Atta", "Foxtail Millet Flour"):
            # Must match at least 80% of tokens
            if matched / len(prod_tokens) < 0.8:
                return False

    # 3. Disqualify snack / derivative modifications if target doesn't ask for them
    for disq in DERIVATIVE_DISQUALIFIERS:
        if disq not in p_lower and re.search(rf'\b{re.escape(disq)}\b', full_text):
            return False

    # 4. Sibling category conflict check for short product names
    for cat, siblings in SIBLING_FLOUR_CATEGORIES.items():
        if cat in p_lower:
            for sib in siblings:
                if sib not in p_lower and re.search(rf'\b{re.escape(sib)}\b', full_text):
                    return False

    # 5. MILLET SPECIES LOCK:
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

    # 6. FORM LOCK: Flour / Atta vs Whole Grain
    has_flour = any(f in p_lower for f in ['flour', 'atta', 'peeth', 'powder'])
    if has_flour:
        cand_has_flour = any(f in full_text for f in ['flour', 'atta', 'peeth', 'powder', 'maida', 'besan', 'sooji', 'sattu'])
        if ('pearled' in full_text or 'parboiled' in full_text) and not cand_has_flour:
            return False

    # 7. DISTINCTIVE FLAVOR / SUB-TYPE CHECK (Order-independent):
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
            elif sub not in full_text and sub not in clean_comb_str:
                return False

    # 8. CORE CATEGORIES LOCK (e.g. sattu != dal, salt != atta, oil != ghee, maida != besan)
    if 'sattu' in p_lower and 'sattu' not in full_text:
        return False
    if 'maida' in p_lower and not any(m in full_text for m in ['maida', 'refined wheat flour', 'all purpose flour']):
        return False
    if 'besan' in p_lower and not any(b in full_text for b in ['besan', 'gram flour', 'chana flour']):
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
    """Scores candidate images by metadata relevance, universal dynamic brand check, and combo filters."""
    if not results:
        return []

    brand_clean = (brand or "").strip().lower()
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

        # 1. STRICT COMPETITOR BRAND FILTER
        if contains_conflicting_brand(combined, brand):
            continue

        # 2. DYNAMIC BRAND CHECK:
        # If target brand is given, candidate MUST match the brand dynamically (no generic single word false positives!)
        if brand and not is_dynamic_brand_match(combined, brand):
            continue

        # 3. MULTI-PACK VS SINGLE PACK FILTER
        cand_is_combo = is_candidate_multipack(title, f"{img_url} {page_url}")
        if not query_is_combo and cand_is_combo:
            continue

        # 4. MILLET SPECIES CONFLICT FILTER
        if 'foxtail' in prod_clean:
            if any(m in combined for m in ['bajra', 'pearl millet', 'ragi', 'jowar', 'kodo']):
                continue
        if 'bajra' in prod_clean and any(m in combined for m in ['foxtail', 'ragi', 'jowar']):
            continue

        # 5. DISTINCTIVE FLAVOR / SUB-TYPE FILTER
        if 'beetroot' in prod_clean and 'beetroot' not in combined:
            continue
        if 'sattu' in prod_clean and 'sattu' not in combined:
            continue
        if 'maida' in prod_clean and not any(m in combined for m in ['maida', 'refined wheat flour']):
            continue

        score = 0.0

        # Prioritize Google Images engine results above all others
        if item.get("engine") == "google images":
            score += 500.0

        # Combo preference
        if query_is_combo:
            if cand_is_combo:
                score += 100.0
            else:
                score -= 50.0

        # Dynamic Brand scoring
        if brand:
            score += 100.0
            has_any_brand_match = True

        # Product name keywords
        for pt in prod_tokens:
            if pt in combined:
                if pt in ['beetroot', 'methi', 'khapli', 'ragi', 'sharbati', 'multigrain', '7grain']:
                    score += 80.0
                elif pt in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 'masala', 'cornflour', 'poha', 'dalia']:
                    score += 50.0
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
            score += 35.0

        # Official brand store domain extra boost
        brand_domain_slug = re.sub(r'[^a-z0-9]', '', brand_clean)
        if brand_domain_slug and brand_domain_slug in page_url:
            score += 50.0

        # Spam / stock penalty
        spam = [
            'pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 
            'cartoon', 'recipe', 'meme', 'fishbioclear', 'alamy', 'shutterstock',
            'dreamstime', 'freepik', 'vecteezy', 'istockphoto'
        ]
        if any(s in combined for s in spam):
            score -= 100.0

        scored.append((score, item))

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
      * Strictly rejects competitor brands and partial generic word matches.
    """
    base_url = base_url.rstrip("/")
    search_url = f"{base_url}/search"

    clean_q = sanitize_search_query(query)

    params = {
        "q": clean_q,
        "category_images": 1,
        "engines": "google images",
        "pageno": 1,
        "language": language,
        "time_range": time_range,
        "safesearch": safesearch,
        "format": "json"
    }

    try:
        raw_results = []
        try:
            # 1. Primary: Use Google Images strictly
            resp_g = await client.get(search_url, params=params)
            if resp_g.status_code == 200:
                raw_results = resp_g.json().get("results", [])
        except Exception:
            raw_results = []

        # 2. Resilient Fallback: If Google Images is temporarily blocked/suspended by SearXNG, query bing images
        if not raw_results:
            fallback_params = {**params, "engines": "bing images"}
            resp_fb = await client.get(search_url, params=fallback_params)
            if resp_fb.status_code == 200:
                raw_results = resp_fb.json().get("results", [])
            else:
                return {
                    "status": "ERROR",
                    "best_image": None,
                    "alternatives": [],
                    "error_message": f"HTTP {resp_fb.status_code}: {resp_fb.text[:120]}"
                }

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
            cand_context = f"{title} {cand_url} {page_url}"

            # Multi-pack check
            cand_multi = is_candidate_multipack(title, cand_context)
            if not query_is_combo and cand_multi:
                continue
            if query_is_combo and not cand_multi:
                continue

            # Fast metadata pre-check
            if not is_dynamic_brand_match(cand_context, brand):
                continue
            if not verify_product_keyword_match([], title, product_name, brand, cand_context):
                continue

            # Run OCR on packaging image
            img_to_ocr = candidate.thumbnail_src or candidate.img_src
            words = await extract_image_ocr_text(client, img_to_ocr, timeout=1.5)

            # Re-verify multi-pack with OCR text
            if not query_is_combo and is_candidate_multipack(title, cand_context, words):
                continue

            # Re-verify brand and category with OCR text
            if not is_dynamic_brand_match(cand_context, brand, words):
                continue
            if not verify_product_keyword_match(words, title, product_name, brand, cand_context):
                continue

            if is_variant and variant_label:
                # WITH SIZE: Must contain exact variant size on pack / title / url
                has_exact = verify_size_strict(words, title, variant_label, cand_context)
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
                cand_context = f"{candidate.title} {candidate.img_src} {candidate.url}"
                cand_multi = is_candidate_multipack(candidate.title, cand_context)
                if not query_is_combo and cand_multi:
                    continue

                words = ocr_cache.get(candidate.thumbnail_src or candidate.img_src, [])
                if is_dynamic_brand_match(cand_context, brand, words) and \
                   verify_product_keyword_match(words, candidate.title, product_name, brand, cand_context):
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
