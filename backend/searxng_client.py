import asyncio
import hashlib
import io
import json
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

# OCR text and search results are cached on disk (cache_store) so nothing is done twice
import cache_store

# Images OCR'd at the same time per search (the OCR pool runs them in parallel)
OCR_BATCH = 4
MAX_ALTERNATIVES = 8

# Image CDNs often block non-browser clients; without this many downloads (and so OCR) fail.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# Never use images from these hosts: social-media crawler proxies (e.g. lookaside.fbsbx.com/lookaside/crawler/...),
# expiring CDN links, and stock/aggregator sites. Matched against the image URL and the page URL host.
BLOCKED_IMAGE_DOMAINS = [
    'jiomart',
    'fbsbx.com', 'lookaside', 'fbcdn.net', 'facebook.com', 'fb.com', 'instagram.com', 'cdninstagram.com',
    'whatsapp', 'threads.net', 'pinimg.com', 'pinterest', 'twimg.com', 'twitter.com', 'x.com',
    'ytimg.com', 'youtube.com', 'tiktok', 'redd.it', 'reddit.com', 'licdn.com', 'linkedin.com',
    'snapchat', 'sharechat', 'telegram',
    'alamy', 'shutterstock', 'dreamstime', 'freepik', 'vecteezy', 'istockphoto', 'gettyimages', '123rf',
]

def _host(url: str) -> str:
    m = re.match(r'^(?:https?:)?//([^/?#]+)', (url or "").strip().lower())
    return m.group(1) if m else ""

def is_blocked_source(img_url: str = "", page_url: str = "") -> bool:
    """True for images hosted on, or found via, a blocked domain."""
    full = (img_url or "").lower()
    if '/lookaside/' in full or '/crawler/' in full:
        return True
    for host in (_host(img_url), _host(page_url)):
        if not host:
            continue
        for b in BLOCKED_IMAGE_DOMAINS:
            # 'fbsbx.com' matches lookaside.fbsbx.com but 'x.com' must not match box.com
            if ('.' in b and (host == b or host.endswith('.' + b))) or ('.' not in b and b in host):
                return True
    return False

def normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    # BigBasket: replace secondary image indices (_2-, _3-, _4-) with primary front packshot (_1-)
    if 'bbassets' in url.lower():
        url = re.sub(r'(\d+)_[2-9]-', r'\g<1>_1-', url)
    return url

def sanitize_search_query(text: str) -> str:
    """Removes special characters that break or confuse image search engines."""
    s = re.sub(r'[%#*+=\\\[\]{}()<>"~`]', ' ', text or "")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

async def _download_image(client: httpx.AsyncClient, url: str, timeout: float) -> Optional[bytes]:
    if not url or not url.startswith("http"):
        return None
    try:
        resp = await client.get(url, timeout=timeout, headers=BROWSER_HEADERS, follow_redirects=True)
        if resp.status_code != 200 or len(resp.content) < 800:
            return None
        if "html" in resp.headers.get("content-type", ""):
            return None
        return resp.content
    except Exception:
        return None

async def extract_image_ocr_text(client: httpx.AsyncClient, img_url: str, fallback_url: str = "", timeout: float = 6.0) -> List[str]:
    """
    Downloads the FULL image (falling back to the search thumbnail) and runs PP-OCRv6 Medium.
    Small thumbnails hide nutrition tables / batch text, so back panels were slipping through.
    Returns [] when the image could not be downloaded; failures are not cached (may be temporary).
    """
    if not img_url or run_ocr is None:
        return []

    cached = cache_store.get_ocr(img_url)
    if cached is not None:
        return cached

    content = await _download_image(client, img_url, timeout)
    if content is None and fallback_url and fallback_url != img_url:
        content = await _download_image(client, fallback_url, timeout)
    if content is None:
        return []

    try:
        words = await run_ocr(content)
    except Exception:
        return []
    cache_store.put_ocr(img_url, words)
    return words

def ocr_shows_front(words: List[str], brand: str = "", product_name: str = "") -> bool:
    """
    Front-of-pack evidence: OCR could read the image AND the brand or product name is printed on it.
    An image we could not read, or one showing neither (side panel, plain back), is not accepted as a front.
    """
    if not words:
        return False
    ocr = " ".join(words).lower()
    compact = re.sub(r'[^a-z0-9]', '', ocr)
    if not brand and not product_name:
        return True
    tokens = [t for t in re.split(r'[\s\-_,()/&]+', f"{brand} {product_name}".lower())
              if len(t) >= 4 and t not in GENERIC_BRAND_WORDS and not re.match(r'^\d', t)]
    return any(t in compact for t in tokens) if tokens else True

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

SPECIALTY_FLOUR_MODIFIERS = [
    'singhara', 'singhada', 'water chestnut', 'kuttu', 'buckwheat',
    'samak', 'sama', 'rajgira', 'amaranth', 'makka', 'makki', 'maize',
    'chawal', 'rice flour', 'soya', 'soyabean', 'bajra', 'jowar', 'ragi',
    'oats', 'oat', 'barley', 'jau', 'khapli', 'emmer', 'beetroot', 'methi',
    'multigrain', 'multi grain', '7 grain', '7grain', 'missi'
]

def has_specialty_flour_conflict(target_prod: str, cand_text: str) -> bool:
    """
    If target product is a staple (e.g. Atta, Besan, Maida, Sooji) without a specialty modifier,
    reject candidates that introduce specialty flour varieties (e.g. Singhara Aata for Plain Atta).
    """
    p_lower = target_prod.lower()
    c_lower = cand_text.lower()
    for spec in SPECIALTY_FLOUR_MODIFIERS:
        if spec not in p_lower and re.search(rf'\b{re.escape(spec)}\b', c_lower):
            return True
    return False

MILLET_GROUPS = {
    'bajra': ['bajra', 'pearl millet', 'pearlmillet', 'sajjalu', 'kambu'],
    'foxtail': ['foxtail', 'kangni', 'kakum', 'korra', 'tenai'],
    'ragi': ['ragi', 'finger millet', 'fingermillet', 'mandua', 'nachni', 'kezhvaragu'],
    'jowar': ['jowar', 'sorghum', 'cholam', 'jola'],
    'kodo': ['kodo', 'kodra', 'varagu', 'arika'],
    'little': ['little millet', 'kutki', 'sama', 'samai'],
    'barnyard': ['barnyard', 'sanwa', 'samak', 'kuthiraivali'],
    'proso': ['proso', 'chena', 'barri', 'panivaragu']
}

BACK_PANEL_URL_PATTERNS = [
    r'[-_]back\b', r'/back/', r'[-_]bk\b', r'[-_]b\.(?:jpg|jpeg|png|webp)',
    r'nutrition', r'ingredient', r'backpanel', r'back-panel', r'back_panel',
    r'barcode', r'facts[-_]panel', r'[-_]pt0[1-9]', r'\.pt0[1-9]\.'
]

BACK_PANEL_KEYWORDS = [
    'nutritional information', 'nutrition facts', 'nutritional facts', 'nutrition table',
    'nutritional values', 'nutritional value', 'typical values', 'approximate values',
    'per 100g', 'per 100 g', 'serving size', 'servings per container', 'servings per pack',
    'energy (kcal)', 'energy kcal', 'dietary fiber', 'dietary fibre',
    'saturated fat', 'trans fat', 'added sugars', 'added sugar',
    'lic. no.', 'lic no.', 'lic no', 'licence no', 'license no', 'fssai lic',
    'manufactured by', 'mfd. by', 'mfd by', 'mfg. by', 'mfg by', 'packed by', 'pkd. by', 'pkd by',
    'marketed by', 'consumer care', 'customer care', 'feedback/queries',
    'store in a cool', 'keep in an airtight', 'batch no', 'lot no', 'mfg date', 'exp date', 'best before'
]

def is_back_panel_or_nutrition(title: str = "", url: str = "", words: Optional[List[str]] = None) -> bool:
    """
    Strictly detects if an image shows a back panel, nutrition table, ingredients list, or regulatory label.
    Front packshots must be clean single packaging shots, never back panels or nutrition tables.
    """
    t_lower = (title or "").lower()
    u_lower = (url or "").lower()
    w_lower = " ".join(words or []).lower()
    combined = f"{t_lower} {u_lower} {w_lower}"

    # 1. Obvious URL markers
    for pat in BACK_PANEL_URL_PATTERNS:
        if re.search(pat, u_lower):
            return True

    # BigBasket secondary image check (_2-, _3-, _4- are back/side panels)
    if 'bbassets' in u_lower and re.search(r'\d+_[2-9]-', u_lower):
        return True

    # 2. Definite nutrition table phrases in OCR or Title
    direct_nutrition_phrases = [
        'nutritional information', 'nutrition facts', 'nutritional facts',
        'nutrition table', 'nutritional values', 'typical values', 'approximate values',
        'per 100g', 'per 100 g', 'serving size', 'servings per container', 'servings per pack'
    ]
    if any(p in combined for p in direct_nutrition_phrases):
        return True

    # 3. Macro nutrients cluster (Energy kcal / carbohydrate / protein / fat / fiber).
    # Indian front packs carry a small "per serve: energy, protein, sugar" badge, so one or two
    # macros alone are not enough: require a real table (3+ macros) or a text-heavy image.
    line_count = len(words or [])
    has_energy = 'energy' in w_lower or 'kcal' in w_lower
    macro_count = sum(1 for m in ['carbohydrate', 'protein', 'total fat', 'saturated', 'sugar', 'fiber', 'fibre', 'cholesterol', 'sodium']
                      if m in w_lower)
    if has_energy and (macro_count >= 3 or (macro_count >= 1 and line_count >= 25)):
        return True
    if ('energy' in t_lower or 'kcal' in t_lower) and any(m in t_lower for m in ['carbohydrate', 'protein', 'fat']):
        return True

    # 3.1 Text density: a front pack has a brand, a name and a few claims (~5-25 lines).
    # Back panels and front+back composites (e.g. a 1000px image showing both sides) read 40+ lines.
    if line_count >= 40:
        return True

    # 3.2 Contact / address block (only printed on the back)
    contact_hits = sum(1 for p in [
        r'@[a-z0-9-]+\.(?:com|in|co)', r'\b1800\s*\d{3}', r'toll\s*free', r'p\.?\s*o\.?\s*box', r'feedback',
        r'complaint', r'\bpin\s*code\b', r'\b\d{3}\s?\d{3}\b.*\b(?:india|bengaluru|mumbai|delhi|kolkata|chennai)\b',
        r'(?:batch|lot)\s*no', r'\bpkd\.?\b', r'\bmfd\.?\b', r'use\s*by', r'\bmrp\b.*incl', r'net\s*(?:weight|wt|qty|quantity)\s*[:.]'
    ] if re.search(p, w_lower))
    if contact_hits >= 2:
        return True

    # 4. Regulatory & Manufacturer Back-panel Cluster
    has_regulatory = any(r in combined for r in ['lic. no', 'lic no', 'licence no', 'license no', 'fssai'])
    has_mfg = any(m in combined for m in ['manufactured by', 'mfd by', 'mfg by', 'packed by', 'pkd by', 'marketed by', 'customer care', 'consumer care'])
    has_ingredients = any(i in combined for i in ['ingredient:', 'ingredients:', 'ingredient :', 'ingredients :'])
    
    if (has_regulatory and has_mfg) or (has_regulatory and has_ingredients) or (has_mfg and has_ingredients):
        return True

    # 5. Multiple back-panel keywords
    indicator_count = sum(1 for kw in BACK_PANEL_KEYWORDS if kw in combined)
    if indicator_count >= 2:
        return True

    return False

def is_candidate_multipack(title: str = "", url: str = "", words: Optional[List[str]] = None, resolution: str = "") -> bool:
    """Detects if candidate image shows or describes a multi-pack / twin pack / combo / brand range banner / wide collage."""
    ocr_str = " ".join(words or []).lower()
    combined = f"{title.lower()} {url.lower()} {ocr_str}"
    
    # 1. URL signatures (e.g. Amazon A+ promotional multi-product banners)
    if any(k in url.lower() for k in ['aplus-media', '/aplus/', 'aplusimaging', 'aplus_data', 'banner']):
        return True
    if re.search(r'pack[-_]?of[-_]?[2-9]', url.lower()) or 'pack-of-' in url.lower():
        return True
    if any(k in url.lower() for k in ['combo', 'bundle', 'twin-pack', 'twin_pack', 'multipack']):
        return True

    # 2. Text / OCR signatures of multi-product brand range banners & collages
    range_patterns = [
        r'\bexplore\s+(?:the\s+)?(?:wholesome\s+)?range\b',
        r'\bwholesome\s+range\b',
        r'\brange\s+of\s+[a-z]+\b',
        r'\bour\s+range\b',
        r'\bentire\s+range\b',
        r'\bcomplete\s+range\b',
        r'\bproduct\s+range\b',
        r'\bvariety\s+pack\b',
        r'\ball\s+variants\b',
        r'\bwide\s+range\b'
    ]
    if any(re.search(p, combined) for p in range_patterns):
        return True

    # 3. Standard multi-pack patterns
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
    if any(re.search(p, combined) for p in patterns):
        return True

    # 4. Aspect Ratio check for wide horizontal banners or multi-angle stitched collages
    if resolution:
        nums = re.findall(r'\d+', resolution)
        if len(nums) >= 2:
            w, h = float(nums[0]), float(nums[1])
            if h > 0:
                ratio = w / h
                # Single packshots are square or portrait (0.55 <= ratio <= 1.50).
                # Banners and 3-pack collages have ratio > 1.52 or < 0.50
                if ratio > 1.25 or ratio < 0.50:
                    return True

    return False

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

_UNIT_TO_BASE = {
    'kg': ('mass', 1000), 'kgs': ('mass', 1000), 'g': ('mass', 1), 'gm': ('mass', 1), 'gms': ('mass', 1),
    'gram': ('mass', 1), 'grams': ('mass', 1), 'l': ('vol', 1000), 'ltr': ('vol', 1000), 'litre': ('vol', 1000),
    'liter': ('vol', 1000), 'ml': ('vol', 1),
}
_SIZE_RE = re.compile(r'(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*(kgs?|grams?|gms?|gm|g|ltr|litre|liter|l|ml)(?![a-z])')

def _parse_sizes(text: str) -> set:
    """'1 Kg' -> {('mass', 1000.0)}; '500 ml' -> {('vol', 500.0)}."""
    out = set()
    for num, unit in _SIZE_RE.findall((text or "").lower()):
        dim, factor = _UNIT_TO_BASE[unit]
        out.add((dim, round(float(num) * factor, 1)))
    return out

def _ocr_shows_other_size(words: List[str], variant_label: str) -> bool:
    """True when the pack itself shows a weight/volume, but none equals the requested one (500g pack for 1kg)."""
    targets = _parse_sizes(variant_label)
    if not targets:
        return False
    dims = {d for d, _ in targets}
    # "per serve (30 g)" / "per 100 g" badges are not the pack size
    ocr = re.sub(r'per\s*(?:serve|serving|100)\s*\(?\s*\d*(?:\.\d+)?\s*(?:g|gm|ml)?\s*\)?', ' ', " ".join(words or []).lower())
    found = {s for s in _parse_sizes(ocr) if s[0] in dims}
    return bool(found) and not (found & targets)

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

    # Brand transliteration aliases (e.g. Aahar <-> Ahaar)
    BRAND_TRANSLITERATIONS = {
        'aahar': 'ahaar',
        'ahaar': 'aahar'
    }
    trans = BRAND_TRANSLITERATIONS.get(b_clean)
    if trans and trans in clean_comb_str:
        return True

    # 3. Tokenize brand into non-filler words
    filler_words = {'organic', 'pure', 'food', 'foods', 'brand', 'brands', 'india', 'international', 'products', 'pvts', 'ltd'}
    tokens = [t for t in re.split(r'[\s\-_]+', b_clean) if len(t) >= 2 and t not in filler_words]
    if not tokens:
        return True

    # Single-word brand (e.g. "Aashirvaad", "Patanjali", "Amul", "Fortune", "9AM", "Catch", "Everest")
    if len(tokens) == 1:
        t = tokens[0]
        trans_t = BRAND_TRANSLITERATIONS.get(t)
        return bool(
            re.search(rf'\b{re.escape(t)}\b', combined) or t in clean_comb_str or
            (trans_t and (re.search(rf'\b{re.escape(trans_t)}\b', combined) or trans_t in clean_comb_str))
        )

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

    # 0. SPECIALTY FLOUR CONFLICT CHECK (e.g. Singhara Aata for Plain Atta)
    if has_specialty_flour_conflict(p_lower, full_text):
        return False

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
    MILLET_GROUPS = {
        'bajra': ['bajra', 'pearl millet', 'pearlmillet', 'sajjalu', 'kambu'],
        'foxtail': ['foxtail', 'kangni', 'kakum', 'korra', 'tenai'],
        'ragi': ['ragi', 'finger millet', 'fingermillet', 'mandua', 'nachni', 'kezhvaragu'],
        'jowar': ['jowar', 'sorghum', 'cholam', 'jola'],
        'kodo': ['kodo', 'kodra', 'varagu', 'arika'],
        'little': ['little millet', 'kutki', 'sama', 'samai'],
        'barnyard': ['barnyard', 'sanwa', 'samak', 'kuthiraivali'],
        'proso': ['proso', 'chena', 'barri', 'panivaragu']
    }

    target_millets = set()
    for m_key, m_aliases in MILLET_GROUPS.items():
        if any(re.search(rf'\b{re.escape(a)}\b', p_lower) for a in m_aliases):
            target_millets.add(m_key)

    if target_millets:
        matched_target_millet = False
        for tm in target_millets:
            if any(a in full_text for a in MILLET_GROUPS[tm]):
                matched_target_millet = True
                break
        if not matched_target_millet:
            return False

        for other_m, other_aliases in MILLET_GROUPS.items():
            if other_m in target_millets:
                continue
            for oa in other_aliases:
                if re.search(rf'\b{re.escape(oa)}\b', full_text):
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

        # 0. STRICT BLOCKED DOMAINS (JioMart, Facebook/Instagram crawler links, Pinterest, stock sites...)
        if is_blocked_source(img_url, page_url) or is_blocked_source(item.get("thumbnail_src") or ""):
            continue

        # 1. STRICT COMPETITOR BRAND FILTER
        if contains_conflicting_brand(combined, brand):
            continue

        # 2. DYNAMIC BRAND CHECK:
        # If target brand is given, candidate MUST match the brand dynamically (no generic single word false positives!)
        if brand and not is_dynamic_brand_match(combined, brand):
            continue

        # 3. MULTI-PACK VS SINGLE PACK FILTER (including A+ banners & collages)
        cand_is_combo = is_candidate_multipack(title, f"{img_url} {page_url}", resolution=item.get("resolution", ""))
        if not query_is_combo and cand_is_combo:
            continue

        # 3.1 SPECIALTY FLOUR CONFLICT CHECK
        if has_specialty_flour_conflict(prod_clean, combined):
            continue

        # 3.2 BACK PANEL / NUTRITION LABEL FILTER
        if is_back_panel_or_nutrition(title, f"{img_url} {page_url}"):
            continue

        # 4. MILLET SPECIES CONFLICT FILTER
        cand_has_millet_conflict = False
        for m_key, m_aliases in MILLET_GROUPS.items():
            if any(re.search(rf'\b{re.escape(a)}\b', prod_clean) for a in m_aliases):
                for other_m, other_aliases in MILLET_GROUPS.items():
                    if other_m != m_key and any(re.search(rf'\b{re.escape(oa)}\b', combined) for oa in other_aliases):
                        cand_has_millet_conflict = True
                        break
            if cand_has_millet_conflict:
                break
        if cand_has_millet_conflict:
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

_inflight: Dict[str, "asyncio.Future"] = {}

def _result_to_cache(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **result,
        "best_image": result["best_image"].model_dump() if result.get("best_image") else None,
        "alternatives": [a.model_dump() for a in result.get("alternatives", [])],
    }

def _result_from_cache(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **data,
        "best_image": ImageResult(**data["best_image"]) if data.get("best_image") else None,
        "alternatives": [ImageResult(**a) for a in data.get("alternatives", [])],
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
    Cached entry point:
    - A SUCCESS result is stored on disk, so the same search (another size of the same product,
      a re-run of the file, a server restart) is answered instantly.
    - Identical searches running at the same moment share one search + OCR pass.
    - NOT_FOUND / ERROR are not cached, so "Retry Not Found" really searches again.
    """
    clean_q = sanitize_search_query(query)
    # The product image (without size) is the same for every size of a product: leave size out of its key
    size_key = variant_label if (is_variant or is_query_multipack(product_name, variant_label)) else ""
    key = hashlib.sha1(json.dumps(
        [base_url.rstrip("/"), clean_q, brand, product_name, size_key, is_variant, safesearch, time_range, language, top_limit]
    ).encode()).hexdigest()

    cached = cache_store.get_search(key)
    if cached is not None:
        return _result_from_cache(cached)

    task = _inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_search_uncached(
            client, base_url, query, brand, product_name, variant_label,
            is_variant, safesearch, time_range, language, top_limit
        ))
        _inflight[key] = task
        task.add_done_callback(lambda _t: _inflight.pop(key, None))
    result = await asyncio.shield(task)

    if result["status"] == "SUCCESS" and result.get("best_image"):
        cache_store.put_search(key, _result_to_cache(result))
    # Each row cycles through its own alternatives list
    return {**result, "alternatives": list(result.get("alternatives", []))}

async def _search_uncached(
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

            # A page URL is not an image: only real image links are candidates
            direct_img = img_src or thumbnail_src
            thumb = thumbnail_src or img_src

            if direct_img:
                if is_blocked_source(direct_img, page_url):
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

        async def inspect(candidate: ImageResult) -> str:
            """
            'exact'    -> passes the strict rules (exact size / clean packshot)
            'fallback' -> right brand + product front, but strict size/clean rule not met
            'reject'   -> wrong product, back panel, multi-pack, unreadable, or other size
            """
            cand_url = candidate.img_src
            title = candidate.title
            cand_context = f"{title} {cand_url} {candidate.url}"

            # Cheap metadata checks first: no download/OCR for obvious rejects
            cand_multi = is_candidate_multipack(title, cand_context, resolution=candidate.resolution)
            if cand_multi != query_is_combo:
                return "reject"
            if is_back_panel_or_nutrition(title, cand_url):
                return "reject"
            meta_ok = is_dynamic_brand_match(cand_context, brand) and \
                verify_product_keyword_match([], title, product_name, brand, cand_context)

            # Run OCR on the full packaging image (thumbnail only if the full image can't be downloaded)
            words = await extract_image_ocr_text(client, cand_url, candidate.thumbnail_src)

            # FRONT-ONLY RULE: the image must be readable, must not be a back panel,
            # and must show the brand or product name printed on it
            if is_back_panel_or_nutrition(title, cand_url, words):
                return "reject"
            if not ocr_shows_front(words, brand, product_name):
                return "reject"

            # Re-verify multi-pack, brand and category with OCR text
            if not query_is_combo and is_candidate_multipack(title, cand_context, words, resolution=candidate.resolution):
                return "reject"
            if not is_dynamic_brand_match(cand_context, brand, words):
                return "reject"
            if not verify_product_keyword_match(words, title, product_name, brand, cand_context):
                return "reject"

            if is_variant and variant_label:
                # WITH SIZE: Must contain exact variant size on pack / title / url
                if meta_ok and verify_size_strict(words, title, variant_label, cand_context):
                    return "exact"
                return "reject" if _ocr_shows_other_size(words, variant_label) else "fallback"
            # WITHOUT SIZE: prefer a pack without a size callout (1kg, 500g, etc.)
            if meta_ok and not has_any_size_callout(words, title, cand_url):
                return "exact"
            return "fallback"

        # Step 2: Strict Vision / PP-OCRv6 Inspection, OCR_BATCH candidates at a time in rank order.
        # Stops at the first batch containing an exact match, so most searches OCR only 4-8 images.
        verdicts: Dict[int, str] = {}
        for start in range(0, len(candidate_items), OCR_BATCH):
            batch = list(range(start, min(start + OCR_BATCH, len(candidate_items))))
            results = await asyncio.gather(*(inspect(candidate_items[i]) for i in batch))
            verdicts.update(zip(batch, results))
            if "exact" in results:
                break

        def first(kind: str) -> int:
            return next((i for i in sorted(verdicts) if verdicts[i] == kind), -1)

        selected_index = first("exact")
        if selected_index == -1:
            selected_index = first("fallback")

        # Alternatives never include rejected images (back panels, wrong products)
        rank = {"exact": 0, "fallback": 1}
        alt_order = sorted(
            (i for i in range(len(candidate_items)) if i != selected_index and verdicts.get(i) != "reject"),
            key=lambda i: (rank.get(verdicts.get(i, ""), 2), i)
        )
        alternatives = [candidate_items[i] for i in alt_order][:MAX_ALTERNATIVES]

        # If still no candidate matches both brand and product, return NOT_FOUND (Zero false positives!)
        if selected_index == -1:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": alternatives,
                "error_message": f"No verified image matching brand '{brand}' and product '{product_name}'"
            }

        return {
            "status": "SUCCESS",
            "best_image": candidate_items[selected_index],
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
