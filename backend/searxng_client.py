import asyncio
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
import math
from typing import Dict, Any, List, Optional, Tuple
import httpx
import numpy as np
from PIL import Image
from models import ImageResult

logger = logging.getLogger("searxng_client")

# Two-stage PP-OCRv6 (fast small model to screen, strict medium model to confirm)
try:
    from ocr_engine import run_ocr_full
    logger.info("PP-OCRv6 two-stage OCR (PaddleOCR ONNX backend) initialized successfully.")
except Exception as e:
    logger.warning(f"PP-OCRv6 failed to initialize: {e}")
    run_ocr_full = None

# OCR text and search results are cached on disk (cache_store) so nothing is done twice
import os
import cache_store
try:
    import vision_engine
except Exception as e:  # torch / open_clip missing: the pipeline still runs on OCR rules alone
    logger.warning(f"Vision engine unavailable: {e}")
    vision_engine = None
VISION_CACHE_VERSION = "v2"
# Calibrated on real candidates (OpenCLIP ViT-B-32): back panel 0.85, 2-pack 0.86, "pouch OR bottle" 0.95,
# crop 0.83, banner 0.58 ("multiple") vs real fronts 0.83-0.94 front_single.
# "lifestyle" is NOT a reject kind: packs with a food photo printed on them score high on it.
VISION_REJECT_KINDS = ["back_panel", "multiple", "banner", "crop"]
VISION_REJECT_PROB = 0.5
# Display strip / hanger card of many sachets: strips 0.99-1.00, real single bags of candy at most 0.56
DISPLAY_REJECT_PROB = 0.8
# Last tiers, used only when no image passes every rule (so no row is left without an image):
#  - NEAREST: right product, other size printed (company doesn't make the asked size) -> closest size wins
#  - RELAXED: OCR couldn't confirm brand/product text, but vision sees one front pack of this product
SOFT_REJECTS = {"not_front", "product_words", "product_not_on_pack", "low_quality", "brand", "vision_match"}
RELAXED_MAX = 10
RELAXED_MIN_FRONT = 0.5
RELAXED_MIN_MATCH = 0.27
RELAXED_MAX_MULTIPLE = 0.3
#  - LOOSE: very last resort (row would be empty): brand named by the listing, one product word matches
LOOSE_REJECTS = SOFT_REJECTS | {"listing_no_size"}
LOOSE_MIN_MATCH = 0.24
# Image <-> "a photo of <product> product package": right product 0.36-0.45, a different product 0.24
VISION_MIN_MATCH = 0.27
# Embedding similarity: same photo 1.0, same product other photo ~0.7, different products 0.42-0.48
SAME_PHOTO_SIMILARITY = 0.92
SAME_PACK_SIMILARITY = 0.6
# Max candidates screened (fast OCR + vision) per search attempt
MAX_SCREENED = 12
# Text rejects worth a second read by the accurate OCR when vision sees one front pack
TEXT_REJECTS = {"not_front", "brand", "product_words", "product_not_on_pack"}
MAX_SECOND_LOOKS = 4
SECOND_LOOK_FRONT = 0.6
# Image engines (a suspended engine answers empty at once). The main engines answer every attempt;
# the extra engines (Yandex: UK / EU Indian grocers, Blinkit, regional brands) get their own attempts
# when the main engines found no exact image, so their results never push good main results out.
IMAGE_ENGINES = [e.strip() for e in os.environ.get("IMAGE_ENGINES", "google images,bing images").split(",") if e.strip()]
EXTRA_IMAGE_ENGINES = [e.strip() for e in os.environ.get("EXTRA_IMAGE_ENGINES", "yandex images").split(",") if e.strip()]

# Images OCR'd at the same time per search (the OCR pool runs them in parallel)
OCR_BATCH = 4
MAX_ALTERNATIVES = 8
# Border-uniformity score (ocr_engine.packshot_quality) at or above which an image counts as a clean packshot.
# Lower scores are lifestyle photos or cropped / half-cut packs: used only when no clean packshot exists.
CLEAN_PACKSHOT_QUALITY = 0.6
# Store image CDNs that only serve plain packshots (some fill the frame edge to edge, which lowers
# the border score even though the photo is clean)
CLEAN_PACKSHOT_HOSTS = ['bbassets.com', 'media-amazon.com', 'grofers.com', 'instamart-media-assets.swiggy.com',
                        'rukminim', 'flixcart.com', 'dmart.in', 'jiomart.com']
# Design portfolios / packaging blogs: mockups and renders, never the exact pick
MOCKUP_HOSTS = ['blogspot', 'behance.net', 'dribbble', 'packagingoftheworld', 'adgully', 'artstation']

def is_clean_packshot(img_url: str, quality: float) -> bool:
    url = (img_url or "").lower()
    if any(h in url for h in MOCKUP_HOSTS):
        return False
    return quality >= CLEAN_PACKSHOT_QUALITY or any(h in _host(url) for h in CLEAN_PACKSHOT_HOSTS)

# Below this border score an image is a lifestyle scene, a banner or a half-cut close-up: never used
# (measured: cropped close-up 0.05, ad banner 0.00, kitchen-table photo 0.11; packshots 0.45-1.0)
MIN_PACKSHOT_QUALITY = 0.25
MIN_IMAGE_SIDE = 250
MIN_FRONT_SINGLE = 0.25       # vision: below this it is an ad, banner or recipe scene, not one front pack          # real pixels: a 220x220 thumbnail is too small and blurry for a product image
# ocr_engine.repeated_pack_score at or below this = the same pack photographed 2-3 times side by side
# (measured: identical 2-pack 0.07, single packs 0.39-0.80)
REPEATED_PACK_MAX = 0.2

# Text found on ads and promo banners, never on a pack front
BANNER_PATTERNS = [
    r'\bswitch\s+to\b', r'\bshop\s+now\b', r'\bbuy\s+now\b', r'\border\s+now\b', r'\bavailable\s+(?:on|at)\b',
    r'\bdownload\s+the\s+app\b', r'\bfree\s+delivery\b', r'\bcash\s+on\s+delivery\b', r'\bflat\s+\d+\s*%\s*off\b',
]

def is_banner_or_composite(words: List[str], brand: str = "") -> bool:
    """Ad banners, and listing images showing alternatives ('pouch OR bottle', 'packaging may vary')."""
    ocr = " ".join(words or []).lower()
    if any(re.search(p, ocr) for p in BANNER_PATTERNS):
        return True
    if re.search(r'packaging\s+m(?:ay|ight)\s+vary', ocr):
        return True
    # A standalone "OR" between two copies of the brand: two different packs offered as alternatives
    main = max((t for t in _brand_tokens(brand) if len(t) >= 4), key=len, default="")
    if any(w.strip() == "or" for w in words or []) and main and sum(1 for w in words if main in w) >= 2:
        return True
    return False

# Part of every search-cache key: bump it whenever the selection rules change, so results chosen
# under older (weaker) rules are never served from the cache again
SEARCH_RULES_VERSION = "rules-21"
# Part of every OCR-cache key: bump when ocr_engine returns new fields
OCR_CACHE_VERSION = "v3"

# Recently downloaded images, so the strict second OCR pass does not download them again
_IMAGE_BYTES: "Dict[str, bytes]" = {}
_IMAGE_BYTES_MAX = 400
# image URL -> the link that actually downloaded (the URL itself, or its search thumbnail if the full link is dead)
_WORKING_SRC: "Dict[str, str]" = {}

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
    return url

def bigbasket_primary(url: str) -> str:
    """
    BigBasket numbers a product's photos _1, _2 ...; _1 is usually the front. Returns the _1 URL to try
    FIRST as an extra candidate, or "" if not a BigBasket secondary photo. Never a blind rewrite: some
    products have no _1 (404) and their front is e.g. _9, so the original stays a candidate too.
    """
    low = (url or "").lower()
    if ('bbassets' in low or 'bigbasket.com/media/uploads/p/' in low) and re.search(r'\d+_[2-9]-', low):
        return re.sub(r'(\d+)_[2-9]-', r'\g<1>_1-', url)
    return ""

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

async def read_image(client: httpx.AsyncClient, img_url: str, fallback_url: str = "",
                     stage: str = "fast", timeout: float = 6.0) -> Dict[str, Any]:
    """
    Downloads the FULL image (falling back to the search thumbnail) and runs PP-OCRv6.
    stage='fast' screens candidates; stage='strict' is the second, more accurate read.
    Returns {"words": [...], "quality": 0..1}; words is [] when the image could not be downloaded.
    Failures are not cached (may be temporary).
    """
    empty = {"words": [], "quality": 0.0, "repeat": 9.0}
    if not img_url or run_ocr_full is None:
        return empty

    key = f"{stage}:{OCR_CACHE_VERSION}:{img_url}"
    cached = cache_store.get_ocr(key)
    if isinstance(cached, dict):
        return cached

    # "src" records which link actually downloaded: the result must only ever contain working links
    working = img_url
    content = _IMAGE_BYTES.get(img_url)
    if content is None:
        content = await _download_image(client, img_url, timeout)
        if content is None and fallback_url and fallback_url != img_url:
            content = await _download_image(client, fallback_url, timeout)
            working = fallback_url
        if content is None:
            return empty
        if len(_IMAGE_BYTES) >= _IMAGE_BYTES_MAX:
            _IMAGE_BYTES.pop(next(iter(_IMAGE_BYTES)))
        _IMAGE_BYTES[img_url] = content
        _WORKING_SRC[img_url] = working
    else:
        working = _WORKING_SRC.get(img_url, img_url)

    try:
        result = await run_ocr_full(content, stage)
    except Exception:
        return empty
    result = {**result, "src": working, "side": image_min_side(img_url)}
    cache_store.put_ocr(key, result)
    return result

async def extract_image_ocr_text(client: httpx.AsyncClient, img_url: str, fallback_url: str = "", timeout: float = 6.0) -> List[str]:
    """Fast-stage OCR text of an image (see read_image)."""
    return (await read_image(client, img_url, fallback_url, "fast", timeout))["words"]

async def image_loads(client: httpx.AsyncClient, url: str, timeout: float = 5.0) -> bool:
    """The link really serves an image (used before swapping in a larger version)."""
    try:
        async with client.stream("GET", url, headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True) as r:
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                return False
            async for chunk in r.aiter_bytes():
                return len(chunk) > 0
    except Exception:
        return False
    return False

async def read_vision(client: httpx.AsyncClient, img_url: str, fallback_url: str, product_text: str) -> Optional[Dict[str, Any]]:
    """OpenCLIP image understanding (photo kind, name match, embedding). None if vision is off/unavailable."""
    if vision_engine is None or not vision_engine.VISION_ENABLED or not img_url:
        return None
    key = f"vision:{VISION_CACHE_VERSION}:{img_url}|{product_text.lower()}"
    cached = cache_store.get_ocr(key)
    if isinstance(cached, dict):
        return cached
    content = _IMAGE_BYTES.get(img_url)
    if content is None:
        content = await _download_image(client, img_url, 6.0)
        if content is None and fallback_url:
            content = await _download_image(client, fallback_url, 6.0)
        if content is None:
            return None
    result = await vision_engine.analyze(content, product_text)
    if result:
        cache_store.put_ocr(key, result)
    return result

# Several SearXNG instances (comma separated, or SEARXNG_URLS env in Docker): searches are spread over them
_rr_counter = [0]

def pick_searxng(base_url: str) -> str:
    env = os.environ.get("SEARXNG_URLS", "").strip()
    urls = [u.strip().rstrip("/") for u in (env or base_url or "").split(",") if u.strip()]
    if not urls:
        return (base_url or "http://localhost:8888").rstrip("/")
    _rr_counter[0] = (_rr_counter[0] + 1) % len(urls)
    return urls[_rr_counter[0]]

# Stores with clean official packshots, searched first ("site:" filter) before the open web
TRUSTED_STORE_SITES = ["bigbasket.com", "amazon.in", "blinkit.com", "flipkart.com", "swiggy.com", "dmart.in"]
TRUSTED_IMAGE_HOSTS = ["bbassets.com", "bigbasket.com", "media-amazon.com", "grofers.com", "blinkit",
                       "rukminim", "flixcart.com", "swiggy.com", "dmart.in"]

def trusted_query(plain_q: str) -> str:
    # Bing (the engine that answers) ignores "site:" and returns unrelated photos for it;
    # shopping words pull the store listings (BigBasket, Amazon, JioMart...) to the top instead
    return f"{plain_q} buy online"

def _brand_tokens(brand: str) -> List[str]:
    # "Cote D'or" -> ['cote', 'dor'] (punctuation inside a word is dropped, as OCR drops it)
    toks = [re.sub(r"[^a-z0-9]", "", t) for t in re.split(r'[\s\-_,()/&.]+', fold(brand or "").lower())]
    return [t for t in toks if t]

def _token_present(token: str, text: str, compact: str) -> bool:
    """Short tokens ('bb', 'mp', '24') must be whole words; longer ones may be glued ('aashirvaadatta')."""
    if len(token) <= 3:
        return bool(re.search(rf'(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])', text))
    return token in compact

def fold(text: str) -> str:
    """Accents off, so OCR 'nestlē' / 'côte d'or' match the Excel 'Nestle' / 'Cote D'or'."""
    if not text:
        return text or ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

# Company-form words: printed tiny or not at all ("Krishna Confectionery" packs say "Krishna")
CORPORATE_WORDS = {
    'confectionery', 'confectionary', 'confections', 'industries', 'industry', 'company', 'co', 'pvt',
    'ltd', 'limited', 'private', 'foods', 'food', 'products', 'enterprises', 'enterprise', 'international',
    'india', 'indian', 'corporation', 'corp', 'inc', 'group', 'agro', 'mills', 'trading', 'traders', 'brands',
}

def _edit1(a: str, b: str) -> bool:
    """a and b differ by at most one inserted, deleted or changed letter."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] if la == lb else a[i:] == b[i + 1:]

def _near_in(token: str, compact: str) -> bool:
    """token appears in compact OCR text with at most one letter wrong."""
    n = len(token)
    for size in (n - 1, n, n + 1):
        for i in range(0, len(compact) - size + 1):
            if _edit1(token, compact[i:i + size]):
                return True
    return False

# Parent companies whose packs print only the product brand ('Perfetti' row -> 'Chupa Chups' pack)
PARENT_BRANDS = {
    "perfetti": ["chupa chups", "mentos", "alpenliebe", "centerfruit", "center fruit", "centre fruit",
                 "center fresh", "centre fresh", "happydent", "big babol", "fruittella", "chloromint", "marbels"],
    "mondelez": ["cadbury", "oreo", "bournvita", "toblerone", "tang"],
    "nestle": ["maggi", "kitkat", "kit kat", "munch", "milkybar", "nescafe", "polo", "everyday", "a+"],
    "ferrero": ["kinder", "nutella", "tic tac", "ferrero rocher", "rocher"],
    "mars": ["snickers", "galaxy", "bounty", "twix", "m&m", "skittles"],
    "itc": ["aashirvaad", "sunfeast", "bingo", "yippee", "candyman", "fabelle", "mint-o", "mint o"],
}

def effective_brand(brand: str, product_name: str) -> str:
    """The brand printed on the pack: the product brand when the Excel names only the parent company."""
    key = re.sub(r'[^a-z0-9]', '', fold(brand or '').lower())
    for parent, subs in PARENT_BRANDS.items():
        if key == parent or key.startswith(parent):
            name = f" {re.sub(r'[^a-z0-9&+]+', ' ', fold(product_name or '').lower())} "
            for sb in sorted(subs, key=len, reverse=True):
                if f" {sb} " in name:
                    return sb.title()
    return brand

def brand_evidence(words: List[str], brand: str) -> str:
    """
    How clearly the brand is printed on the pack:
      'strong' - the whole brand, every brand word, a multi-word part of it ('5star' of 'Cadbury 5 Star'),
                 or a distinctive brand word ('milkybar', 'krishna')
      'weak'   - only a generic main word ('royal' of 'bb Royal')
      ''       - not on the pack
    OCR often misreads the small parent-company line ('padbury'), so a distinctive part is enough.
    """
    if not brand:
        return "strong"
    ocr = fold(" ".join(words or [])).lower()
    compact = re.sub(r'[^a-z0-9]', '', ocr)
    if not compact:
        return ""
    full = re.sub(r'[^a-z0-9]', '', fold(brand).lower())
    if len(full) >= 4 and full in compact:
        return "strong"
    tokens = [t for t in _brand_tokens(fold(brand)) if t not in CORPORATE_WORDS and t not in {'organic'}]
    if tokens and all(_token_present(t, ocr, compact) for t in tokens):
        return "strong"
    # contiguous multi-word parts: "5 star" -> "5star"
    for i in range(len(tokens)):
        for j in range(i + 2, len(tokens) + 1):
            part = "".join(tokens[i:j])
            if len(part) >= 4 and part in compact:
                return "strong"
    distinctive = [t for t in tokens if len(t) >= 5 and t not in GENERIC_BRAND_WORDS]
    if any(t in compact for t in distinctive):
        return "strong"
    # stylised logos: OCR drops or swaps one letter ('udaiyam' for 'Udhaiyam'); long words only
    if any(len(t) >= 6 and _near_in(t, compact) for t in distinctive):
        return "strong"
    long_tokens = [t for t in tokens if len(t) >= 4]
    main = max([t for t in long_tokens if t not in GENERIC_BRAND_WORDS] or long_tokens, key=len, default="")
    if main and main in compact:
        return "weak"
    return ""

_LARGER_IMAGE_RULES = [
    # Amazon: ._SL500_.jpg / ._AC_UL320_.jpg -> original upload
    (re.compile(r'^(https?://m\.media-amazon\.com/images/I/[^/]+?)\._[^/]+_(\.(?:jpe?g|png|webp))$', re.I), r'\1\2'),
    # IndiaMART: -250x250.jpg / -500x500.jpg -> -1000x1000.jpg
    (re.compile(r'^(https?://\d*\.imimg\.com/.+?)-(?:1[0-9]{2}|[2-9][0-9]{2})x(?:1[0-9]{2}|[2-9][0-9]{2})(\.(?:jpe?g|png|webp))$', re.I), r'\1-1000x1000\2'),
    # BigBasket: small / medium -> large
    (re.compile(r'^(https?://(?:www\.)?(?:bigbasket\.com|bbassets\.com)/media/uploads/p/)(?:s|m|mm|ml)(/.+)$', re.I), r'\1l\2'),
    # Flipkart: /image/312/312/ -> /image/832/832/
    (re.compile(r'^(https?://rukminim\d*\.flixcart\.com/image/)\d{2,3}/\d{2,3}(/.+)$', re.I), r'\g<1>832/832\2'),
]

def larger_image_url(url: str) -> str:
    """The same photo at full size on stores that serve thumbnails by URL ('' when no rule applies)."""
    for pat, repl in _LARGER_IMAGE_RULES:
        if pat.match(url or ""):
            big = pat.sub(repl, url)
            return big if big != url else ""
    return ""

def size_distance(words: List[str], title: str, variant_label: str) -> Optional[float]:
    """How far the size shown on the pack / title is from the requested one (log ratio; None if unknown)."""
    targets = _parse_sizes(variant_label)
    if not targets:
        return None
    found = _parse_sizes(" ".join(words or []) + " " + (title or ""))
    best = None
    for dim, val in targets:
        for d2, v2 in found:
            if d2 == dim and val > 0 and v2 > 0:
                dist = abs(math.log(v2 / val))
                best = dist if best is None else min(best, dist)
    return best

def image_min_side(url: str) -> int:
    """Shorter side in real pixels of a downloaded image (0 = unknown)."""
    data = _IMAGE_BYTES.get(url)
    if not data:
        return 0
    try:
        return min(Image.open(io.BytesIO(data)).size)
    except Exception:
        return 0

def resolution_px(resolution: str) -> int:
    """'1000 x 1000' -> 1000 (shorter side); 0 when unknown."""
    m = re.search(r'(\d{2,5})\s*[x×]\s*(\d{2,5})', resolution or "")
    return min(int(m.group(1)), int(m.group(2))) if m else 0

def brand_on_pack(words: List[str], brand: str) -> bool:
    """The brand is printed on the pack (strongly, or at least its main word)."""
    return bool(brand_evidence(words, brand))

def product_token_hits(words: List[str], brand: str, product_name: str) -> Tuple[int, int]:
    """
    (printed, needed): how many distinctive product-name words are PRINTED on the pack.
    Distinctive = not the brand, not generic ('super', 'gold'), not the category ('rice', 'candy').
    A local / English name of the same thing counts ('toor' pack may say 'tur' / 'arhar').
    """
    compact = re.sub(r'[^a-z0-9]', '', fold(" ".join(words or [])).lower())
    brand_toks = set(_brand_tokens(brand))
    tokens = []
    for t in re.split(r'[\s\-_,()/&.]+', fold(product_name).lower()):
        t = re.sub(r'[^a-z0-9]', '', t)
        if len(t) >= 4 and t not in brand_toks and t not in GENERIC_BRAND_WORDS and t not in CATEGORY_WORDS \
                and t not in PACKAGING_WORDS and not re.match(r'^\d', t) and t not in tokens:
            tokens.append(t)
    hits = 0
    for t in tokens:
        aliases = {t}
        for group in GRAIN_ALIAS_GROUPS:
            if t in [re.sub(r'[^a-z0-9]', '', a) for a in group]:
                aliases.update(re.sub(r'[^a-z0-9]', '', a) for a in group)
        if any(a in compact for a in aliases):
            hits += 1
    return hits, len(tokens)

def product_on_pack(words: List[str], brand: str, product_name: str) -> bool:
    """
    The product's own name is PRINTED on the pack (not only in the shop's title): every distinctive
    word for short names, at least 2 of 3 for long ones. Shops mislabel listings: a "Parry's Bullet Kolam"
    listing showing "Parry's Kolam Steam Rice", or "Aashirvaad Besan" showing "Bansi Rava", is wrong.
    Names with no distinctive word (e.g. "Saffola Gold Oil") pass.
    """
    if not words or not product_name:
        return True
    hits, needed = product_token_hits(words, brand, product_name)
    if needed == 0:
        return True
    return hits >= (needed if needed <= 3 else math.ceil(needed * 2 / 3))

def ocr_repeats_multipack(words: List[str], brand: str = "") -> bool:
    """
    Several packs in one image: OCR reads the brand / the same claims once per pack.
    e.g. 3 x 24 Mantra Rice Flour -> 'mantra' x3, 'gluten free' x3, 'truly ghar ka khana' x3.
    A single pack may print its name twice (top and bottom band), so the threshold is 3.
    """
    if not words:
        return False
    lines = [re.sub(r'[^a-z0-9]', '', w) for w in words]
    lines = [l for l in lines if len(l) >= 4]
    main = max((t for t in _brand_tokens(brand) if len(t) >= 4 and t not in GENERIC_BRAND_WORDS), key=len, default="")
    if main and sum(1 for l in lines if main in l) >= 3:
        return True
    # OCR reads each copy slightly differently ("truly ghar ka khana" / "truly char ka khana"),
    # so lines are grouped by similarity, not exact equality
    from difflib import SequenceMatcher
    groups: List[List[str]] = []
    for l in lines:
        for g in groups:
            if l == g[0] or (len(l) >= 8 and SequenceMatcher(None, l, g[0]).ratio() >= 0.8):
                g.append(l)
                break
        else:
            groups.append([l])
    return sum(1 for g in groups if len(g) >= 3) >= 2

# Words that make a DIFFERENT product variant. If the listing / pack has one that the Excel name
# doesn't, it is the wrong product ("Aashirvaad Organic Atta" is not "Aashirvaad Atta").
VARIANT_MODIFIERS = [
    'organic', 'select', 'superior', 'mp', 'sugar release', 'sugar control', 'low gi', 'diet', 'lite',
    'sugar free', 'sugarfree', 'no added sugar', 'low fat', 'fat free', 'low sodium', 'unsalted',
    'double toned', 'toned', 'skimmed', 'full cream',
    'millet', 'millets', 'multigrain', 'multigrains', 'multi grain', 'multi grains',
    'tea bags', 'tea bag', 'premix', 'green tea',
    'gulbarga', 'lachkari', 'surti', 'hmt', 'wada', 'jsr', 'rnr', 'bpt', 'sona masoori', 'ponni',
    'mogra', 'dubar', 'tibar', 'rozana', 'supreme',
]
EDITION_MARKERS = ['heartblush', 'youaremy', 'aremylove', 'valentine', 'limitededition', 'rakhi', 'diwali',
                   'christmas', 'xmas', 'halloween', 'holiedition', 'festiveedition', 'specialedition', 'giftpack']
# Checked in the title only: packs often print these as claims ("rich in protein")
TITLE_ONLY_MODIFIERS = ['high fibre', 'high fiber', 'protein', 'high protein', 'kids', 'baby', 'junior', 'mini', 'jumbo', 'refill', 'travel', 'sample']

# How the grain was processed makes a different product: Steam Rice Kolam is not Raw Rice Kolam
RICE_PROCESS = {"raw": "raw", "steam": "steam", "steamed": "steam", "boiled": "boiled",
                "parboiled": "boiled", "sella": "boiled"}
RICE_WORDS = ("rice", "kolam", "basmati", "sona masoori", "sonamasoori", "ponni", "idli")

def rice_processes(text: str) -> set:
    t = (text or "").lower()
    return {v for k, v in RICE_PROCESS.items() if re.search(rf'(?<![a-z0-9]){k}(?![a-z0-9])', t)}

def has_process_conflict(name: str, cand_text: str) -> bool:
    """Rice only. A name without a process word (plain 'Kolam Rice') accepts raw, not steam / boiled."""
    if not any(w in name for w in RICE_WORDS):
        return False
    want, got = rice_processes(name), rice_processes(cand_text)
    extra = got - want
    return bool(extra) if want else bool(extra - {"raw"})

def has_variant_conflict(title: str, words: List[str], product_name: str, brand: str = "") -> bool:
    name = f"{product_name} {brand}".lower()
    t = (title or "").lower()
    ocr = " ".join(words or []).lower()
    if has_process_conflict(name, f"{t} {ocr}"):
        return True
    if has_pulse_conflict(name, f"{t} {ocr}"):
        return True   # the pack prints Besan / Chana, the row is Toor Dal: shop titles lie
    # Special / seasonal editions are other products unless the Excel name says so
    # (OCR runs script words together: 'heartblush', 'aremy' = "you are my love")
    name_c = re.sub(r'[^a-z0-9]', '', name)
    cand_c = re.sub(r'[^a-z0-9]', '', f"{t} {ocr}")
    if any(m in cand_c and m not in name_c for m in EDITION_MARKERS):
        return True
    for m in VARIANT_MODIFIERS:
        pat = rf'(?<![a-z0-9]){re.escape(m)}(?![a-z0-9])'
        # singular / plural of a word in the name is the same variant ("Multigrain" vs "Multigrains")
        stem = re.sub(r's$', '', m)
        if re.search(pat, name) or re.search(rf'(?<![a-z0-9]){re.escape(stem)}s?(?![a-z0-9])', name):
            continue
        if re.search(pat, t) or re.search(pat, ocr):
            return True
    for m in TITLE_ONLY_MODIFIERS:
        pat = rf'(?<![a-z0-9]){re.escape(m)}(?![a-z0-9])'
        if not re.search(pat, name) and re.search(pat, t):
            return True
    return False

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
    product_tokens = [t for t in re.split(r'[\s\-_,()/&]+', product_name.lower())
                      if len(t) >= 4 and t not in GENERIC_BRAND_WORDS and not re.match(r'^\d', t)
                      and t not in _brand_tokens(brand)]
    # Brand printed on the pack is the strongest front evidence; with only the logo word read
    # (an infographic or banner showing just "ROYAL"), a product word must be printed too
    evidence = brand_evidence(words, brand) if brand else ""
    if evidence == "strong":
        return True
    if evidence == "weak" and (not product_tokens or any(t in compact for t in product_tokens)):
        return True
    # When the Excel name itself starts with the brand ("bb Royal Chakki Atta"), the front MUST show it:
    # infographics and recipe cards ("chakki atta laddoo...") carry the product words but no brand
    if brand and re.sub(r'[^a-z0-9]', '', brand.lower()) in re.sub(r'[^a-z0-9]', '', product_name.lower()):
        return False
    # Sub-brands (product "Maggi Noodles", brand "Nestle"): accept when the product name is printed on the pack
    return any(t in compact for t in product_tokens) if product_tokens else not brand

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

# Grocery staples that are never each other (only checked when the Excel product is one of them)
STAPLE_CATEGORY_WORDS = ['atta', 'rice', 'basmati', 'dal', 'oil', 'ghee', 'salt', 'sugar', 'tea', 'coffee',
                         'besan', 'maida', 'sooji', 'suji', 'poha', 'jaggery']

GRAIN_ALIAS_GROUPS = [
    ['khapli', 'emmer'], ['kuttu', 'buckwheat'], ['singhara', 'singhada', 'water chestnut'],
    ['rajgira', 'amaranth'], ['makka', 'makki', 'maize'], ['jau', 'barley'], ['samak', 'sama'],
    ['chawal', 'rice flour'], ['soya', 'soyabean'], ['multigrain', 'multi grain'], ['7 grain', '7grain'],
    ['ragi', 'nachni', 'finger millet'], ['jowar', 'sorghum'], ['bajra', 'pearl millet'],
    ['toor', 'tur', 'tuvar', 'toovar', 'arhar'], ['chana', 'chickpea', 'bengal gram'], ['moong', 'mung', 'green gram'],
    ['urad', 'black gram'], ['masoor', 'red lentil', 'lentil'], ['rajma', 'kidney bean', 'kidney beans'],
    ['kabuli', 'chickpeas'], ['sooji', 'suji', 'rava', 'semolina'], ['besan', 'gram flour'],
    ['steam', 'steamed'], ['parboiled', 'boiled'],
]

# Which pulse / flour it is. Generic words ('lentil', 'dal', 'gram') are left out on purpose.
PULSE_GROUPS = [
    ['toor', 'tur dal', 'tuvar', 'toovar', 'arhar'], ['chana', 'bengal gram', 'chickpea'], ['moong', 'mung', 'green gram'],
    ['urad', 'black gram'], ['masoor', 'red lentil'], ['rajma', 'kidney bean'], ['besan', 'gram flour'],
    ['sooji', 'suji', 'rava', 'semolina'], ['ragi', 'nachni', 'finger millet'], ['jowar', 'sorghum'],
    ['bajra', 'pearl millet'], ['poha', 'flattened rice'], ['maida'], ['kulthi', 'horse gram', 'kollu'],
    ['matar', 'watana', 'vatana', 'green peas', 'white peas'], ['salt', 'namak'], ['pepper', 'kali mirch'],
    ['til', 'sesame'], ['jaggery', 'gur', 'gud'],
]

def _pulse_groups_in(text: str) -> set:
    # links run words together ('roasted-soojirava'): split on every non-letter AND camel / joined words stay
    t = f" {re.sub(r'[^a-z0-9]+', ' ', fold(text or '').lower())} "
    return {i for i, g in enumerate(PULSE_GROUPS) if any(f" {w} " in t for w in g)}

FILLER_WORDS = {'with', 'and', 'for', 'the', 'from', 'made', 'new', 'pure'}

def _word_hit(t: str, compact: str) -> bool:
    """t is printed / named: also its singular ('multigrains' -> 'multigrain') or a local name ('tur' / 'arhar')."""
    if t in compact or (len(t) > 4 and t.endswith('s') and t[:-1] in compact):
        return True
    for group in GRAIN_ALIAS_GROUPS:
        g = [re.sub(r'[^a-z0-9]', '', a) for a in group]
        if t in g and any(a in compact for a in g):
            return True
    return False

def has_pulse_conflict(name: str, ocr: str) -> bool:
    """The pack PRINTS another pulse / flour and never this one (Fine Besan pack for 'Toor Dal')."""
    want = _pulse_groups_in(name)
    if not want or not ocr:
        return False
    got = _pulse_groups_in(ocr)
    return bool(got - want) and not (got & want)

def has_specialty_flour_conflict(target_prod: str, cand_text: str) -> bool:
    """
    If target product is a staple (e.g. Atta, Besan, Maida, Sooji) without a specialty modifier,
    reject candidates that introduce specialty flour varieties (e.g. Singhara Aata for Plain Atta).
    """
    p_lower = target_prod.lower()
    c_lower = cand_text.lower()
    for spec in SPECIALTY_FLOUR_MODIFIERS:
        if spec in p_lower:
            continue
        # Local and English names of the SAME grain are not a conflict (Khapli atta says "emmer wheat")
        if any(spec in group and any(alias in p_lower for alias in group) for group in GRAIN_ALIAS_GROUPS):
            continue
        if re.search(rf'\b{re.escape(spec)}\b', c_lower):
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
        r'(?:batch|lot)\s*no', r'\bpkd\.?\b', r'\bmfd\.?\b', r'use\s*by', r'\bmrp\b.*incl', r'net\s*(?:weight|wt|qty|quantity)\s*[:.]',
        # Manufacturer / company / price block (front + back shown side by side, e.g. Aashirvaad Besan "95.00")
        r'manuf[a-z]*\s*by', r'marketed\s*by', r'quality\s*produ[a-z]*\s*of', r'\b(?:pvt|private)\.?\s*(?:ltd|limited)\b',
        r'\b(?:group|industries|foods?\s+products?)\b.*\b(?:ltd|limited|group)\b', r'(?<![\d.])\d{2,4}\.00(?![\d])'
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
        r'\bmulti-pack\b',
        r'\bmulti\s+pack\b',
        r'\b[2-9]\s*packs\b',     # OCR reads "5" and "PACKS" as separate lines; they are joined with a space
        r'\b[2-9]\s*[x×]\s+(?:[a-z]+\s+){0,3}\d+(?:\.\d+)?\s*(?:kg|g|gm|ml|l)\b',   # "6x detergent bar 250g"
        r'\b[2-9]\s*[x×]\s*\d+(?:\.\d+)?\s*(?:kg|g|gm|ml|l)\b'
    ]
    # A line that is only a count multiplier ("6x", "x6") is a multipack badge
    if any(re.fullmatch(r'\s*(?:[2-9]\d?\s*[x×]|[x×]\s*[2-9]\d?)\s*', w or "") for w in (words or [])):
        return True
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

def _meta_shows_only_other_size(meta: str, variant_label: str) -> bool:
    """The listing title / link names a size of the same kind (weight / volume), and never the row's size."""
    want = _parse_sizes(variant_label)
    if not want:
        return False
    dims = {d for d, _ in want}
    got = {s for s in _parse_sizes(meta) if s[0] in dims}
    return bool(got) and not (got & want)

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

    # Short brands ('MTR', '9AM', 'bb') must be whole words: as a substring they match random
    # URL hashes (".../216490_99bc311420bb4af..." matched 'bb' and let the US brand "Royal" in)
    if _token_present(clean_b_str, combined, clean_comb_str) if len(clean_b_str) <= 3 else clean_b_str in clean_comb_str:
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
    filler_words = {'organic', 'pure', 'food', 'foods', 'brand', 'brands', 'india', 'international', 'products', 'pvts', 'ltd'} | CORPORATE_WORDS
    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in re.split(r'[\s\-_]+', b_clean)]
    tokens = [t for t in tokens if len(t) >= 2 and t not in filler_words]
    if not tokens:
        return True

    # Single-word brand (e.g. "Aashirvaad", "Patanjali", "Amul", "Fortune", "9AM", "Catch", "Everest")
    if len(tokens) == 1:
        t = tokens[0]
        trans_t = BRAND_TRANSLITERATIONS.get(t)
        return bool(
            _token_present(t, combined, clean_comb_str) or
            (trans_t and _token_present(trans_t, combined, clean_comb_str))
        )

    # Multi-word brand (e.g. "bb Super Saver", "bb Royal", "Tata Sampann", "Tata Salt"):
    matched_count = 0
    for t in tokens:
        if _token_present(t, combined, clean_comb_str):
            matched_count += 1

    match_ratio = matched_count / len(tokens)

    # For 2-word brands (e.g. "Tata Sampann", "bb Royal"): both words must match (100%)
    if len(tokens) == 2:
        return match_ratio >= 1.0
    else:
        # For 3+ words (e.g. 'bb super saver'): at least 67% must match AND primary first token must match
        first_token = tokens[0]
        first_matches = _token_present(first_token, combined, clean_comb_str)
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
    # Negative claims name OTHER products: "0% maida 100% atta" on an atta pack is not maida
    ocr_str = re.sub(r'(?:0\s*%|no|zero|without|free\s+from|non)[\s-]*(maida|sugar|palm\s*oil|oil|msg|preservatives?|gluten|added\s+\w+)', ' ', ocr_str)
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

    # 4.1 Staple category lock: an Atta listing that also shows Rice / Dal / Oil is another product or a
    # combo (bb Royal "Basmati Rice + Chakki Atta value pack"). Flour products naturally say "atta".
    staple_text = f"{ocr_str} {title} {url}".lower().replace("-", " ").replace("_", " ")
    staple_text = re.sub(r'rice\s*bran', ' ', staple_text)   # "rice bran oil" is an oil, not rice
    target_staples = {s for s in STAPLE_CATEGORY_WORDS if re.search(rf'\b{s}\b', p_lower)}
    if 'flour' in p_lower:
        target_staples.add('atta')
    if target_staples:
        for other in STAPLE_CATEGORY_WORDS:
            if other in target_staples or (other == 'rice' and 'basmati' in target_staples):
                continue
            if other == 'basmati' and 'rice' in target_staples:
                continue
            if re.search(rf'\b{other}\b', staple_text):
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
    # In Docker the instances come from SEARXNG_URLS, not the URL typed in the UI
    base_url = pick_searxng(base_url)
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
    top_limit: int = 20,
    deep: bool = False,
    fresh: bool = False,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Cached entry point:
    - fresh=True (the Retry button) skips the cache and searches again; `exclude` lists image links
      that must not be picked again (the image the user rejected).
    - A SUCCESS result is stored on disk, so the same search (another size of the same product,
      a re-run of the file, a server restart) is answered instantly.
    - Identical searches running at the same moment share one search + OCR pass.
    - NOT_FOUND / ERROR are not cached, so "Retry Not Found" really searches again.
    """
    clean_q = sanitize_search_query(query)
    # The product image (without size) is the same for every size of a product: leave size out of its key
    # The product image may not show ANOTHER size than the row's, so the size is part of both keys
    size_key = variant_label
    key = hashlib.sha1(json.dumps(
        [SEARCH_RULES_VERSION, base_url.rstrip("/"), clean_q, brand, product_name, size_key, is_variant, safesearch, time_range, language, top_limit]
    ).encode()).hexdigest()

    if fresh:
        result = await _search_uncached(
            client, base_url, query, brand, product_name, variant_label,
            is_variant, safesearch, time_range, language, top_limit, deep, exclude
        )
        if result["status"] == "SUCCESS" and result.get("best_image"):
            cache_store.put_search(key, _result_to_cache(result))
        return {**result, "alternatives": list(result.get("alternatives", []))}

    cached = cache_store.get_search(key)
    if cached is not None:
        return _result_from_cache(cached)

    task = _inflight.get(key)
    if task is None:
        task = asyncio.ensure_future(_search_uncached(
            client, base_url, query, brand, product_name, variant_label,
            is_variant, safesearch, time_range, language, top_limit, deep
        ))
        _inflight[key] = task
        task.add_done_callback(lambda _t: _inflight.pop(key, None))
    result = await asyncio.shield(task)

    if result["status"] == "SUCCESS" and result.get("best_image"):
        cache_store.put_search(key, _result_to_cache(result))
    # Each row cycles through its own alternatives list
    return {**result, "alternatives": list(result.get("alternatives", []))}

QUERY_SUFFIXES = [
    "single pack front packaging", "combo pack front packaging",
    "front packaging pouch", "front view product shot",
]

def _strip_suffix(query: str) -> str:
    q = query
    for s in QUERY_SUFFIXES:
        q = re.sub(re.escape(s), " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()

_SIZE_IN_URL = [
    re.compile(r"[-_]\d{2,4}x\d{2,4}(?=\.\w+$|$)"),      # product-jpeg-500x500.jpg (IndiaMART, WordPress)
    re.compile(r"/\d{2,4}x\d{2,4}/"),                   # /500x500/ path segment
    re.compile(r"\._[a-z0-9,_]+_(?=\.\w+$)"),            # Amazon ._SL1500_.jpg / ._AC_UL320_.jpg
    re.compile(r"/(?:s|l|m|xl|xxl|thumb|small|medium|large)/(?=[^/]+$)"),
]

def photo_key(url: str) -> str:
    """The same photo served at different sizes (500x500 vs 1000x1000, Amazon size codes) gets one key."""
    if not url:
        return ""
    u = url.lower().split("?")[0].split("#")[0]
    u = re.sub(r"^https?://", "", u)
    for pat in _SIZE_IN_URL:
        u = pat.sub("/" if pat.pattern.startswith("/") else "", u)
    return u

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
    top_limit: int = 20,
    deep: bool = False,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Automatic retry ladder: stops at the first attempt that finds an EXACT image.
      1. trusted stores (site:bigbasket.com OR site:amazon.in ...) + the query as built, merged
      2. the plain query (brand + product [+ size]): different, often better results
      3. page 2 of the plain query
      deep=True (Retry Not Found) adds: page 3, and the size written with a space ("1 kg")
    If no attempt is exact, the best fallback (right brand + product front) from any attempt is used.
    """
    brand, product_name, variant_label = fold(brand), fold(product_name), fold(variant_label)
    brand = effective_brand(brand, product_name)
    clean_q = sanitize_search_query(fold(query))
    plain_q = _strip_suffix(clean_q) or clean_q
    main, extra_eng = IMAGE_ENGINES, EXTRA_IMAGE_ENGINES
    attempts: List[Tuple[List[str], int, List[str]]] = [([trusted_query(plain_q), clean_q], 1, main)]
    if plain_q.lower() != clean_q.lower():
        attempts.append(([plain_q], 1, main))
    if extra_eng:
        attempts.append(([plain_q], 1, extra_eng))   # another index: shops the main engines miss
    attempts.append(([plain_q], 2, main))
    if deep:
        attempts.append(([plain_q], 3, main))
        if extra_eng:
            attempts.append(([plain_q], 2, extra_eng))
        spaced = re.sub(r'(\d)(kg|g|gm|ml|l|ltr)\b', r'\1 \2', plain_q, flags=re.I)
        if spaced != plain_q:
            attempts.append(([spaced], 1, main))
    # Hard rows only (nothing verified yet): simpler wording finds listings the full name misses,
    # and for the variant the size-free search finds the other sizes of this product (nearest size)
    extra: List[Tuple[List[str], int, List[str]]] = []
    for q in simplified_queries(brand, product_name, variant_label if is_variant else ""):
        if q.lower() not in {plain_q.lower(), clean_q.lower()}:
            extra.append(([q], 1, main))
            if extra_eng:
                extra.append(([q], 1, extra_eng))

    seen: set = set(u for u in (exclude or []) if u)
    excluded = frozenset(photo_key(u) for u in (exclude or []) if u)
    fallback_result: Optional[Dict[str, Any]] = None
    last_tier: Dict[str, Dict[str, Any]] = {}
    extra_alts: List[ImageResult] = []
    errors: List[str] = []
    all_reasons: Dict[str, str] = {}
    tried = 0
    for n, (queries, page, engines) in enumerate(attempts + extra):
        if n >= len(attempts) and fallback_result is not None and not (is_variant and variant_label):
            break   # extra attempts: rows with nothing verified, or a variant without its exact size yet
        tried += 1
        res = await _search_once(client, base_url, queries, page, brand, product_name, variant_label,
                                 is_variant, safesearch, time_range, language, top_limit, seen, excluded, engines)
        all_reasons.update(res.get("reject_reasons") or {})
        if res["status"] == "ERROR":
            errors.append(res.get("error_message") or "error")
            continue
        m = res.get("match")
        # Stop retrying on an exact match, or (without size) on a verified clean packshot that merely shows its size
        if res["status"] == "SUCCESS" and (m == "exact" or (not is_variant and res.get("clean"))):
            res["alternatives"] = (res["alternatives"] + extra_alts)[:MAX_ALTERNATIVES]
            return res
        if res["status"] == "SUCCESS" and m in ("nearest", "relaxed", "loose"):
            last_tier.setdefault(m, res)
            extra_alts.extend(res.get("alternatives", []))
        elif res["status"] == "SUCCESS" and fallback_result is None:
            fallback_result = res
        else:
            extra_alts.extend(res.get("alternatives", []))

    # verified fallback > nearest size of this product > relaxed (vision-confirmed) pick
    best = fallback_result or last_tier.get("nearest") or last_tier.get("relaxed") or last_tier.get("loose")
    if best:
        best["alternatives"] = (best["alternatives"] + extra_alts)[:MAX_ALTERNATIVES]
        return best
    if errors and len(errors) == tried:
        return {"status": "ERROR", "best_image": None, "alternatives": [], "error_message": errors[0]}
    return {
        "status": "NOT_FOUND",
        "best_image": None,
        "alternatives": extra_alts[:MAX_ALTERNATIVES],
        "reject_reasons": all_reasons,
        "error_message": f"No front-pack image of '{product_name}' found online"
    }

# Product families: a toffee is never ghee, spices or wipes. The Excel product's family is its LAST
# family word ("Kali Mirch Goli" is a goli/candy, not pepper); a pack showing only another family is wrong.
CATEGORY_GROUPS = {
    "confectionery": ["candy", "candies", "toffee", "toffees", "tofee", "lollipop", "lollipops", "eclairs", "eclair",
                      "chocolate", "chocolates", "goli", "gum", "bubblegum", "mints", "jelly", "gummies", "truffle",
                      "truffles", "bonbon", "wafer", "lolly"],
    "ghee_oil": ["ghee", "vanaspati", "refined oil", "mustard oil", "sunflower oil", "groundnut oil", "cooking oil"],
    "spices": ["spices", "masala", "chilli powder", "chili powder", "haldi", "turmeric powder", "dhaniya powder",
               "coriander powder", "garam masala"],
    "personal_care": ["wipes", "soap", "shampoo", "lotion", "diaper", "diapers", "sanitary", "toothpaste", "detergent",
                      "handwash", "face wash", "facewash"],
    "staples": ["atta", "basmati", "rice", "besan", "maida", "sooji", "suji", "dal", "poha"],
    "beverages": ["tea", "chai"],
    "dairy": ["paneer", "curd", "dahi", "milk powder"],
}

def _family_hits(text: str) -> List[Tuple[int, str]]:
    t = f" {re.sub(r'[^a-z0-9]+', ' ', fold(text or '').lower())} "
    hits = []
    for fam, words in CATEGORY_GROUPS.items():
        for w in words:
            for m in re.finditer(rf' {re.escape(w)} ', t):
                hits.append((m.start(), fam))
    return hits

def product_family(product_name: str) -> str:
    hits = _family_hits(product_name)
    return max(hits)[1] if hits else ""

def shows_other_family(product_name: str, title: str, words: List[str]) -> bool:
    """The pack / listing names another product family and never this one (Krishna Cow Ghee for a toffee)."""
    fam = product_family(product_name)
    if not fam:
        return False
    seen = {f for _, f in _family_hits(f"{title} {' '.join(words or [])}")}
    return bool(seen) and fam not in seen

CATEGORY_WORDS = {
    'candy', 'candies', 'toffee', 'toffees', 'tofee', 'chocolate', 'chocolates', 'choco', 'bar', 'bars', 'sweets',
    'sweet', 'mints', 'mint', 'gum', 'lollipop', 'lollipops', 'cream', 'creamy', 'flavour', 'flavor', 'flavoured',
    'flavored', 'milk', 'dark', 'white', 'biscuit', 'biscuits', 'cookies', 'snack', 'snacks', 'drink', 'juice',
    'powder', 'oil', 'atta', 'rice', 'tea', 'coffee', 'soap', 'shampoo', 'masala', 'mix', 'assorted', 'fruit',
    'fruity', 'filled', 'premium', 'classic', 'delights', 'treats', 'goodies', 'confectionery',
}

PACKAGING_WORDS = {
    'pouch', 'jar', 'packet', 'packets', 'pack', 'pcs', 'pc', 'pieces', 'piece', 'bag', 'box', 'gift', 'tin',
    'bottle', 'can', 'carton', 'combo', 'per', 'units', 'unit', 'refill', 'sachet', 'sachets', 'family', 'value',
}

def simplified_queries(brand: str, product_name: str, size: str = "") -> List[str]:
    """
    Plainer searches for hard rows:
      'Krishna Confectionery Krishna Pop Mix Lollipop Jar' -> 'Krishna Pop Mix Lollipop' (+ size for the variant)
      and for the variant the same without size (finds other sizes -> nearest size).
    """
    name = re.sub(r'\(.*?\)', ' ', product_name or "")
    name = re.sub(r',.*$', ' ', name)   # "Parle Poppins ..., 100 g" -> drop trailing size/notes
    words = [w for w in re.split(r'\s+', name) if w]
    brand_toks = set(_brand_tokens(brand))
    corp = {t for t in brand_toks if t in CORPORATE_WORDS}
    core = [w for w in words if re.sub(r'[^a-z0-9]', '', w.lower()) not in PACKAGING_WORDS | corp
            and not _parse_sizes(w)]
    # drop a repeated brand word ("Krishna Confectionery Krishna ...")
    dedup: List[str] = []
    for w in core:
        if not dedup or w.lower() != dedup[-1].lower():
            if w.lower() not in [d.lower() for d in dedup] or w.lower() not in brand_toks:
                dedup.append(w)
    base = " ".join(dedup).strip()
    if brand and not set(_brand_tokens(brand)) - CORPORATE_WORDS <= set(_brand_tokens(base)):
        base = f"{fold(brand)} {base}".strip()
    out: List[str] = []
    if base:
        out.append(f"{base} {size}".strip() if size else base)
        if size:
            out.append(base)
    return out

async def _fetch_image_results(client: httpx.AsyncClient, base_url: str, q: str, page: int,
                               language: str, time_range: str, safesearch: int,
                               engines: Optional[List[str]] = None) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    One SearXNG image query over every working image engine at once (results | None, error):
    Google (when SearXNG is not blocked by it), Bing and Yandex. Yandex indexes many shops Bing misses
    (UK / EU Indian grocers, Blinkit, regional brands), so hard rows get far more candidates.
    Results of the engines are interleaved, so one engine's junk never pushes out the other's hits.
    """
    search_url = f"{pick_searxng(base_url)}/search"
    params = {
        "q": q,
        "pageno": page,
        "language": language,
        "time_range": time_range,
        "safesearch": safesearch,
        "format": "json"
    }

    async def one(engine: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        try:
            resp = await client.get(search_url, params={**params, "engines": engine})
            if resp.status_code == 200:
                return resp.json().get("results", []), ""
            return None, f"HTTP {resp.status_code}: {resp.text[:120]}"
        except httpx.TimeoutException:
            return None, "SearXNG request timed out"
        except Exception as e:
            return None, str(e) or type(e).__name__

    got = await asyncio.gather(*(one(e) for e in (engines or IMAGE_ENGINES)))
    lists = [r for r, _ in got if r]
    if not lists:
        errs = [e for _, e in got if e]
        return (None, errs[0]) if errs and len(errs) == len(got) else ([], "")
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for k in range(max(len(l) for l in lists)):
        for l in lists:
            if k < len(l):
                src = l[k].get("img_src") or ""
                if src and src not in seen:
                    seen.add(src)
                    merged.append(l[k])
    return merged, ""

async def _search_once(
    client: httpx.AsyncClient,
    base_url: str,
    queries: List[str],
    page: int,
    brand: str,
    product_name: str,
    variant_label: str,
    is_variant: bool,
    safesearch: int,
    time_range: str,
    language: str,
    top_limit: int,
    seen: set,
    excluded: frozenset = frozenset(),
    engines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search SearXNG + two-stage PP-OCRv6 Strict Inspection Pipeline:
    - Fetches candidate images from SearXNG (Google Images, Bing fallback).
    - Stage 1 (fast OCR) screens candidates in rank order; Stage 2 (strict OCR, medium model)
      re-reads the image before it is accepted. Both reads must pass every rule:
      * front of pack only: readable, not a back panel / nutrition table, brand (or product) printed on it
      * one single pack (no combos, no 2-3 packs in one image) unless a combo is asked
      * right brand, right product, no extra variant words (Organic, Select, Lite...)
      * WITH SIZE: exact size; WITHOUT SIZE: clean packshot without a size badge preferred
      * clean e-commerce packshot preferred over lifestyle photos and cropped / half-cut packs
    Returns status SUCCESS / NOT_FOUND / ERROR and match 'exact' / 'fallback'.
    Candidates already inspected by an earlier attempt (`seen`) are skipped.
    """
    try:
        # All queries of this attempt run in parallel; results are merged in query order
        # (trusted-store results first), duplicates removed
        fetched = await asyncio.gather(*(
            _fetch_image_results(client, base_url, q, page, language, time_range, safesearch, engines) for q in queries
        ))
        raw_results: List[Dict[str, Any]] = []
        merged_src: set = set()
        errors = [err for res, err in fetched if res is None]
        for res, _ in fetched:
            for item in res or []:
                src = item.get("img_src") or item.get("thumbnail_src")
                if src and src not in merged_src:
                    merged_src.add(src)
                    raw_results.append(item)
        if not raw_results and len(errors) == len(queries):
            return {
                "status": "ERROR",
                "best_image": None,
                "alternatives": [],
                "error_message": errors[0]
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

            if direct_img and direct_img not in seen:
                # The image the user rejected may come back under its thumbnail link
                if excluded and (photo_key(direct_img) in excluded or photo_key(thumbnail_src) in excluded):
                    continue
                if is_blocked_source(direct_img, page_url):
                    continue
                seen.add(direct_img)
                # BigBasket secondary photo: try its _1 (usually the front) first; a dead link fails verification
                primary = bigbasket_primary(direct_img)
                if primary and primary not in seen:
                    seen.add(primary)
                    candidate_items.append(ImageResult(title=title, img_src=primary, thumbnail_src="",
                                                       url=page_url, resolution="", engine=engine))
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

        reject_reasons: Dict[str, str] = {}

        def rej(url: str, why: str) -> str:
            """Records which rule rejected a candidate (debugging / 'why not found')."""
            reject_reasons.setdefault(url, why)
            return "reject"

        def vision_says_single(url: str) -> bool:
            """Vision is sure it is one pack (a bag's repeating print pattern is not several packs)."""
            k = (vision_map.get(url) or {}).get("kinds", {})
            return k.get("front_single", 0.0) >= 0.8 and k.get("multiple", 1.0) < 0.1

        def listing_evidence(candidate: ImageResult, words: List[str], meta_ok: bool) -> bool:
            """A generic brand bag (prints no product name at all) on the store's listing of this product.
            A bag that prints SOME other name ('Kolam Steam Rice' for 'Bullet Kolam') is another product."""
            if not (meta_ok and words and is_trusted(candidate) and brand_evidence(words, brand)):
                return False
            if contains_conflicting_brand(" ".join(words), brand):
                return False
            return product_token_hits(words, brand, product_name)[0] == 0

        def judge(candidate: ImageResult, read: Dict[str, Any], meta_ok: bool) -> str:
            """Applies every rule to one OCR read: 'exact', 'fallback' or 'reject'."""
            words = [fold(w) for w in read["words"]]
            quality = read.get("quality", 0.0)
            cand_url = candidate.img_src
            title = fold(candidate.title)
            cand_context = f"{title} {cand_url} {candidate.url}"

            # FRONT-ONLY RULE: the image must be readable, must not be a back panel,
            # and must show the brand or product name printed on it
            if is_back_panel_or_nutrition(title, cand_url, words):
                return rej(cand_url, "back_panel")
            # The store's own listing of THIS product (BigBasket / Amazon ... page named after it) whose
            # photo shows this brand: stores often use one generic bag per brand ("bb Royal ... Net Qty 10 kg")
            listing = listing_evidence(candidate, words, meta_ok)
            if not ocr_shows_front(words, brand, product_name) and not listing:
                return rej(cand_url, "not_front")

            # PROPER PHOTO: no lifestyle scenes, half-cut close-ups, ad banners or "pouch OR bottle" composites
            if quality < MIN_PACKSHOT_QUALITY and not any(h in _host(cand_url) for h in CLEAN_PACKSHOT_HOSTS):
                return rej(cand_url, "low_quality")
            nums = [int(n) for n in re.findall(r'\d+', candidate.resolution or "")[:2]]
            side = read.get("side") or image_min_side(cand_url) or (min(nums) if len(nums) == 2 else 0)
            if 0 < side < MIN_IMAGE_SIDE:
                return rej(cand_url, "low_quality")   # tiny thumbnail
            if is_banner_or_composite(words, brand):
                return rej(cand_url, "banner")

            # ONE PACK ONLY: combos, "pack of 2", and 2-3 packs photographed together
            if not query_is_combo and (
                is_candidate_multipack(title, cand_context, words, resolution=candidate.resolution)
                or ocr_repeats_multipack(words, brand)
                or (read.get("repeat", 9.0) <= REPEATED_PACK_MAX and not vision_says_single(cand_url))
            ):
                return rej(cand_url, "multipack")

            # Right brand, right product, no extra variant (Organic / Select / Lite ...)
            if not is_dynamic_brand_match(cand_context, brand, words):
                return rej(cand_url, "brand")
            name_ok = verify_product_keyword_match(words, title, product_name, brand, cand_context)
            if not name_ok and not listing:
                return rej(cand_url, "product_words")
            if has_variant_conflict(title, words, product_name, brand):
                return rej(cand_url, "variant_conflict")
            if shows_other_family(product_name, title, words):
                return rej(cand_url, "other_family")
            # Mislabelled listings: the product's own name must be printed on the pack
            listing_only = not name_ok or (brand_on_pack(words, brand) and not product_on_pack(words, brand, product_name))
            if listing_only and not listing:
                return rej(cand_url, "product_not_on_pack")

            clean = is_clean_packshot(cand_url, quality)
            if is_variant and variant_label:
                # WITH SIZE: Must contain exact variant size on pack / title / url
                if _ocr_shows_other_size(words, variant_label):
                    return rej(cand_url, "other_size")
                if listing_only:
                    # Generic bag: only the store page names the product, so the bag itself must print
                    # this exact size AND the page must be the listing of this size
                    on_bag = bool(_parse_sizes(" ".join(words)) & _parse_sizes(variant_label))
                    if on_bag and verify_size_strict([], title, variant_label, cand_context):
                        return "exact"
                    return rej(cand_url, "listing_no_size")
                if meta_ok and clean and verify_size_strict(words, title, variant_label, cand_context):
                    return "exact"
                return "fallback"
            # WITHOUT SIZE: a pack (or caption) printing another size ("Moong Dhall 100 g" for the 1 kg row) is wrong
            if variant_label and _ocr_shows_other_size(words, variant_label):
                return rej(cand_url, "product_other_size")
            if variant_label and _meta_shows_only_other_size(f"{title} {cand_url}", variant_label):
                # tiny print OCR misses ('5 kg' in a corner): the listing is the 5 kg pack, not this 1 kg row
                return rej(cand_url, "product_other_size")
            if listing_only:
                return "fallback"   # a pack with the product name printed on it is preferred
            # WITHOUT SIZE: prefer a clean pack without a size callout (1kg, 500g, etc.)
            if meta_ok and clean and not has_any_size_callout(words, title, cand_url):
                return "exact"
            return "fallback"

        meta_ok_map: Dict[str, bool] = {}
        # What the vision model compares each image with ("Aashirvaad Bansi Rava")
        product_text = product_name if brand.lower() in product_name.lower() else f"{brand} {product_name}".strip()

        async def screen(candidate: ImageResult) -> str:
            """
            Metadata checks + Stage 1 (fast OCR):
            'exact'    -> passes the strict rules (pending Stage 2 confirmation)
            'fallback' -> right brand + product front, but a strict rule (size / clean packshot) not met
            'reject'   -> wrong product, back panel, multi-pack, unreadable, or other size
            """
            cand_url = candidate.img_src
            title = fold(candidate.title)
            cand_context = f"{title} {cand_url} {candidate.url}"

            # Cheap metadata checks first: no download/OCR for obvious rejects
            if any(h in f"{cand_url} {candidate.url}".lower() for h in MOCKUP_HOSTS):
                return rej(cand_url, "mockup")   # design mockups / renders, not real product photos
            cand_multi = is_candidate_multipack(title, cand_context, resolution=candidate.resolution)
            if cand_multi != query_is_combo:
                return rej(cand_url, "meta_multipack")
            if is_back_panel_or_nutrition(title, cand_url):
                return rej(cand_url, "meta_back_panel")
            if has_variant_conflict(title, [], product_name, brand):
                return rej(cand_url, "meta_variant_conflict")
            meta_ok = is_dynamic_brand_match(cand_context, brand) and \
                verify_product_keyword_match([], title, product_name, brand, cand_context)
            meta_ok_map[cand_url] = meta_ok

            # Stage 1: fast OCR on the full packaging image (thumbnail only if it can't be downloaded)
            first = await read_image(client, cand_url, candidate.thumbnail_src, "fast")
            clean_shots[cand_url] = is_clean_packshot(cand_url, first["quality"])
            working_src[cand_url] = first.get("src") or ""
            if first.get("repeat", 9.0) <= REPEATED_PACK_MAX and first["words"]:
                # repeated texture: one bag with a pattern, or several packs? vision decides
                vision_map[cand_url] = await read_vision(client, cand_url, candidate.thumbnail_src, product_text)
            verdict = judge(candidate, first, meta_ok)
            if verdict == "reject" and reject_reasons.get(cand_url) in TEXT_REJECTS and second_looks[0] < MAX_SECOND_LOOKS:
                # The fast model garbles stylised logos ('udhhihdddnm' for UDHAIYAM). When vision sees one
                # front pack, the accurate model reads it again and the rules judge what it read.
                vis = vision_map.get(cand_url) or await read_vision(client, cand_url, candidate.thumbnail_src, product_text)
                vision_map[cand_url] = vis
                if vis and vis.get("kinds", {}).get("front_single", 0.0) >= SECOND_LOOK_FRONT:
                    second_looks[0] += 1
                    second = await read_image(client, cand_url, candidate.thumbnail_src, "strict")
                    if second["words"]:
                        reason = reject_reasons.pop(cand_url, None)
                        merged = {**first, "words": list(first["words"]) + list(second["words"]),
                                  "repeat": min(first.get("repeat", 9.0), second.get("repeat", 9.0))
                                  if second.get("repeat") is not None else first.get("repeat", 9.0)}
                        verdict = judge(candidate, merged, meta_ok)
                        if verdict == "reject" and cand_url not in reject_reasons:
                            reject_reasons[cand_url] = reason or "reject"
            if verdict == "reject":
                return "reject"

            # Vision (OpenCLIP): what kind of photo is it, and is it this product at all?
            vis = vision_map.get(cand_url) or await read_vision(client, cand_url, candidate.thumbnail_src, product_text)
            vision_map[cand_url] = vis
            if vis:
                kinds = vis.get("kinds", {})
                bad_kinds = [k for k in VISION_REJECT_KINDS if not (k == "multiple" and query_is_combo)]
                worst = max(bad_kinds, key=lambda k: kinds.get(k, 0.0))
                if kinds.get(worst, 0.0) >= VISION_REJECT_PROB and kinds.get(worst, 0.0) > kinds.get("front_single", 0.0):
                    return rej(cand_url, "vision_kind")
                if kinds and kinds.get("front_single", 0.0) < MIN_FRONT_SINGLE:
                    return rej(cand_url, "vision_not_front")   # ad / banner / recipe scene, not a packshot
                if not query_is_combo and kinds.get("multiple", 0.0) >= RELAXED_MAX_MULTIPLE:
                    return rej(cand_url, "vision_multiple")   # fanned / stacked bars: several packs
                if kinds.get("display", 0.0) >= DISPLAY_REJECT_PROB:
                    return rej(cand_url, "vision_display")   # counter display strip of many small packets, not one pack
                if vis.get("match") is not None and vis["match"] < VISION_MIN_MATCH                         and not listing_evidence(candidate, [fold(w) for w in first["words"]], meta_ok):
                    # (a brand's generic bag on the store's own listing of this product scores low on
                    #  "photo of <product>": the bag names no product, the listing does)
                    return rej(cand_url, "vision_match")   # looks like a different product altogether
            return verdict

        confirmed: Dict[int, str] = {}

        async def confirm(i: int) -> str:
            """
            Stage 2: strict OCR (more accurate model) must agree before an image is accepted.
            Text either model read counts as evidence (size, brand); any rule the strict read breaks rejects.
            Run only for the image about to be picked, so most searches need one strict OCR.
            """
            if i in confirmed:
                return confirmed[i]
            candidate = candidate_items[i]
            verdict = verdicts[i]
            second = await read_image(client, candidate.img_src, candidate.thumbnail_src, "strict")
            if second["words"]:
                strict_verdict = judge(candidate, second, meta_ok_map.get(candidate.img_src, False))
                if strict_verdict == "reject":
                    verdict = "reject"
                elif verdict == "fallback" and strict_verdict == "exact":
                    verdict = "exact"
            confirmed[i] = verdicts[i] = verdict
            return verdict

        def is_trusted(c: ImageResult) -> bool:
            return any(h in f"{c.img_src} {c.url}".lower() for h in TRUSTED_IMAGE_HOSTS)

        def consensus(i: int) -> int:
            """How many OTHER verified candidates are the same photo (official packshot reused by stores)."""
            emb = (vision_map.get(candidate_items[i].img_src) or {}).get("embedding")
            if not emb or vision_engine is None:
                return 0
            n = 0
            for j, v in verdicts.items():
                if j == i or v == "reject":
                    continue
                other = (vision_map.get(candidate_items[j].img_src) or {}).get("embedding")
                sim = vision_engine.similarity(emb, other)
                if sim is not None and sim >= SAME_PHOTO_SIMILARITY:
                    n += 1
            return n

        def score(i: int) -> float:
            """Ranks verified candidates: exact rules > vision front/name match > same photo on several stores > store > rank."""
            c = candidate_items[i]
            vis = vision_map.get(c.img_src)
            s = (100.0 if verdicts[i] == "exact" else 0.0) + max(0, 20 - i)
            s += 15.0 if clean_shots.get(c.img_src) else 0.0
            s += 20.0 if is_trusted(c) else 0.0
            s += min(10.0, resolution_px(c.resolution) / 100.0)   # bigger, sharper photo first
            if vis:
                s += 40.0 * vis.get("kinds", {}).get("front_single", 0.0)
                s -= 20.0 * vis.get("kinds", {}).get("lifestyle", 0.0)
                if vis.get("match") is not None:
                    s += 200.0 * max(0.0, vis["match"] - 0.25)
                s += 25.0 * min(consensus(i), 3)
            return s

        def confident(i: int) -> bool:
            """Top result that needs no comparison: exact, clean, from a store, and vision agrees."""
            c = candidate_items[i]
            vis = vision_map.get(c.img_src)
            vision_ok = vis is None or (vis.get("kinds", {}).get("front_single", 0) >= 0.6 and (vis.get("match") or 0) >= 0.33)
            return verdicts.get(i) == "exact" and clean_shots.get(c.img_src, False) and is_trusted(c) and vision_ok

        relaxed_tokens = [t for t in (re.sub(r'[^a-z0-9]', '', w) for w in re.split(r'[\s\-_,()/&.]+', product_name.lower()))
                          if len(t) >= 3 and t not in set(_brand_tokens(brand)) and t not in GENERIC_BRAND_WORDS
                          and t not in FILLER_WORDS
                          and t not in PACKAGING_WORDS and t not in CORPORATE_WORDS and not re.match(r'^\d', t)]
        # category words name the kind of product, not which one
        category_tokens = [t for t in relaxed_tokens if t in CATEGORY_WORDS]
        relaxed_tokens = [t for t in relaxed_tokens if t not in CATEGORY_WORDS]
        product_word_share: Dict[int, float] = {}

        async def _relaxed_ok(i: int, loose: bool = False) -> bool:
            """
            A last-tier candidate must still be ONE front pack (vision) and not another brand.
            loose=True (very last resort, so a row is never left empty): the brand may be named by the
            listing instead of read on the pack, and one distinctive product word is enough.
            """
            c = candidate_items[i]
            if c.img_src not in vision_map:
                vision_map[c.img_src] = await read_vision(client, c.img_src, c.thumbnail_src, product_text)
            vis = vision_map.get(c.img_src)
            if not vis:
                return False
            kinds = vis.get("kinds", {})
            bad = [k for k in VISION_REJECT_KINDS if not (k == "multiple" and query_is_combo)]
            if any(kinds.get(k, 0.0) >= VISION_REJECT_PROB for k in bad) or kinds.get("display", 0.0) >= DISPLAY_REJECT_PROB:
                return False
            if kinds.get("front_single", 0.0) < RELAXED_MIN_FRONT or                     (vis.get("match") or 0.0) < (LOOSE_MIN_MATCH if loose else RELAXED_MIN_MATCH):
                return False
            if kinds.get("multiple", 0.0) >= RELAXED_MAX_MULTIPLE and not query_is_combo:
                return False
            read = await read_image(client, c.img_src, c.thumbnail_src, "fast")   # cached
            words = [fold(w) for w in read["words"]]
            if contains_conflicting_brand(f"{fold(c.title)} {' '.join(words)}", brand):
                return False
            # a stack of the same bar: the same name read 3+ times ('5star3d' x3)
            lines = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in words]
            if not query_is_combo and (is_candidate_multipack(c.title, f"{c.title} {c.img_src} {c.url}", words)
                                       or any(lines.count(l) >= 3 for l in set(lines) if len(l) >= 4)):
                return False
            if shows_other_family(product_name, fold(c.title), words):
                return False
            if not is_variant and variant_label and _ocr_shows_other_size(words, variant_label):
                return False   # product image never shows another size
            if has_variant_conflict(f"{fold(c.title)} {c.img_src} {c.url}", words, product_name, brand):
                return False   # Raw Rice Kolam is not Steam Rice Kolam
            # It must be THIS brand PRINTED ON THE PACK: shop names and listing titles lie
            # ('saikrishnastore' is not Krishna Confectionery; a Perfetti listing shows Centerfruit for Alpenliebe)
            context = f"{fold(c.title)} {c.img_src} {c.url}"
            if not brand_evidence(words, brand):
                if not (loose and is_dynamic_brand_match(context, brand)):
                    return False
            # ...and this product: a distinctive product word ('nariyal', 'halloween', 'cola'), not just the
            # category ('candy', 'toffee'). Krishna Rich Butter is not Krishna Nariyal Creamy Candy.
            compact = re.sub(r'[^a-z0-9]', '', f"{' '.join(words)} {context}".lower())
            needed = relaxed_tokens or category_tokens
            if needed:
                hits = sum(1 for t in needed if _word_hit(t, compact))
                # every distinctive word ('Black Salt' needs 'salt', 'Honey Cinnamon' needs 'cinnamon');
                # very long names: 3 of 4. Loose differs from relaxed only in where the brand is shown.
                if hits < (len(needed) if len(needed) <= 3 else math.ceil(len(needed) * 3 / 4)):
                    return False
                product_word_share[i] = hits / len(needed)
            return True

        def _relaxed_score(i: int) -> float:
            c = candidate_items[i]
            bonus = 50.0 if is_dynamic_brand_match(f"{fold(c.title)} {c.img_src} {c.url}", brand) else 0.0
            return score(i) + bonus + 80.0 * product_word_share.get(i, 0.0)

        async def _last_tiers() -> Tuple[int, str]:
            idx_by_url = {c.img_src: i for i, c in enumerate(candidate_items)}
            if is_variant and variant_label:
                near = sorted(idx_by_url[u] for u, w in reject_reasons.items() if w == "other_size" and u in idx_by_url)[:RELAXED_MAX]
                oks = await asyncio.gather(*(_relaxed_ok(i) for i in near))
                ranked = []
                for i, ok in zip(near, oks):
                    if not ok:
                        continue
                    c = candidate_items[i]
                    read = await read_image(client, c.img_src, c.thumbnail_src, "fast")
                    d = size_distance([fold(w) for w in read["words"]], fold(c.title), variant_label)
                    if d is not None:
                        ranked.append((d, -_relaxed_score(i), i))
                if ranked:
                    return min(ranked)[2], "nearest"
            soft = sorted(idx_by_url[u] for u, w in reject_reasons.items() if w in SOFT_REJECTS and u in idx_by_url)[:RELAXED_MAX]
            oks = await asyncio.gather(*(_relaxed_ok(i) for i in soft))
            good = [i for i, ok in zip(soft, oks) if ok]
            if good:
                return max(good, key=_relaxed_score), "relaxed"
            # Very last resort: this brand's front pack named by the listing, one product word matching
            loose = sorted(idx_by_url[u] for u, w in reject_reasons.items() if w in LOOSE_REJECTS and u in idx_by_url)[:RELAXED_MAX]
            oks = await asyncio.gather(*(_relaxed_ok(i, loose=True) for i in loose))
            good = [i for i, ok in zip(loose, oks) if ok]
            if good:
                return max(good, key=_relaxed_score), "loose"
            return -1, ""

        # Step 2: Screening in rank order: the top image alone first, then OCR_BATCH in parallel.
        # Screening stops once there are enough verified candidates to compare (or an obviously
        # perfect top result); then the best-scoring candidate must also pass the strict OCR.
        verdicts: Dict[int, str] = {}
        clean_shots: Dict[str, bool] = {}
        vision_map: Dict[str, Any] = {}
        working_src: Dict[str, str] = {}
        second_looks = [0]
        selected_index = -1
        match = "exact"
        start = 0
        while start < len(candidate_items) and start < MAX_SCREENED:
            size = 1 if start == 0 else OCR_BATCH
            batch = list(range(start, min(start + size, len(candidate_items))))
            results = await asyncio.gather(*(screen(candidate_items[i]) for i in batch))
            verdicts.update(zip(batch, results))
            exact_found = [i for i, v in verdicts.items() if v == "exact"]
            if start == 0 and exact_found and confident(exact_found[0]):
                break
            if len(exact_found) >= 2:
                break
            start += size

        for kind in ("exact", "fallback"):
            for i in sorted((i for i, v in verdicts.items() if v == kind), key=score, reverse=True):
                v = await confirm(i)
                if v == "exact" or (kind == "fallback" and v == "fallback"):
                    selected_index, match = i, v
                    break
            if selected_index != -1:
                break

        # Alternatives never include rejected images (back panels, multi-packs, wrong products);
        # verified ones come first, best score first
        rank = {"exact": 0, "fallback": 1}
        alt_order = sorted(
            (i for i in range(len(candidate_items)) if i != selected_index and verdicts.get(i) != "reject"),
            key=lambda i: (rank.get(verdicts.get(i, ""), 2), -score(i) if i in verdicts else i)
        )
        # Never offer a dead link: unscreened candidates (e.g. a BigBasket _1 guessed from a _2 photo) are checked
        alt_pool = alt_order[:MAX_ALTERNATIVES * 2]
        unchecked = [i for i in alt_pool if not working_src.get(candidate_items[i].img_src)]
        loads = await asyncio.gather(*(image_loads(client, candidate_items[i].img_src, 4.0) for i in unchecked))
        dead = {i for i, ok in zip(unchecked, loads) if not ok}
        alt_order = [i for i in alt_pool if i not in dead]
        alternatives = [candidate_items[i] for i in alt_order][:MAX_ALTERNATIVES]
        alt_embeddings = {
            candidate_items[i].img_src: (vision_map.get(candidate_items[i].img_src) or {}).get("embedding")
            for i in alt_order[:MAX_ALTERNATIVES] if (vision_map.get(candidate_items[i].img_src) or {}).get("embedding")
        }

        if selected_index == -1:
            selected_index, match = await _last_tiers()

        # Nothing usable at all (no candidate is even a front pack of a similar product)
        if selected_index == -1:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": alternatives,
                "reject_reasons": reject_reasons,
                "error_message": f"No verified image matching brand '{brand}' and product '{product_name}'"
            }

        chosen = candidate_items[selected_index]
        vis = vision_map.get(chosen.img_src)
        clean = clean_shots.get(chosen.img_src, False)
        # VERIFIED LINK ONLY: if the full-size link was dead and only the thumbnail downloaded, use the thumbnail
        src = working_src.get(chosen.img_src)
        if src and src != chosen.img_src:
            chosen = chosen.model_copy(update={"img_src": src, "thumbnail_src": src})
        # Bigger version of the same photo when the store serves a thumbnail by URL (verified to load)
        big = larger_image_url(chosen.img_src)
        if big and await image_loads(client, big):
            chosen = chosen.model_copy(update={"img_src": big})
        vision_ok = vis is None or (vis.get("kinds", {}).get("front_single", 0) >= 0.5 and (vis.get("match") or 0) >= 0.3)
        if match in ("nearest", "relaxed", "loose"):
            confidence = "LOW"
        elif match == "exact" and clean and vision_ok:
            confidence = "HIGH"
        elif match == "exact" or (clean and vision_ok):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        return {
            "status": "SUCCESS",
            "best_image": chosen,
            "alternatives": alternatives,
            "match": match,
            "clean": clean,
            "confidence": confidence,
            "embedding": (vis or {}).get("embedding"),
            "alt_embeddings": alt_embeddings,
            "reject_reasons": reject_reasons,
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
