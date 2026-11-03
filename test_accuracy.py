import httpx
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def sanitize_search_query(text: str) -> str:
    # Remove symbols that confuse search engines like %, #, *, quotes, brackets
    s = re.sub(r'[%#*+=\\\[\]{}()<>"~`]', ' ', text)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def score_and_rank_images(
    results: list,
    brand: str,
    product_name: str,
    variant_label: str = "",
    is_variant: bool = False
) -> list:
    """
    Intelligently scores and ranks images to ensure ZERO competitor brand leaks
    and maximum accuracy to the query.
    """
    if not results:
        return []

    brand_clean = brand.strip().lower()
    brand_tokens = [t for t in re.split(r'[\s\-_]+', brand_clean) if len(t) >= 2]
    
    stop_words = {'100', 'pure', 'pack', 'pouch', 'packet', 'the', 'and', 'for', 'with', 'original', 'best', 'organic'}
    prod_clean = product_name.strip().lower()
    prod_tokens = [t for t in re.split(r'[\s\-_]+', prod_clean) if len(t) >= 3 and t not in stop_words]

    scored = []
    has_any_brand_match = False

    for item in results:
        title = (item.get("title") or "").lower()
        img_url = (item.get("img_src") or "").lower()
        page_url = (item.get("url") or "").lower()
        combined = f"{title} {img_url} {page_url}"

        score = 0.0

        # 1. BRAND MATCHING
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
                # Brand is specified, but this image does NOT contain the brand!
                score -= 80.0

        # 2. PRODUCT NAME TOKENS
        for pt in prod_tokens:
            if pt in brand_tokens:
                continue
            if pt in combined:
                # Core product keywords (e.g. sattu, atta, ghee, dal, tea, rice, oil)
                if pt in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice', 'tea', 'coffee', 'sugar', 'salt', 'masala', 'cornflour']:
                    score += 40.0
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
                    rf'\b{num}kg\b'
                ]
                if any(re.search(p, combined) for p in patterns):
                    score += 50.0

        # 4. TRUSTED E-COMMERCE RETAILERS (High quality packshots)
        trusted = ['amazon', 'flipkart', 'bigbasket', 'bbassets', 'blinkit', 'grofers', 'zepto', 'jiomart', 'swiggy', 'tataneu', 'dailymart4u', 'f.nooncdn', 'exportersindia']
        if any(t in combined for t in trusted):
            score += 20.0

        # 5. SPAM / UNRELATED SITE PENALTY
        spam = ['pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 'cartoon', 'recipe']
        if any(s in combined for s in spam):
            score -= 60.0

        scored.append((score, item, brand_match_count))

    # If at least one image had a brand match, strictly filter out images that had ZERO brand match
    if has_any_brand_match and brand_tokens:
        scored = [x for x in scored if x[2] > 0 or x[0] > 0]

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]

# Test with Tata Sampann
test_cases = [
    {
        "brand": "Tata Sampann",
        "product_name": "Tata Sampann 100% Chana Sattu",
        "variant_label": "500g"
    },
    {
        "brand": "24 Mantra Organic",
        "product_name": "24 Mantra Organic 7 Grain Methi Atta",
        "variant_label": "1kg"
    },
    {
        "brand": "9AM",
        "product_name": "9AM Cornflour",
        "variant_label": "500g"
    }
]

client = httpx.Client(base_url="http://localhost:8888", timeout=15.0)

for tc in test_cases:
    print(f"\n=======================================================")
    print(f"TEST: Brand='{tc['brand']}', Product='{tc['product_name']}', Variant='{tc['variant_label']}'")
    
    # 1. Product Query (Without Size)
    p_raw = f"{tc['brand']} {tc['product_name']}" if tc['brand'].lower() not in tc['product_name'].lower() else tc['product_name']
    p_query = sanitize_search_query(p_raw)
    resp_p = client.get('/search', params={'q': p_query, 'category_images': 1, 'format': 'json', 'engines': 'google images,bing images,duckduckgo images'}).json()
    ranked_p = score_and_rank_images(resp_p.get('results', []), tc['brand'], tc['product_name'], is_variant=False)
    
    print(f"\n-> PRODUCT SEARCH (WITHOUT SIZE): Query='{p_query}'")
    if ranked_p:
        top_p = ranked_p[0]
        print(f"   BEST MATCH: Title: {top_p.get('title')}")
        print(f"               URL:   {top_p.get('img_src') or top_p.get('thumbnail_src')}")
    else:
        print("   NO RESULT FOUND!")

    # 2. Variant Query (With Size)
    v_raw = f"{p_raw} {tc['variant_label']}"
    v_query = sanitize_search_query(v_raw)
    resp_v = client.get('/search', params={'q': v_query, 'category_images': 1, 'format': 'json', 'engines': 'google images,bing images,duckduckgo images'}).json()
    ranked_v = score_and_rank_images(resp_v.get('results', []), tc['brand'], tc['product_name'], tc['variant_label'], is_variant=True)
    
    print(f"\n-> VARIANT SEARCH (WITH SIZE): Query='{v_query}'")
    if ranked_v:
        top_v = ranked_v[0]
        print(f"   BEST MATCH: Title: {top_v.get('title')}")
        print(f"               URL:   {top_v.get('img_src') or top_v.get('thumbnail_src')}")
    else:
        print("   NO RESULT FOUND!")
