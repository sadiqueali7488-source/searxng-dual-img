import httpx
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def score_image(img, brand: str, product_name: str, variant_label: str = "", is_variant: bool = False) -> float:
    title = (img.get("title") or "").lower()
    url = (img.get("img_src") or img.get("url") or "").lower()
    combined_text = f"{title} {url}"

    score = 0.0

    # 1. BRAND MATCHING (Most Important)
    brand_tokens = [t.lower() for t in re.split(r'[\s\-_]+', brand) if len(t) > 2]
    if brand_tokens:
        brand_matches = sum(1 for t in brand_tokens if t in combined_text)
        if brand_matches == len(brand_tokens):
            score += 60.0  # Full brand match
        elif brand_matches > 0:
            score += 35.0  # Partial brand match
        else:
            score -= 50.0  # Penalize results missing the brand!

    # 2. CORE PRODUCT WORDS MATCHING
    # Extract important keywords from product name (skip generic words like '100%', 'pouch', 'pure')
    stop_words = {'100%', 'pure', 'pack', 'pouch', 'packet', 'the', 'and', 'for', 'with', 'original', 'best'}
    prod_tokens = [t.lower() for t in re.split(r'[\s\-_]+', product_name) if len(t) > 2 and t.lower() not in stop_words]
    
    prod_matches = 0
    for t in prod_tokens:
        if t in brand_tokens:
            continue
        if t in combined_text:
            prod_matches += 1
            # Special bonus for specific nouns like 'sattu', 'atta', 'ghee', 'oil', 'tea', 'coffee'
            if t in ['sattu', 'besan', 'atta', 'maida', 'sooji', 'flour', 'oil', 'ghee', 'dal', 'rice']:
                score += 25.0
            else:
                score += 10.0

    # 3. VARIANT / SIZE MATCHING (For variant search)
    if is_variant and variant_label:
        v_clean = variant_label.lower().replace(" ", "")
        # e.g., 500g -> check '500g', '500 g', '500gm', '500 gm'
        v_num_match = re.search(r'(\d+)\s*(kg|g|gm|ml|l|ltr)?', variant_label.lower())
        if v_num_match:
            num = v_num_match.group(1)
            unit = v_num_match.group(2) or ""
            patterns = [
                rf'\b{num}\s*{unit}\b',
                rf'\b{num}{unit}\b',
                rf'\b{num}\s*gm\b',
                rf'\b{num}gm\b'
            ]
            if any(re.search(p, combined_text) for p in patterns):
                score += 30.0  # Size matched!

    # 4. TRUSTED E-COMMERCE PACKSHOT DOMAINS
    trusted_domains = [
        'amazon', 'flipkart', 'bigbasket', 'bbassets', 'grofers', 'blinkit', 
        'zepto', 'jiomart', 'swiggy', 'tataneu', 'tatanutrikorner', 'f.nooncdn', 'dailymart4u'
    ]
    if any(d in url for d in trusted_domains):
        score += 15.0

    # 5. PENALIZE SUSPICIOUS / SPAM / UNRELATED SITES
    spam_indicators = ['pinterest', 'wallpaper', 'facebook', 'instagram', 'clipart', 'vector', 'logo-vector']
    if any(s in url for s in spam_indicators):
        score -= 40.0

    return score

# Let's test on the Tata Sampann query
resp = httpx.get('http://localhost:8888/search', params={
    'q': 'Tata Sampann 100% Chana Sattu 500g',
    'category_images': 1,
    'format': 'json',
    'engines': 'google images,bing images,duckduckgo images'
})
results = resp.json().get('results', [])

scored = []
for r in results:
    s = score_image(r, brand="Tata Sampann", product_name="Tata Sampann 100% Chana Sattu", variant_label="500g", is_variant=True)
    scored.append((s, r))

scored.sort(key=lambda x: x[0], reverse=True)

print(f"Top 5 Ranked Results for 'Tata Sampann 100% Chana Sattu 500g':")
for i, (score, r) in enumerate(scored[:5]):
    print(f"#{i+1} [Score: {score:.1f}] Title: {r.get('title')}")
    print(f"    Engine: {r.get('engine')}")
    print(f"    URL: {r.get('img_src') or r.get('thumbnail_src')}")

