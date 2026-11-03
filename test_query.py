import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')
url = 'http://localhost:8888/search'

for q in [
    'Tata Sampann 100% Chana Sattu',
    'Tata Sampann 100% Chana Sattu 500g',
    '"Tata Sampann" "Chana Sattu"',
    'Tata Sampann Sattu 500g',
    'Tata Sampann Chana Sattu packet'
]:
    print(f"\n=================== QUERY: {q} ===================")
    params = {
        'q': q,
        'category_images': 1,
        'format': 'json',
        'engines': 'google images,bing images,duckduckgo images'
    }
    resp = httpx.get(url, params=params)
    data = resp.json()
    results = data.get('results', [])
    print('Total results:', len(results))
    for i, r in enumerate(results[:6]):
        print(f"[{i}] Title: {r.get('title')}")
        print(f"    Engine: {r.get('engine')}")
        print(f"    Img: {r.get('img_src') or r.get('thumbnail_src')}")
