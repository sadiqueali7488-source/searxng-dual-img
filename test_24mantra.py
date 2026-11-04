import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = 'http://localhost:8888/search'
query = '24 Mantra Organic 7 Grain Multigrain Atta'

params = {
    'q': query,
    'category_images': 1,
    'format': 'json',
    'engines': 'google images,bing images,duckduckgo images'
}

resp = httpx.get(url, params=params)
data = resp.json()
results = data.get('results', [])
print(f"Total results returned: {len(results)}")

for i, r in enumerate(results[:20]):
    title = r.get('title') or ''
    img = r.get('img_src') or r.get('thumbnail_src') or ''
    engine = r.get('engine') or ''
    print(f"[{i:02d}] Engine: {engine:12s} | Title: {title[:60]}")
    print(f"     URL: {img}")
