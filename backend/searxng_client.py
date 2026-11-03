import httpx
import logging
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

async def test_searxng_connection(base_url: str = "http://localhost:8888") -> Dict[str, Any]:
    """Test if SearXNG is reachable and format=json works."""
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
    safesearch: int = 0,
    time_range: str = "",
    language: str = "auto",
    top_limit: int = 6
) -> Dict[str, Any]:
    """
    Search SearXNG for image results for a given query.
    Returns:
    {
        "status": "SUCCESS" | "NOT_FOUND" | "ERROR",
        "best_image": ImageResult,
        "alternatives": List[ImageResult],
        "error_message": Optional[str]
    }
    """
    base_url = base_url.rstrip("/")
    search_url = f"{base_url}/search"
    params = {
        "q": query,
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
        results = data.get("results", [])

        if not results:
            return {
                "status": "NOT_FOUND",
                "best_image": None,
                "alternatives": [],
                "error_message": "No images found for this query"
            }

        image_items: List[ImageResult] = []
        for item in results[:top_limit]:
            img_src = normalize_url(item.get("img_src"))
            thumbnail_src = normalize_url(item.get("thumbnail_src"))
            page_url = normalize_url(item.get("url"))
            title = (item.get("title") or "").strip()
            resolution = item.get("resolution") or ""
            engine = item.get("engine") or ""

            # Primary image candidate
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

        # By default, take 2nd image (index 1) as requested by user ("xng me 2nd wla img lene blo 1st nahi")
        selected_idx = 1 if len(image_items) > 1 else 0
        best_image = image_items[selected_idx]
        alternatives = [img for i, img in enumerate(image_items) if i != selected_idx]

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

