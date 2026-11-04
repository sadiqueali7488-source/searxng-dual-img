import asyncio
import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend directory is in sys.path
sys.path.insert(0, 'backend')
from searxng_client import search_searxng_image

async def run_test():
    async with httpx.AsyncClient() as client:
        print("Testing SearXNG + PaddleOCR Pipeline...")
        print("Searching for: Tata Sampann 100% Chana Sattu (Variant: 500g)...")
        
        # Test 1: Product Image (Without Size)
        res_p = await search_searxng_image(
            client=client,
            base_url="http://localhost:8888",
            query="Tata Sampann 100 Chana Sattu",
            brand="Tata Sampann",
            product_name="Tata Sampann 100% Chana Sattu",
            is_variant=False
        )
        print("\n-> PRODUCT IMAGE RESULT:")
        print("   Status:    ", res_p["status"])
        if res_p["best_image"]:
            print("   Title:     ", res_p["best_image"].title)
            print("   URL:       ", res_p["best_image"].img_src)
            print("   Alternatives count:", len(res_p["alternatives"]))

        # Test 2: Variant Image (With Size - PaddleOCR checks 500g on packet)
        res_v = await search_searxng_image(
            client=client,
            base_url="http://localhost:8888",
            query="Tata Sampann 100 Chana Sattu 500g",
            brand="Tata Sampann",
            product_name="Tata Sampann 100% Chana Sattu",
            variant_label="500g",
            is_variant=True
        )
        print("\n-> VARIANT IMAGE RESULT (With OCR Size Check):")
        print("   Status:    ", res_v["status"])
        if res_v["best_image"]:
            print("   Title:     ", res_v["best_image"].title)
            print("   URL:       ", res_v["best_image"].img_src)
            print("   Alternatives count:", len(res_v["alternatives"]))

asyncio.run(run_test())

