import asyncio
import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'backend')
from searxng_client import search_searxng_image

async def run_test():
    async with httpx.AsyncClient() as client:
        print("Testing Strict Product (No Size) vs Variant (With Size 1kg) Selection...")
        print("Product: 24 Mantra Organic 7 Grain Multigrain Atta\n")

        # 1. Product Image (WITHOUT SIZE)
        res_p = await search_searxng_image(
            client=client,
            base_url="http://localhost:8888",
            query="24 Mantra Organic 7 Grain Multigrain Atta",
            brand="24 Mantra Organic",
            product_name="24 Mantra Organic 7 Grain Multigrain Atta",
            is_variant=False
        )

        # 2. Variant Image (WITH SIZE)
        res_v = await search_searxng_image(
            client=client,
            base_url="http://localhost:8888",
            query="24 Mantra Organic 7 Grain Multigrain Atta 1kg",
            brand="24 Mantra Organic",
            product_name="24 Mantra Organic 7 Grain Multigrain Atta",
            variant_label="1kg",
            is_variant=True
        )

        print("=============================================================")
        print("PRODUCT IMAGE (WITHOUT SIZE):")
        if res_p["best_image"]:
            print("  Title: ", res_p["best_image"].title)
            print("  URL:   ", res_p["best_image"].img_src)
        else:
            print("  NOT FOUND!")

        print("\nVARIANT IMAGE (WITH SIZE 1kg):")
        if res_v["best_image"]:
            print("  Title: ", res_v["best_image"].title)
            print("  URL:   ", res_v["best_image"].img_src)
        else:
            print("  NOT FOUND!")
        print("=============================================================")

asyncio.run(run_test())
