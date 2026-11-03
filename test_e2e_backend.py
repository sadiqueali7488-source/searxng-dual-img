import httpx
import json
import pandas as pd
import io
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Create test dataframe mimicking the user's exact data from screenshots
data = [
    {
        "sku": "AFS-24M-7GRAINMETHI-1KG-NA-001",
        "variant_label": "1kg",
        "product_name": "24 Mantra Organic 7 Grain Methi Atta",
        "product_description": "24 Mantra Organic 7 Grain Methi Atta - Multigrain (Atta).",
        "product_prescription_required": "FALSE",
        "brand_name": "24 Mantra Organic",
        "brand_description": "",
        "brand_image": "",
        "parent_category_name": "Grocery",
        "parent_category_description": "",
        "parent_category_icon": "",
        "parent_category_prescription_required": "FALSE",
        "parent_category_is_featured": "FALSE",
        "category_name": "Atta, Flour & Sooji",
        "category_description": "",
        "category_icon": "",
        "category_prescription_required": "FALSE",
        "category_is_featured": "FALSE",
        "mrp": "₹130.00",
        "sale_price": "130",
        "tax_inclusive": "inclusive",
        "barcode": "379",
        "gtin": "",
        "mpn": "2400",
        "color_hex": "",
        "product_attributes": "Speciality:Multigrain\nPack Size:1kg\nType:Atta",
        "variant_attributes": "Pack Size:1kg",
        "variant_axes": '["Pack Size"]',
        "product_images": '["https://cdn.flymystore.com/old.webp"]',
        "variant_images": '["https://cdn.flymystore.com/old_var.webp"]'
    },
    {
        "sku": "9AM-CORN-500G-001",
        "variant_label": "500g",
        "product_name": "9AM Cornflour",
        "product_description": "9AM Cornflour 500g pouch",
        "product_prescription_required": "FALSE",
        "brand_name": "9:00 AM", # Excel bug in user's file!
        "brand_description": "",
        "brand_image": "",
        "parent_category_name": "Grocery",
        "parent_category_description": "",
        "parent_category_icon": "",
        "parent_category_prescription_required": "FALSE",
        "parent_category_is_featured": "FALSE",
        "category_name": "Atta, Flour & Sooji",
        "category_description": "",
        "category_icon": "",
        "category_prescription_required": "FALSE",
        "category_is_featured": "FALSE",
        "mrp": "70",
        "sale_price": "65.00",
        "tax_inclusive": "inclusive",
        "barcode": "9812",
        "gtin": "",
        "mpn": "901",
        "color_hex": "",
        "product_attributes": "Pack Size:500g; Type:Cornflour",
        "variant_attributes": "Pack Size:500g",
        "variant_axes": '["Pack Size"]',
        "product_images": "",
        "variant_images": ""
    }
]

df = pd.DataFrame(data)
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode('utf-8')

# Test upload against running FastAPI backend
with httpx.Client(base_url="http://localhost:8000") as client:
    resp = client.post("/api/upload", files={"file": ("test_products.csv", csv_bytes, "text/csv")})
    print("Upload status:", resp.status_code)
    upload_res = resp.json()
    job_id = upload_res["job_id"]
    print("Job ID:", job_id)
    print("Detected columns count:", len(upload_res["columns"]))

    # Start search
    config = {
        "searxng_url": "http://localhost:8888",
        "query_column": "product_name",
        "concurrency": 2,
        "safesearch": 0,
        "language": "auto"
    }
    start_resp = client.post(f"/api/start-search/{job_id}", json=config)
    print("Start status:", start_resp.status_code)

    # Wait for completion
    import time
    for _ in range(20):
        time.sleep(1)
        st_resp = client.get(f"/api/job/{job_id}")
        st_data = st_resp.json()
        st = st_data["status"]
        print(f"Status: {st['state']}, Progress: {st['progress_percent']}% ({st['processed_rows']}/{st['total_rows']})")
        if st["state"] in ["completed", "error"]:
            break

    # Download result CSV
    dl_resp = client.get(f"/api/download/{job_id}?format=csv")
    print("Download status:", dl_resp.status_code)
    result_csv = dl_resp.content.decode('utf-8-sig')
    print("\n--- RESULT CSV PREVIEW ---")
    lines = result_csv.splitlines()
    for l in lines[:5]:
        print(l)

    # Check assertions:
    res_df = pd.read_csv(io.StringIO(result_csv), dtype=str)
    print("\n--- CHECKS ---")
    for idx, r in res_df.iterrows():
        print(f"Row {idx+1}:")
        print(f"  product_name: {r['product_name']}")
        print(f"  brand_name: {r['brand_name']}")
        print(f"  mrp: {r['mrp']}, sale_price: {r['sale_price']}")
        print(f"  variant_axes: {r['variant_axes']}")
        print(f"  product_attributes: {r['product_attributes']}")
        print(f"  product_images: {r['product_images']}")
        print(f"  variant_images: {r['variant_images']}")
