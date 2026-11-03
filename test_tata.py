import httpx
import json
import pandas as pd
import io
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

data = [
    {
        "sku": "AFS-TATAS-SATTU-500G-PCH-001",
        "variant_label": "500g",
        "product_name": "Tata Sampann 100% Chana Sattu",
        "product_description": "Tata Sampann 100% Chana Sattu",
        "product_prescription_required": "FALSE",
        "brand_name": "Tata Sampann",
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
        "mrp": "₹99.00",
        "sale_price": "89.00",
        "tax_inclusive": "inclusive",
        "barcode": "8901234567890",
        "gtin": "",
        "mpn": "TS500",
        "color_hex": "",
        "product_attributes": "Speciality:Chana Sattu|Pack Size:500g|Type:Sattu|Brand:Tata Sampann",
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

with httpx.Client(base_url="http://localhost:8000", timeout=20.0) as client:
    upload_res = client.post("/api/upload", files={"file": ("tata_test.csv", csv_bytes, "text/csv")}).json()
    job_id = upload_res["job_id"]
    print("Job ID:", job_id)

    client.post(f"/api/start-search/{job_id}", json={
        "searxng_url": "http://localhost:8888",
        "query_column": "product_name",
        "concurrency": 1
    })

    for _ in range(15):
        time.sleep(1)
        st = client.get(f"/api/job/{job_id}").json()["status"]
        print(f"Status: {st['state']}, Progress: {st['progress_percent']}%")
        if st["state"] in ["completed", "error"]:
            break

    job_data = client.get(f"/api/job/{job_id}").json()
    row = job_data["rows"][0]
    print("\n================ FINAL RESULT FOR TATA SAMPANN ================")
    print("Product Query:       ", row.get("product_query"))
    print("Product Image URL:   ", row.get("product_image_url"))
    print("Product Engine:      ", row.get("product_engine"))
    print("Variant Query:       ", row.get("variant_query"))
    print("Variant Image URL:   ", row.get("variant_image_url"))
    print("Variant Engine:      ", row.get("variant_engine"))

