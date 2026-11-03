import httpx
import time
import io
import pandas as pd

client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)

csv_text = """sku,variant_label,product_name,product_description,product_prescription_required,brand_name,brand_description,brand_image,parent_category_name,parent_category_description,parent_category_icon,parent_category_prescription_required,parent_category_is_featured,category_name,category_description,category_icon,category_prescription_required,category_is_featured,mrp,sale_price,tax_inclusive,barcode,gtin,mpn,color_hex,product_attributes,variant_attributes,variant_axes,product_images,variant_images
AFS-24M-7GRAINMETHI-1KG-NA-001,1kg,24 Mantra Organic 7 Grain Methi Atta,24 Mantra Organic 7 Grain Methi Atta - Multigrain (Atta),FALSE,24 Mantra Organic,,,Grocery,,,FALSE,FALSE,Atta Flour and Sooji,,,FALSE,FALSE,130.00,130.00,inclusive,379,,2400,,Speciality:Multigrain,Pack Size:1kg,["Pack Size"],["https://old-cdn.com/old1.webp"],["https://old-cdn.com/old2.webp"]"""

files = {"file": ("product_master.csv", csv_text.encode("utf-8"), "text/csv")}
upload = client.post("/api/upload", files=files).json()
job_id = upload["job_id"]
print("Uploaded! Total cols:", len(upload["columns"]))

config = {
    "searxng_url": "http://localhost:8888",
    "query_column": "product_name",
    "additional_columns": [],
    "concurrency": 2
}
client.post(f"/api/start-search/{job_id}", json=config)

for _ in range(15):
    time.sleep(1)
    st = client.get(f"/api/job/{job_id}").json()
    if st["status"]["state"] in ["completed", "cancelled", "error"]:
        break

csv_bytes = client.get(f"/api/download/{job_id}?format=csv").content
df_res = pd.read_csv(io.BytesIO(csv_bytes))
print("Result cols count:", len(df_res.columns))
print("product_images value:", df_res["product_images"].iloc[0])
print("variant_images value:", df_res["variant_images"].iloc[0])
expected_cols = [
    'sku', 'variant_label', 'product_name', 'product_description',
    'product_prescription_required', 'brand_name', 'brand_description',
    'brand_image', 'parent_category_name', 'parent_category_description',
    'parent_category_icon', 'parent_category_prescription_required',
    'parent_category_is_featured', 'category_name', 'category_description',
    'category_icon', 'category_prescription_required', 'category_is_featured',
    'mrp', 'sale_price', 'tax_inclusive', 'barcode', 'gtin', 'mpn', 'color_hex',
    'product_attributes', 'variant_attributes', 'variant_axes', 'product_images',
    'variant_images'
]
print("Columns match exactly 30 in order:", list(df_res.columns) == expected_cols)

