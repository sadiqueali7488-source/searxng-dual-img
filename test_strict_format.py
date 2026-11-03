import httpx
import time
import io
import pandas as pd
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)

# CSV containing 9:00 AM bug, missing rupee signs, array brackets in variant_axes and images
csv_content = """sku,variant_label,product_name,product_description,product_prescription_required,brand_name,brand_description,brand_image,parent_category_name,parent_category_description,parent_category_icon,parent_category_prescription_required,parent_category_is_featured,category_name,category_description,category_icon,category_prescription_required,category_is_featured,mrp,sale_price,tax_inclusive,barcode,gtin,mpn,color_hex,product_attributes,variant_attributes,variant_axes,product_images,variant_images
TEST-001,500g,9AM Cornflour,9AM Cornflour pure starch,FALSE,9:00 AM,,,Grocery,,,FALSE,FALSE,Flour,,,FALSE,FALSE,120,110,inclusive,379,,2400,,Speciality:Pure;Pack Size:500g;Type:Starch,Pack Size:500g;Brand:9AM,["Pack Size"],["https://old.com/img1.webp"],["https://old.com/img2.webp"]
TEST-002,1kg,24 Mantra Organic 7 Grain Methi Atta,24 Mantra Organic Atta,FALSE,24 Mantra Organic,,,Grocery,,,FALSE,FALSE,Atta,,,FALSE,FALSE,₹130.00,₹130.00,inclusive,380,,2401,,Speciality:Multigrain|Pack Size:1kg,Pack Size:1kg,["Pack Size"],["https://old.com/img3.webp"],["https://old.com/img4.webp"]"""

files = {"file": ("test_products.csv", csv_content.encode("utf-8"), "text/csv")}
upload = client.post("/api/upload", files=files).json()
job_id = upload["job_id"]
print(f"Uploaded! Job ID: {job_id}")

config = {
    "searxng_url": "http://localhost:8888",
    "query_column": "product_name",
    "concurrency": 2
}
client.post(f"/api/start-search/{job_id}", json=config)

for _ in range(15):
    time.sleep(1)
    st = client.get(f"/api/job/{job_id}").json()
    if st["status"]["state"] in ["completed", "cancelled", "error"]:
        break

csv_bytes = client.get(f"/api/download/{job_id}?format=csv").content
df = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig", dtype=str)

row1 = df.iloc[0]
print("\n--- ROW 1 (9AM Product) ---")
print("Brand Name (expected '9AM'):", repr(row1["brand_name"]))
print("MRP (expected '₹120.00'):", repr(row1["mrp"]))
print("Sale Price (expected '₹110.00'):", repr(row1["sale_price"]))
print("Variant Axes (expected 'Pack Size'):", repr(row1["variant_axes"]))
print("Product Attributes (expected '|' separated):", repr(row1["product_attributes"]))
print("Variant Attributes (expected '|' separated):", repr(row1["variant_attributes"]))
print("Product Images (pure link):", repr(row1["product_images"]))
print("Variant Images (pure link):", repr(row1["variant_images"]))

assert row1["brand_name"] == "9AM", f"Expected '9AM' but got {row1['brand_name']}"
assert "₹" in row1["mrp"], f"Rupee symbol missing in {row1['mrp']}"
assert "[" not in row1["variant_axes"], f"Array bracket in variant_axes: {row1['variant_axes']}"
assert "[" not in row1["product_images"], f"Array bracket in product_images: {row1['product_images']}"
assert row1["product_images"].startswith("http"), f"Image link invalid: {row1['product_images']}"
assert ";" not in row1["product_attributes"], f"Semicolon in product_attributes: {row1['product_attributes']}"

print("\nALL 6 STRICT FORMAT TESTS PASSED 100%!")

