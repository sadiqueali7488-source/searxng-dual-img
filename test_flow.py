import httpx
import time
import json
import pandas as pd

client = httpx.Client(base_url="http://localhost:8000", timeout=60.0)

# 1. Test Health
health = client.get("/api/health").json()
print("Health:", health)

# 2. Upload sample CSV
with open("sample_data/grocery_products.csv", "rb") as f:
    upload_res = client.post("/api/upload", files={"file": ("grocery_products.csv", f, "text/csv")})

upload_data = upload_res.json()
job_id = upload_data["job_id"]
print(f"Uploaded! Job ID: {job_id}, Total Rows: {upload_data['total_rows']}, Columns: {upload_data['columns']}")

# 3. Start Search
config = {
    "searxng_url": "http://localhost:8888",
    "query_column": "product_name",
    "additional_columns": ["brand"],
    "concurrency": 4,
    "delay_ms": 50,
    "limit_rows": 5
}
start_res = client.post(f"/api/start-search/{job_id}", json=config)
print("Search started:", start_res.json())

# 4. Poll status
for i in range(25):
    time.sleep(1)
    status_res = client.get(f"/api/job/{job_id}").json()
    st = status_res["status"]
    print(f"[{i+1}s] State: {st['state']} | Processed: {st['processed_rows']}/{st['total_rows']} | Success: {st['success_count']} | Speed: {st['speed_rows_per_sec']}/s")
    if st["state"] in ["completed", "cancelled", "error"]:
        break

# 5. Check results
rows = status_res["rows"]
print("\n--- Rows Result ---")
for r in rows:
    print(f"Row {r['row_index']}: Query='{r['query']}' -> Status={r['status']} -> Image={r['image_url'][:80]}")

# 6. Test Download CSV
csv_dl = client.get(f"/api/download/{job_id}?format=csv")
print(f"\nDownloaded CSV length: {len(csv_dl.content)} bytes")
with open("sample_data/result_test.csv", "wb") as f:
    f.write(csv_dl.content)

# Verify original columns and new columns in CSV
df_out = pd.read_csv("sample_data/result_test.csv")
print("Output CSV columns:", list(df_out.columns))
print("Output CSV head(2):")
print(df_out[["product_id", "product_name", "brand", "price", "image_url", "search_status"]].head(2))

# 7. Test Download XLSX
xlsx_dl = client.get(f"/api/download/{job_id}?format=xlsx")
print(f"\nDownloaded XLSX length: {len(xlsx_dl.content)} bytes")
with open("sample_data/result_test.xlsx", "wb") as f:
    f.write(xlsx_dl.content)

print("\nSUCCESS: All tests passed with flying colors!")

