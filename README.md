# ⚡ SearXNG Product Image Finder (CSV & Excel)

A high-performance full-stack application built with **Python (FastAPI + Async HTTPX)** and **React (Vite + Tailwind CSS)** that automates finding and downloading product image URLs for datasets, replacing fragile n8n workflows with a dedicated, ultra-fast dashboard.

---

## ✨ Features

- **100% Original Data Preservation**: Keeps every single column, row order, data type, and special characters from your uploaded CSV or Excel (`.xlsx`, `.xls`) file intact.
- **SearXNG Image Search Integration**: Queries your local SearXNG instance (`http://localhost:8888/search?q=...&category_images=1&format=json`) directly with configurable safe search, language, and timeouts.
- **Asynchronous Concurrent Search**: Process hundreds or thousands of products concurrently (configurable from 1 to 20 parallel workers) with controllable pacing/delay.
- **Real-Time Live Dashboard**:
  - Live progress bar with completion percentage and row counts.
  - Real-time stats: Speed (rows/sec), ETA countdown, Image Found count, Not Found/Errors.
  - Pause, Resume, and Cancel job controls.
- **Interactive Results Table**:
  - Thumbnail previews with instant click-to-enlarge lightbox.
  - One-click copy for found image URLs.
  - Resolution and search engine tags.
  - **Alternative Image Picker**: Preview top 5 alternative SearXNG results for any item and swap it with 1 click!
- **Quick Search Playground**: Test any keyword (e.g. `pizza hut`, `coca cola can`) instantly right in the app without uploading a file.
- **Multiple Export Formats**:
  - **CSV**: UTF-8 with BOM for clean opening in Microsoft Excel without character encoding glitches.
  - **Excel (.xlsx)**: Clean spreadsheet with all original columns + new image columns:
    * `image_url`
    * `thumbnail_url`
    * `image_title`
    * `image_resolution`
    * `search_engine`
    * `search_status`

---

## 🚀 Quick Start (1 Click)

Simply double-click:
```bat
run.bat
```
This starts the backend and automatically opens the dashboard in your default browser at **`http://localhost:8000`**.

---

## 🐳 Docker (recommended: app + 3 SearXNG instances)

```bash
docker compose up -d --build
```
Open **`http://localhost:8000`**. The first start downloads the PP-OCRv6 and OpenCLIP models (~1 GB) into the `models` volume; later starts are instant.

- **Jobs survive restarts**: every finished row is saved (`app-cache` volume); a running job continues after `docker compose restart`.
- **Search capacity**: searches are spread over `searxng-1..3` (`SEARXNG_URLS`). They share this machine's IP, so they add parallel capacity, not protection from engine rate limits.
- **Tuning** (environment of `app`): `OCR_THREADS_PER_ENGINE`, `OCR_WORKERS`, `STRICT_OCR_WORKERS`, `VISION_ENABLED=0` (switch vision off), `OCR_MODEL` / `STRICT_OCR_MODEL` (`tiny|small|medium`).

### How an image is chosen
1. Search: Bing (and Google when SearXNG is not blocked by it), then **Yandex Images** as a second index for hard rows; automatic retries (plain query, page 2, simpler product name; Retry Not Found adds page 3, Yandex page 2 and the size written with a space).
   If no image passes every rule, a LOW-flagged pick is used only when it is still this exact product (nearest size, or the same pack OCR could not fully read). A sibling product (other flavour / variant of the brand) is never used: such a row stays Not Found.
2. Every candidate is downloaded (dead links are never used) and checked by **PP-OCRv6** (brand, product, exact size, back panel, multi-pack text) and **OpenCLIP** (single front pack vs back / several packs / banner / crop, and whether it looks like this product).
3. The best candidate (rules + vision + same official photo on several stores + trusted store) must also pass a **second, stricter OCR**.
4. Product and variant images must show the same pack; if the variant (with size) isn't found, the product image is used and flagged.
5. The app shows each image's confidence (HIGH / MEDIUM / LOW) and source (own search, borrowed, nearest size, similar) as badges; check the LOW rows there. The downloaded file keeps your original columns with the image links filled in.

---

## 🛠️ Manual Startup

### 1. Backend (FastAPI)
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend (React Dev Server - Optional)
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** (proxies API calls automatically to port 8000).

---

## 📁 Project Structure

```
image search/
├── backend/
│   ├── main.py              # FastAPI endpoints, file upload & static files
│   ├── processor.py         # Concurrent batch processing, SSE broadcast & Excel/CSV export
│   ├── searxng_client.py    # Async SearXNG API integration & query handler
│   └── models.py            # Pydantic data schemas
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React application & dashboard
│   │   └── index.css        # Tailwind CSS styles
│   ├── package.json
│   └── vite.config.js       # Vite configuration & API proxy
├── sample_data/
│   ├── grocery_products.csv # Sample dataset with 10 products
│   └── grocery_products.xlsx# Sample Excel dataset
├── run.bat                  # One-click Windows launcher
├── start_dev.bat            # Developer launcher (hot-reloading)
└── test_flow.py             # Automated end-to-end test script
```

---

## ⚙️ SearXNG Configuration

Ensure SearXNG is running on your machine:
```
http://localhost:8888
```
Endpoint format used:
```
http://localhost:8888/search?q={query}&category_images=1&pageno=1&language=auto&time_range=&safesearch=0&format=json
```
If your SearXNG is hosted on a different port or host, you can edit it directly from the gear icon in the top header of the application.

