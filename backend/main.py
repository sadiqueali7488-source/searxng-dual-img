import os
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from typing import Optional, List

from models import SearchConfig, JobStatus, RowResult, ImageResult
from processor import job_manager
from searxng_client import test_searxng_connection, search_searxng_image
import httpx

app = FastAPI(
    title="SearXNG Product Image Finder",
    description="Bulk Image URL Extractor for CSV and Excel files powered by SearXNG",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    searxng_status = await test_searxng_connection("http://localhost:8888")
    return {
        "status": "healthy",
        "searxng_connected": searxng_status.get("ok", False),
        "searxng_details": searxng_status
    }

@app.get("/api/test-searxng")
async def test_searxng(url: str = Query("http://localhost:8888")):
    result = await test_searxng_connection(url)
    return result

@app.get("/api/quick-search")
async def quick_search(
    q: str = Query(..., description="Search query"),
    url: str = Query("http://localhost:8888"),
    safesearch: int = Query(0),
    language: str = Query("auto")
):
    async with httpx.AsyncClient(timeout=15.0) as client:
        result = await search_searxng_image(
            client=client,
            base_url=url,
            query=q,
            safesearch=safesearch,
            language=language,
            top_limit=12
        )
        return result

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file.csv"
    ext = filename.lower().split(".")[-1]

    content = await file.read()
    try:
        if ext in ["xlsx", "xls"]:
            df = pd.read_excel(content, dtype=str)
        elif ext == "csv":
            try:
                # Try UTF-8 first with dtype=str to prevent '9AM' from becoming '9:00 AM'
                df = pd.read_csv(pd.io.common.BytesIO(content), encoding="utf-8", dtype=str)
            except UnicodeDecodeError:
                # Fallback to latin1 / cp1252
                df = pd.read_csv(pd.io.common.BytesIO(content), encoding="latin1", dtype=str)
        else:
            raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .xls files are supported.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file contains no data.")

    # Fill NaN values with empty string for clean JSON serialization
    clean_df = df.fillna("")
    job = job_manager.create_job_from_dataframe(clean_df, filename)

    # Preview first 10 rows
    preview = clean_df.head(10).to_dict(orient="records")
    columns = list(clean_df.columns)

    return {
        "job_id": job.job_id,
        "filename": filename,
        "total_rows": len(clean_df),
        "columns": columns,
        "preview": preview
    }

@app.post("/api/start-search/{job_id}")
async def start_search(job_id: str, config: SearchConfig):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if config.query_column not in job.original_columns:
        raise HTTPException(status_code=400, detail=f"Column '{config.query_column}' not found in file")

    await job_manager.start_job(job_id, config)
    return {
        "message": "Search started successfully",
        "job_id": job_id,
        "total_rows": job.status.total_rows
    }

@app.get("/api/stream-progress/{job_id}")
async def stream_progress(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        q = job.add_listener()
        try:
            # Yield initial status immediately
            yield f"event: status_update\ndata: {json.dumps(job.status.model_dump())}\n\n"
            
            # Send all current row states if already populated
            if job.row_results:
                batch_rows = [r.model_dump() for r in job.row_results]
                yield f"event: initial_rows\ndata: {json.dumps(batch_rows)}\n\n"

            while True:
                msg = await q.get()
                event_type = msg.get("event", "message")
                data = json.dumps(msg.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"
                if event_type == "job_done":
                    break
        except asyncio.CancelledError:
            pass
        finally:
            job.remove_listener(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/job/{job_id}")
async def get_job_details(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job.status.model_dump(),
        "columns": job.original_columns,
        "rows": [r.model_dump() for r in job.row_results]
    }

@app.post("/api/job/{job_id}/pause")
async def pause_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.pause()
    await job.broadcast("status_update", job.status.model_dump())
    return {"message": "Job paused", "status": job.status.state}

@app.post("/api/job/{job_id}/resume")
async def resume_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.resume()
    await job.broadcast("status_update", job.status.model_dump())
    return {"message": "Job resumed", "status": job.status.state}

@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel()
    await job.broadcast("status_update", job.status.model_dump())
    return {"message": "Job cancelled", "status": job.status.state}

@app.post("/api/job/{job_id}/swap-image")
async def swap_row_image(
    job_id: str,
    row_index: int = Form(...),
    new_image_json: str = Form(...),
    target: str = Form("product")
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        img_dict = json.loads(new_image_json)
        image_obj = ImageResult(**img_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    job.update_row_image(row_index, image_obj, target=target)
    updated_row = job.row_results[row_index]
    await job.broadcast("row_update", updated_row.model_dump())
    return {"message": "Row image updated", "row": updated_row.model_dump()}

@app.post("/api/job/{job_id}/cycle-image/{row_index}")
async def cycle_image(job_id: str, row_index: int, target: str = Query("product")):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    updated = job.cycle_row_image(row_index, target=target)
    if not updated:
        raise HTTPException(status_code=400, detail="No alternative images to cycle")

    await job.broadcast("row_update", updated.model_dump())
    return {"message": "Cycled to next alternative image", "row": updated.model_dump()}

@app.post("/api/job/{job_id}/retry-row/{row_index}")
async def retry_row(
    job_id: str,
    row_index: int,
    custom_query: Optional[str] = Form(None),
    target: str = Form("both")
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async with httpx.AsyncClient(timeout=15.0) as client:
        updated = await job.retry_row_search(row_index, client, custom_query, target=target)
    if not updated:
        raise HTTPException(status_code=404, detail="Row not found")

    return {"message": "Row search retried", "row": updated.model_dump()}

@app.post("/api/job/{job_id}/retry-not-found")
async def retry_not_found(job_id: str):
    try:
        result = await job_manager.retry_not_found(job_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{job_id}")
async def download_results(job_id: str, format: str = Query("csv")):
    try:
        content_bytes, filename, media_type = job_manager.export_dataset(job_id, format)
        return Response(
            content=content_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Mount frontend production build if available
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

