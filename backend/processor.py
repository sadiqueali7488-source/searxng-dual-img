import asyncio
import time
import uuid
import io
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import httpx
from models import SearchConfig, JobStatus, RowResult, ImageResult
from searxng_client import search_searxng_image, sanitize_search_query, is_query_multipack

PRODUCT_MASTER_30_COLUMNS = [
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

# Column-name synonyms (compared after lower-casing and removing spaces/_/-), in priority order.
# Lets any Excel work: "Item Name", "PRODUCT NAME", "Company", "Pack Size", "Net Wt"...
PRODUCT_COLUMN_NAMES = [
    'productname', 'product', 'productitle', 'producttitle', 'itemname', 'item', 'itemdescription',
    'name', 'title', 'skuname', 'articlename', 'article', 'description', 'productdescription',
]
BRAND_COLUMN_NAMES = [
    'brandname', 'brand', 'brands', 'manufacturer', 'manufacturername', 'mfr', 'company', 'companyname',
    'make', 'maker', 'marketedby', 'vendor', 'vendorname', 'label',
]
VARIANT_COLUMN_NAMES = [
    'variantlabel', 'variant', 'variantname', 'size', 'packsize', 'packingsize', 'packetsize', 'pack',
    'weight', 'netweight', 'netwt', 'netqty', 'netquantity', 'volume', 'netvolume', 'grammage',
    'uom', 'unit', 'quantity', 'qty',
]
# Columns that hold a stock count / price for some sheets: only used if their values look like sizes
_AMBIGUOUS_VARIANT_NAMES = {'quantity', 'qty', 'unit', 'uom', 'pack'}
_SIZE_VALUE_RE = re.compile(r'\d\s*(?:kg|g|gm|gms|gram|grams|mg|ml|l|ltr|litre|liter|lb|oz|pcs|pc|xs|s|m|xl|xxl)\b', re.I)

def _norm_col(name: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def _find_column(columns: List[str], names: List[str], contains: List[str], exclude: Optional[set] = None) -> str:
    exclude = exclude or set()
    normed = {c: _norm_col(c) for c in columns if c not in exclude}
    for n in names:                                   # 1. exact synonym, in priority order
        for col, nc in normed.items():
            if nc == n:
                return col
    for col, nc in normed.items():                    # 2. name contains a keyword ("Brand Name (EN)")
        if any(k in nc for k in contains):
            return col
    return ""

def detect_columns(df: pd.DataFrame, product_hint: str = "") -> Dict[str, str]:
    """Finds the product, brand and size columns of any sheet. Returns '' for a column that isn't there."""
    columns = [str(c) for c in df.columns]
    product = product_hint if product_hint in columns else _find_column(columns, PRODUCT_COLUMN_NAMES, ['productname', 'producttitle', 'itemname', 'itemtitle'])
    brand = _find_column(columns, BRAND_COLUMN_NAMES, ['brand', 'manufactur'], exclude={product})

    variant = ""
    for n in VARIANT_COLUMN_NAMES:
        for col in columns:
            if col in (product, brand) or _norm_col(col) != n:
                continue
            if n in _AMBIGUOUS_VARIANT_NAMES:
                # "Qty" may be a stock count: accept it only if most values look like 500g / 1L / XL
                sample = [str(v) for v in df[col].head(200).tolist() if str(v).strip() and str(v).lower() != "nan"]
                if not sample or sum(bool(_SIZE_VALUE_RE.search(v)) for v in sample) < 0.5 * len(sample):
                    continue
            variant = col
            break
        if variant:
            break
    if not variant:
        variant = _find_column(columns, [], ['variant', 'packsize', 'weight', 'size'], exclude={product, brand})
    return {"product": product, "brand": brand, "variant": variant}

def clean_brand(brand: Any, prod_name: str = "") -> str:
    b = "" if brand is None else str(brand).strip()
    if b.lower() == "nan":
        b = ""
    b = re.sub(r"^0?([0-9]+):00(?::00)?\s*(AM|PM)$", r"\1\2", b, flags=re.I)
    if "9:00" in b:
        b = "9AM"
    if prod_name and re.match(r"^9\s*AM\b", prod_name.strip(), re.I):
        b = "9AM"
    return b

def clean_currency(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip().replace(",", "")
    if not s or s.lower() == "nan":
        return ""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if match:
        num = float(match.group(1))
        return f"₹{num:.2f}"
    return s

def clean_axes(axes: Any) -> str:
    if axes is None:
        return ""
    s = str(axes).strip()
    if not s or s.lower() == "nan":
        return ""
    s = re.sub(r"^[\[\(\{\"\']+|[\]\)\}\"\']+$", "", s)
    s = s.replace('"', "").replace("'", "")
    parts = [p.strip() for p in re.split(r"[,;\|]", s) if p.strip()]
    return " | ".join(parts)

def clean_attributes(attr: Any) -> str:
    if attr is None:
        return ""
    s = str(attr).strip()
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    parts = [p.strip() for p in re.split(r"[;\|]", s) if p.strip()]
    return "|".join(parts)

def clean_image_link(img_val: Any) -> str:
    if img_val is None:
        return ""
    s = str(img_val).strip()
    if not s or s.lower() == "nan":
        return ""
    s = re.sub(r"^[\[\(\{\"\']+|[\]\)\}\"\']+$", "", s)
    s = s.replace('"', "").replace("'", "")
    return s.strip()


class SearchJob:
    def __init__(self, job_id: str, df: pd.DataFrame, filename: str):
        self.job_id = job_id
        self.filename = filename

        existing_cols = [c.lower() for c in df.columns]
        is_pm = any("sku" in c or "product_name" in c or "variant" in c for c in existing_cols)
        self.is_product_master = is_pm

        for col in df.columns:
            df[col] = df[col].fillna("").astype(str)

        if is_pm:
            for col in PRODUCT_MASTER_30_COLUMNS:
                if col not in df.columns:
                    df[col] = ""

            # Column-wise cleaning (row-by-row df.at is far too slow for 1 lakh rows)
            p_names = df['product_name'].tolist() if 'product_name' in df.columns else [""] * len(df)
            if 'brand_name' in df.columns:
                df['brand_name'] = [clean_brand(b, p) for b, p in zip(df['brand_name'].tolist(), p_names)]
            for col, fn in (('mrp', clean_currency), ('sale_price', clean_currency), ('variant_axes', clean_axes),
                            ('product_attributes', clean_attributes), ('variant_attributes', clean_attributes)):
                if col in df.columns:
                    df[col] = df[col].map(fn)
            for col in ('product_images', 'variant_images'):
                if col in df.columns:
                    df[col] = ""

            ordered_cols = [c for c in PRODUCT_MASTER_30_COLUMNS if c in df.columns]
            extra_cols = [c for c in df.columns if c not in PRODUCT_MASTER_30_COLUMNS]
            df = df[ordered_cols + extra_cols]

        self.df = df
        self.original_columns = list(df.columns)
        # Product / brand / size columns, whatever the sheet calls them (refined with the chosen query column on start)
        self.columns = detect_columns(df)
        self.config: Optional[SearchConfig] = None
        self.status = JobStatus(job_id=job_id, filename=filename, total_rows=len(df))
        self.row_results: List[RowResult] = []
        self.is_paused = False
        self.is_cancelled = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.listeners: List[asyncio.Queue] = []
        self.task: Optional[asyncio.Task] = None
        self.start_time: float = 0.0

    def row_values(self, orig: Dict[str, Any]) -> tuple:
        """(brand, product name, size) of one row, read from the detected columns."""
        def get(kind: str) -> str:
            col = self.columns.get(kind)
            val = str(orig.get(col, "")).strip() if col else ""
            return "" if val.lower() == "nan" else val
        product = get("product")
        return clean_brand(get("brand"), product), product, get("variant")

    def add_listener(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        if q in self.listeners:
            self.listeners.remove(q)

    async def broadcast(self, event_type: str, data: Any):
        payload = {"event": event_type, "data": data}
        for q in list(self.listeners):
            try:
                await q.put(payload)
            except Exception:
                pass

    def pause(self):
        self.is_paused = True
        self.pause_event.clear()
        self.status.state = "paused"

    def resume(self):
        self.is_paused = False
        self.pause_event.set()
        self.status.state = "running"

    def cancel(self):
        self.is_cancelled = True
        self.pause_event.set()
        self.status.state = "cancelled"
        if self.task and not self.task.done():
            self.task.cancel()

    def update_row_image(self, row_index: int, new_image: ImageResult, target: str = "product"):
        if 0 <= row_index < len(self.row_results):
            row = self.row_results[row_index]
            clean_url = clean_image_link(new_image.img_src)
            clean_thumb = clean_image_link(new_image.thumbnail_src) or clean_url

            if target == "variant":
                row.variant_image_url = clean_url
                row.variant_thumbnail_url = clean_thumb
                row.variant_resolution = new_image.resolution
                row.variant_engine = new_image.engine
                row.variant_status = "SUCCESS"
            else:
                row.product_image_url = clean_url
                row.product_thumbnail_url = clean_thumb
                row.product_resolution = new_image.resolution
                row.product_engine = new_image.engine
                row.product_status = "SUCCESS"

            # Backward compatibility
            row.image_url = row.product_image_url or row.variant_image_url
            row.thumbnail_url = row.product_thumbnail_url or row.variant_thumbnail_url
            row.status = "SUCCESS"

    def cycle_row_image(self, row_index: int, target: str = "product") -> Optional[RowResult]:
        if 0 <= row_index < len(self.row_results):
            row = self.row_results[row_index]
            alts = row.variant_alternatives if target == "variant" else row.product_alternatives

            if alts and len(alts) > 0:
                current_img = ImageResult(
                    img_src=row.variant_image_url if target == "variant" else row.product_image_url,
                    thumbnail_src=row.variant_thumbnail_url if target == "variant" else row.product_thumbnail_url,
                    resolution=row.variant_resolution if target == "variant" else row.product_resolution,
                    engine=row.variant_engine if target == "variant" else row.product_engine
                )
                next_img = alts.pop(0)
                clean_url = clean_image_link(next_img.img_src)
                clean_thumb = clean_image_link(next_img.thumbnail_src) or clean_url

                if target == "variant":
                    row.variant_image_url = clean_url
                    row.variant_thumbnail_url = clean_thumb
                    row.variant_resolution = next_img.resolution
                    row.variant_engine = next_img.engine
                    row.variant_status = "SUCCESS"
                else:
                    row.product_image_url = clean_url
                    row.product_thumbnail_url = clean_thumb
                    row.product_resolution = next_img.resolution
                    row.product_engine = next_img.engine
                    row.product_status = "SUCCESS"

                if current_img.img_src:
                    alts.append(current_img)

                row.image_url = row.product_image_url or row.variant_image_url
                row.thumbnail_url = row.product_thumbnail_url or row.variant_thumbnail_url
                row.status = "SUCCESS"
                return row
        return None

    async def retry_row_search(
        self,
        row_index: int,
        client: httpx.AsyncClient,
        custom_query: Optional[str] = None,
        target: str = "both"
    ) -> Optional[RowResult]:
        if 0 <= row_index < len(self.row_results):
            row = self.row_results[row_index]
            base_url = self.config.searxng_url if self.config else "http://localhost:8888"
            safesearch = self.config.safesearch if self.config else 0
            time_range = self.config.time_range if self.config else ""
            language = self.config.language if self.config else "auto"

            orig = row.original_data or {}
            brand_val, prod_val, var_val = self.row_values(orig)

            # 1. Search Product Image (Without Size)
            if target in ["both", "product"]:
                p_query = custom_query if (target == "product" and custom_query) else row.product_query
                row.product_status = "SEARCHING"
                await self.broadcast("row_update", row.model_dump())

                res_p = await search_searxng_image(
                    client=client,
                    base_url=base_url,
                    query=p_query,
                    brand=brand_val,
                    product_name=prod_val,
                    variant_label=var_val,
                    is_variant=False,
                    safesearch=safesearch,
                    time_range=time_range,
                    language=language
                )
                row.product_status = res_p["status"]
                if res_p["status"] == "SUCCESS" and res_p["best_image"]:
                    best = res_p["best_image"]
                    row.product_image_url = clean_image_link(best.img_src)
                    row.product_thumbnail_url = clean_image_link(best.thumbnail_src) or row.product_image_url
                    row.product_resolution = best.resolution
                    row.product_engine = best.engine
                    row.product_alternatives = res_p.get("alternatives", [])

            # 2. Search Variant Image (With Size)
            if target in ["both", "variant"]:
                v_query = custom_query if (target == "variant" and custom_query) else row.variant_query
                row.variant_status = "SEARCHING"
                await self.broadcast("row_update", row.model_dump())

                res_v = await search_searxng_image(
                    client=client,
                    base_url=base_url,
                    query=v_query,
                    brand=brand_val,
                    product_name=prod_val,
                    variant_label=var_val,
                    is_variant=True,
                    safesearch=safesearch,
                    time_range=time_range,
                    language=language
                )
                row.variant_status = res_v["status"]
                if res_v["status"] == "SUCCESS" and res_v["best_image"]:
                    best_v = res_v["best_image"]
                    row.variant_image_url = clean_image_link(best_v.img_src)
                    row.variant_thumbnail_url = clean_image_link(best_v.thumbnail_src) or row.variant_image_url
                    row.variant_resolution = best_v.resolution
                    row.variant_engine = best_v.engine
                    row.variant_alternatives = res_v.get("alternatives", [])

            # Overall status
            if row.product_image_url and row.variant_image_url:
                row.status = "SUCCESS"
            elif row.product_image_url or row.variant_image_url:
                row.status = "PARTIAL"
            else:
                row.status = "NOT_FOUND"

            row.image_url = row.product_image_url or row.variant_image_url
            row.thumbnail_url = row.product_thumbnail_url or row.variant_thumbnail_url

            await self.broadcast("row_update", row.model_dump())
            return row
        return None


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, SearchJob] = {}

    def create_job_from_dataframe(self, df: pd.DataFrame, filename: str) -> SearchJob:
        job_id = str(uuid.uuid4())
        job = SearchJob(job_id=job_id, df=df, filename=filename)
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[SearchJob]:
        return self.jobs.get(job_id)

    async def start_job(self, job_id: str, config: SearchConfig):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.config = config
        # The column the user picked as the product name is also the one the checks verify against
        job.columns = detect_columns(job.df, product_hint=config.query_column)
        job.status.state = "running"
        job.is_paused = False
        job.is_cancelled = False
        job.pause_event.set()

        rows_to_process = job.df
        if config.limit_rows and config.limit_rows > 0:
            rows_to_process = rows_to_process.head(config.limit_rows)

        total_rows = len(rows_to_process)
        job.status.total_rows = total_rows
        job.status.processed_rows = 0
        job.status.success_count = 0
        job.status.not_found_count = 0
        job.status.error_count = 0
        job.status.progress_percent = 0.0

        job.row_results = []
        for idx, original_data in enumerate(rows_to_process.to_dict(orient="records")):
            clean_original_data = {
                k: ("" if pd.isna(v) else str(v)) for k, v in original_data.items()
            }

            prod_name = clean_original_data.get(config.query_column, "").strip()
            brand, _, variant = job.row_values(clean_original_data)

            if "brand_name" in clean_original_data:
                clean_original_data["brand_name"] = brand

            # Clean product name to remove any size badges (1kg, 500g, etc.) for Product Image query
            clean_p_name = re.sub(r'\b\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr|pack\s+of\s+\d+|pcs|pieces)\b', '', prod_name, flags=re.I)
            clean_p_name = re.sub(r'\s+', ' ', clean_p_name).strip()
            if not clean_p_name:
                clean_p_name = prod_name

            is_combo = is_query_multipack(prod_name, variant)

            # Detect category type for smart search suffixes (Apparel / Toys vs FMCG Packaged Goods)
            apparel_keywords = {'top', 't-shirt', 'tshirt', 'shirt', 'jeans', 'trousers', 'pants', 'dress', 'kurti', 'kurta', 'saree', 'skirt', 'jacket', 'hoodie', 'shoes', 'sandals', 'sneakers', 'shorts', 'crop top', 'toy', 'toys', 'game'}
            is_apparel_or_goods = any(re.search(rf'\b{re.escape(k)}\b', prod_name.lower()) for k in apparel_keywords)

            # Query 1: Product Image (WITHOUT SIZE)
            p_parts = []
            if config.query_prefix:
                p_parts.append(config.query_prefix.strip())
            if brand and brand.lower() not in clean_p_name.lower():
                p_parts.append(brand)
            if clean_p_name:
                p_parts.append(clean_p_name)
            if config.query_suffix:
                p_parts.append(config.query_suffix.strip())
            else:
                if is_apparel_or_goods:
                    p_parts.append("front view product shot")
                else:
                    p_parts.append("single pack front packaging" if not is_combo else "front packaging pouch")
            product_query = sanitize_search_query(" ".join(p_parts).strip() or clean_p_name or brand or "product")

            # Query 2: Variant Image (WITH SIZE)
            v_parts = []
            if config.query_prefix:
                v_parts.append(config.query_prefix.strip())
            if brand and brand.lower() not in prod_name.lower():
                v_parts.append(brand)
            if prod_name:
                v_parts.append(prod_name)
            if variant and variant.lower() not in prod_name.lower():
                v_parts.append(variant)
            if config.query_suffix:
                v_parts.append(config.query_suffix.strip())
            else:
                if is_apparel_or_goods:
                    v_parts.append("front view product shot")
                else:
                    v_parts.append("combo pack front packaging" if is_combo else "single pack front packaging")
            variant_query = sanitize_search_query(" ".join(v_parts).strip() or product_query)

            job.row_results.append(RowResult(
                row_index=idx,
                original_data=clean_original_data,
                product_query=product_query,
                variant_query=variant_query,
                query=variant_query,
                product_status="PENDING",
                variant_status="PENDING",
                status="PENDING"
            ))

        job.task = asyncio.create_task(self._execute_job(job))
        return job

    async def _execute_job(self, job: SearchJob, specific_rows: Optional[List[RowResult]] = None):
        config = job.config
        job.start_time = time.time()
        timeout = httpx.Timeout(15.0, connect=5.0)
        last_status_push = 0.0

        async def push_status():
            # Throttled: with 1 lakh rows a status event per row floods the browser
            nonlocal last_status_push
            now = time.time()
            if now - last_status_push >= 0.5:
                last_status_push = now
                await job.broadcast("status_update", job.status.model_dump())

        # Each row runs two searches at once, each downloading up to 4 images in parallel
        limits = httpx.Limits(max_keepalive_connections=config.concurrency * 4, max_connections=config.concurrency * 10)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            target_rows = specific_rows if specific_rows is not None else job.row_results

            async def process_single_row(row_res: RowResult):
                if job.is_cancelled:
                    return
                await job.pause_event.wait()
                if job.is_cancelled:
                    return

                row_res.status = "SEARCHING"
                row_res.product_status = "SEARCHING"
                row_res.variant_status = "SEARCHING"
                job.status.current_query = f"{row_res.product_query} / {row_res.variant_query}"
                await job.broadcast("row_update", row_res.model_dump())

                if config.delay_ms > 0:
                    await asyncio.sleep(config.delay_ms / 1000.0)

                orig_row = row_res.original_data or {}
                b_val, p_val, v_val = job.row_values(orig_row)

                # 1. Product Image (Without Size) and 2. Variant Image (With Size):
                # separate independent searches and inspections, run side by side
                res_p, res_v = await asyncio.gather(
                    search_searxng_image(
                        client=client,
                        base_url=config.searxng_url,
                        query=row_res.product_query,
                        brand=b_val,
                        product_name=p_val,
                        variant_label=v_val,
                        is_variant=False,
                        safesearch=config.safesearch,
                        time_range=config.time_range,
                        language=config.language
                    ),
                    search_searxng_image(
                        client=client,
                        base_url=config.searxng_url,
                        query=row_res.variant_query,
                        brand=b_val,
                        product_name=p_val,
                        variant_label=v_val,
                        is_variant=True,
                        safesearch=config.safesearch,
                        time_range=config.time_range,
                        language=config.language
                    )
                )

                row_res.product_status = res_p["status"]
                if res_p["status"] == "SUCCESS" and res_p["best_image"]:
                    best_p = res_p["best_image"]
                    row_res.product_image_url = clean_image_link(best_p.img_src)
                    row_res.product_thumbnail_url = clean_image_link(best_p.thumbnail_src) or row_res.product_image_url
                    row_res.product_resolution = best_p.resolution
                    row_res.product_engine = best_p.engine
                    row_res.product_alternatives = res_p.get("alternatives", [])

                row_res.variant_status = res_v["status"]
                if res_v["status"] == "SUCCESS" and res_v["best_image"]:
                    best_v = res_v["best_image"]
                    row_res.variant_image_url = clean_image_link(best_v.img_src)
                    row_res.variant_thumbnail_url = clean_image_link(best_v.thumbnail_src) or row_res.variant_image_url
                    row_res.variant_resolution = best_v.resolution
                    row_res.variant_engine = best_v.engine
                    row_res.variant_alternatives = res_v.get("alternatives", [])

                # Determine overall status
                if row_res.product_image_url and row_res.variant_image_url:
                    row_res.status = "SUCCESS"
                    job.status.success_count += 1
                elif "ERROR" in (res_p["status"], res_v["status"]):
                    row_res.status = "ERROR"
                    row_res.error_message = res_p.get("error_message") or res_v.get("error_message")
                    job.status.error_count += 1
                else:
                    row_res.status = "NOT_FOUND"
                    job.status.not_found_count += 1

                row_res.image_url = row_res.product_image_url or row_res.variant_image_url
                row_res.thumbnail_url = row_res.product_thumbnail_url or row_res.variant_thumbnail_url

                job.status.processed_rows += 1
                elapsed = max(0.1, time.time() - job.start_time)
                job.status.elapsed_seconds = round(elapsed, 1)
                speed = job.status.processed_rows / elapsed
                job.status.speed_rows_per_sec = round(speed, 2)
                remaining = job.status.total_rows - job.status.processed_rows
                job.status.eta_seconds = round(remaining / speed, 1) if speed > 0 else 0
                job.status.progress_percent = round((job.status.processed_rows / job.status.total_rows) * 100, 1)

                await job.broadcast("row_update", row_res.model_dump())
                await push_status()

            # Fixed pool of workers pulling rows from a queue (not one task per row for 1 lakh rows)
            queue: asyncio.Queue = asyncio.Queue()
            for row in target_rows:
                queue.put_nowait(row)

            async def worker():
                while not job.is_cancelled:
                    try:
                        row_res = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await process_single_row(row_res)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        row_res.status = "ERROR"
                        row_res.error_message = str(e)
                        job.status.error_count += 1
                        job.status.processed_rows += 1
                        await job.broadcast("row_update", row_res.model_dump())

            try:
                await asyncio.gather(*(worker() for _ in range(config.concurrency)))
            except asyncio.CancelledError:
                job.status.state = "cancelled"
            except Exception as e:
                job.status.state = "error"

        if not job.is_cancelled:
            job.status.state = "completed"
            job.status.progress_percent = 100.0

        await job.broadcast("status_update", job.status.model_dump())
        await job.broadcast("job_done", {"job_id": job.job_id, "state": job.status.state})

    async def retry_not_found(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Find rows that were not found or failed
        failed_rows = [
            r for r in job.row_results
            if r.status in ["NOT_FOUND", "ERROR"] or not r.product_image_url or not r.variant_image_url
        ]

        if not failed_rows:
            return {"message": "No failed or not-found rows to retry!", "count": 0}

        job.status.state = "running"
        job.is_paused = False
        job.is_cancelled = False
        job.pause_event.set()

        # Retried rows are counted again when they finish (keeps progress at or below 100%)
        for r in failed_rows:
            if r.status == "ERROR":
                job.status.error_count = max(0, job.status.error_count - 1)
            elif r.status == "NOT_FOUND":
                job.status.not_found_count = max(0, job.status.not_found_count - 1)
            elif r.status == "SUCCESS":
                job.status.success_count = max(0, job.status.success_count - 1)
            if r.status not in ("PENDING", "SEARCHING"):
                job.status.processed_rows = max(0, job.status.processed_rows - 1)

        # Launch background retry task for failed rows
        job.task = asyncio.create_task(self._execute_job(job, specific_rows=failed_rows))
        return {"message": f"Retrying {len(failed_rows)} not-found rows...", "count": len(failed_rows)}

    def export_dataset(self, job_id: str, format_type: str = "csv") -> tuple[bytes, str, str]:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        rows_data = []
        for row_res in job.row_results:
            row_dict = dict(row_res.original_data)

            prod_link = clean_image_link(row_res.product_image_url)
            var_link = clean_image_link(row_res.variant_image_url)

            prod_name = row_dict.get("product_name", "")
            if "brand_name" in row_dict:
                row_dict["brand_name"] = clean_brand(row_dict.get("brand_name"), prod_name)
            if "mrp" in row_dict:
                row_dict["mrp"] = clean_currency(row_dict.get("mrp"))
            if "sale_price" in row_dict:
                row_dict["sale_price"] = clean_currency(row_dict.get("sale_price"))
            if "variant_axes" in row_dict:
                row_dict["variant_axes"] = clean_axes(row_dict.get("variant_axes"))
            if "product_attributes" in row_dict:
                row_dict["product_attributes"] = clean_attributes(row_dict.get("product_attributes"))
            if "variant_attributes" in row_dict:
                row_dict["variant_attributes"] = clean_attributes(row_dict.get("variant_attributes"))

            # DIRECT CLEAN LINKS WITHOUT ARRAY BRACKETS
            row_dict["product_images"] = prod_link
            row_dict["variant_images"] = var_link

            if not job.is_product_master:
                row_dict["product_image_url"] = prod_link
                row_dict["variant_image_url"] = var_link
                row_dict["image_url"] = prod_link or var_link
                row_dict["search_status"] = row_res.status

            rows_data.append(row_dict)

        result_df = pd.DataFrame(rows_data)

        if job.is_product_master:
            ordered_cols = [c for c in PRODUCT_MASTER_30_COLUMNS if c in result_df.columns]
            extra_cols = [c for c in result_df.columns if c not in PRODUCT_MASTER_30_COLUMNS]
            result_df = result_df[ordered_cols + extra_cols]

        base_name = job.filename.rsplit(".", 1)[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if format_type.lower() == "xlsx" or format_type.lower() == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Product Master")
            output.seek(0)
            filename = f"{base_name}_with_images_{timestamp}.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return output.getvalue(), filename, media_type
        else:
            csv_str = result_df.to_csv(index=False, encoding="utf-8-sig")
            filename = f"{base_name}_with_images_{timestamp}.csv"
            media_type = "text/csv; charset=utf-8"
            return csv_str.encode("utf-8-sig"), filename, media_type

job_manager = JobManager()
