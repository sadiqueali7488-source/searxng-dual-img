import asyncio
import time
import uuid
import io
import re
import pandas as pd
from typing import Dict, Any, List, Optional
import httpx
from models import SearchConfig, JobStatus, RowResult, ImageResult
from searxng_client import search_searxng_image, sanitize_search_query, is_query_multipack, SAME_PACK_SIMILARITY
import job_store
try:
    import vision_engine
except Exception:
    vision_engine = None


def parse_table(content: bytes, filename: str) -> pd.DataFrame:
    """Reads an uploaded .xlsx / .xls / .csv as text columns (so '9AM' never becomes '9:00 AM')."""
    ext = (filename or "").lower().split(".")[-1]
    if ext in ["xlsx", "xls"]:
        df = pd.read_excel(io.BytesIO(content), dtype=str)
    elif ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1", dtype=str)
    else:
        raise ValueError("Only .csv, .xlsx, and .xls files are supported.")
    df = df.fillna("")
    # Text saved twice with the wrong encoding ("EldÃ¶re" -> "Eldöre"): repaired so search + OCR match
    for col in df.columns:
        mask = df[col].str.contains(_MOJIBAKE_MARK, regex=True, na=False)
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].map(fix_mojibake)
    return df

_MOJIBAKE_MARK = r"Ã.|Â.|â[-¿‘-›€]"

def fix_mojibake(text: str) -> str:
    """'EldÃ¶re' -> 'Eldöre'; text that isn't double-encoded is returned unchanged."""
    for enc in ("cp1252", "latin1"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            return fixed if fixed != text else text
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text

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


def _set_image(row: RowResult, target: str, img: ImageResult, alternatives: Optional[List[ImageResult]] = None):
    url = clean_image_link(img.img_src)
    thumb = clean_image_link(img.thumbnail_src) or url
    if target == "variant":
        row.variant_image_url, row.variant_thumbnail_url = url, thumb
        row.variant_resolution, row.variant_engine = img.resolution, img.engine
        if alternatives is not None:
            row.variant_alternatives = alternatives
    else:
        row.product_image_url, row.product_thumbnail_url = url, thumb
        row.product_resolution, row.product_engine = img.resolution, img.engine
        if alternatives is not None:
            row.product_alternatives = alternatives


def apply_search_results(row: RowResult, res_p: Optional[Dict[str, Any]], res_v: Optional[Dict[str, Any]]):
    """
    Writes both searches into the row, then reconciles them:
      1. product and variant image must show the same pack (vision embedding similarity);
         on a mismatch a matching verified variant alternative is used, else the variant is flagged LOW
      2. variant not found -> the product image is used for the variant (variant_source=PRODUCT_IMAGE)
      3. product not found -> the variant image is used for the product (product_source=VARIANT_IMAGE)
    Last-tier picks are flagged: variant_source=NEAREST_SIZE (closest size the company makes),
    *_source=SIMILAR (vision-confirmed, OCR could not read the name). A SIMILAR pick never replaces a
    verified image of the other search: the verified one is borrowed instead.
    """
    tier_p = (res_p or {}).get("match") if (res_p or {}).get("status") == "SUCCESS" else None
    tier_v = (res_v or {}).get("match") if (res_v or {}).get("status") == "SUCCESS" else None
    # a SIMILAR variant is never better than the product's own image (same pack on both sides)
    if tier_v == "relaxed" and tier_p:
        res_v = {**res_v, "status": "NOT_FOUND", "best_image": None}     # -> variant borrows the verified product image
        tier_v = None
    if tier_p == "relaxed" and tier_v and tier_v != "relaxed":
        res_p = {**res_p, "status": "NOT_FOUND", "best_image": None}     # -> product borrows the verified variant image
        tier_p = None
    if res_p is not None:
        row.product_status = res_p["status"]
        row.product_source = ""
        if res_p["status"] == "SUCCESS" and res_p.get("best_image"):
            _set_image(row, "product", res_p["best_image"], res_p.get("alternatives", []))
            row.product_confidence = res_p.get("confidence") or "MEDIUM"
            row.product_source = "SIMILAR" if tier_p == "relaxed" else ""
        else:
            row.product_image_url = row.product_thumbnail_url = ""
            row.product_confidence = ""
    if res_v is not None:
        row.variant_status = res_v["status"]
        row.variant_source = ""
        if res_v["status"] == "SUCCESS" and res_v.get("best_image"):
            _set_image(row, "variant", res_v["best_image"], res_v.get("alternatives", []))
            row.variant_confidence = res_v.get("confidence") or "MEDIUM"
            row.variant_source = {"nearest": "NEAREST_SIZE", "relaxed": "SIMILAR"}.get(tier_v, "")
        else:
            row.variant_image_url = row.variant_thumbnail_url = ""
            row.variant_confidence = ""

    # 1. Same pack check (only when both searches ran now and both have vision embeddings)
    if res_p and res_v and row.product_image_url and row.variant_image_url and vision_engine is not None:
        p_emb, v_emb = res_p.get("embedding"), res_v.get("embedding")
        sim = vision_engine.similarity(p_emb, v_emb)
        if sim is not None and sim < SAME_PACK_SIMILARITY:
            alt_embs = res_v.get("alt_embeddings") or {}
            best_alt, best_sim = None, sim
            for alt in res_v.get("alternatives", []):
                s = vision_engine.similarity(p_emb, alt_embs.get(alt.img_src))
                if s is not None and s >= SAME_PACK_SIMILARITY and s > best_sim:
                    best_alt, best_sim = alt, s
            if best_alt is not None:
                others = [a for a in row.variant_alternatives if a.img_src != best_alt.img_src]
                _set_image(row, "variant", best_alt, others)
                row.variant_confidence = "MEDIUM"
                row.variant_source = ""
            elif row.product_confidence == "HIGH" and row.variant_confidence != "HIGH":
                # The variant shows a different pack (e.g. Bansi Rava for Besan) and the product image is
                # verified: a wrong product must never stay, so the verified product image is used instead
                others = row.variant_alternatives
                _set_image(row, "variant", ImageResult(img_src=row.product_image_url, thumbnail_src=row.product_thumbnail_url,
                                                       resolution=row.product_resolution, engine=row.product_engine), others)
                row.variant_source, row.variant_confidence = "PRODUCT_IMAGE", "LOW"
            else:
                row.variant_confidence = "LOW"

    # 2./3. Borrow the other image when one search found nothing (same product; flagged in the export)
    if row.product_image_url and not row.variant_image_url:
        _set_image(row, "variant", ImageResult(img_src=row.product_image_url, thumbnail_src=row.product_thumbnail_url,
                                               resolution=row.product_resolution, engine=row.product_engine))
        row.variant_status, row.variant_source, row.variant_confidence = "SUCCESS", "PRODUCT_IMAGE", "LOW"
    elif row.variant_image_url and not row.product_image_url:
        _set_image(row, "product", ImageResult(img_src=row.variant_image_url, thumbnail_src=row.variant_thumbnail_url,
                                               resolution=row.variant_resolution, engine=row.variant_engine))
        row.product_status, row.product_source, row.product_confidence = "SUCCESS", "VARIANT_IMAGE", "MEDIUM"

    if row.product_image_url and row.variant_image_url:
        row.status = "SUCCESS"
    elif "ERROR" in ((res_p or {}).get("status"), (res_v or {}).get("status")):
        row.status = "ERROR"
        row.error_message = (res_p or {}).get("error_message") or (res_v or {}).get("error_message")
    else:
        row.status = "NOT_FOUND"
    row.image_url = row.product_image_url or row.variant_image_url
    row.thumbnail_url = row.product_thumbnail_url or row.variant_thumbnail_url


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

    def persist_row(self, row: RowResult):
        """Every finished row goes to disk at once, so a crash or restart loses nothing."""
        try:
            job_store.save_row(self.job_id, row.row_index, row.model_dump_json())
        except Exception:
            pass

    def persist_status(self):
        try:
            job_store.save_status(self.job_id, self.status.model_dump())
        except Exception:
            pass

    def pause(self):
        self.is_paused = True
        self.pause_event.clear()
        self.status.state = "paused"
        self.persist_status()

    def resume(self):
        self.is_paused = False
        self.pause_event.set()
        self.status.state = "running"
        self.persist_status()

    def cancel(self):
        self.is_cancelled = True
        self.pause_event.set()
        self.status.state = "cancelled"
        self.persist_status()
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

    def _recount_status(self, before: str, after: str):
        """Keeps the job's found / not found / error counters right after a single row changes."""
        names = {"SUCCESS": "success_count", "NOT_FOUND": "not_found_count", "ERROR": "error_count"}
        if before == after:
            return
        if before in names:
            setattr(self.status, names[before], max(0, getattr(self.status, names[before]) - 1))
        if after in names:
            setattr(self.status, names[after], getattr(self.status, names[after]) + 1)
        self.persist_status()

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
            res_p = res_v = None
            before_status = row.status

            # 1. Search Product Image (Without Size)
            if target in ["both", "product"]:
                p_query = custom_query if (target == "product" and custom_query) else row.product_query
                old_p = (row.product_image_url, row.product_thumbnail_url)
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
                    language=language,
                    deep=True,
                    fresh=True,
                    exclude=list(old_p)
                )

            # 2. Search Variant Image (With Size)
            if target in ["both", "variant"]:
                v_query = custom_query if (target == "variant" and custom_query) else row.variant_query
                old_v = (row.variant_image_url, row.variant_thumbnail_url)
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
                    language=language,
                    deep=True,
                    fresh=True,
                    exclude=list(old_v)
                )

            # Retry looks for a different image than the current one
            row.retry_note = ""
            kept = []
            prev = {}
            if res_p is not None and res_p["status"] != "SUCCESS" and old_p[0]:
                prev["product"] = (old_p, row.product_confidence, row.product_source)
                kept.append("product")
            if res_v is not None and res_v["status"] != "SUCCESS" and old_v[0]:
                prev["variant"] = (old_v, row.variant_confidence, row.variant_source)
                kept.append("variant")
            apply_search_results(row, res_p, res_v)
            # The other image is borrowed when one is still missing (variant <- product image); the old image
            # comes back only when nothing at all was found, so the row is never wiped by a retry
            notes = []
            for side in kept:
                other = "variant" if side == "product" else "product"
                if getattr(row, f"{side}_image_url"):
                    notes.append(f"No other verified {side} image found; using the {other} image")
                    continue
                (url, thumb), conf, src = prev[side]
                setattr(row, f"{side}_image_url", url)
                setattr(row, f"{side}_thumbnail_url", thumb)
                setattr(row, f"{side}_confidence", conf)
                setattr(row, f"{side}_source", src)
                setattr(row, f"{side}_status", "SUCCESS")
                notes.append(f"No other verified {side} image found; kept the current one")
            row.retry_note = "; ".join(notes)
            if row.product_image_url and row.variant_image_url:
                row.status, row.error_message = "SUCCESS", None
            row.image_url = row.product_image_url or row.variant_image_url
            row.thumbnail_url = row.product_thumbnail_url or row.variant_thumbnail_url
            self._recount_status(before_status, row.status)
            await self.broadcast("row_update", row.model_dump())
            await self.broadcast("status_update", self.status.model_dump())
            self.persist_row(row)
            return row
        return None


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, SearchJob] = {}

    def create_job_from_dataframe(self, df: pd.DataFrame, filename: str, content: Optional[bytes] = None,
                                  job_id: Optional[str] = None) -> SearchJob:
        job_id = job_id or str(uuid.uuid4())
        job = SearchJob(job_id=job_id, df=df, filename=filename)
        self.jobs[job_id] = job
        if content is not None:
            job_store.save_upload(job_id, filename, content)
        return job

    def get_job(self, job_id: str) -> Optional[SearchJob]:
        return self.jobs.get(job_id)

    async def restore_jobs(self):
        """On server start: reload stored jobs; jobs that were running continue where they stopped."""
        for stored in job_store.load_jobs():
            try:
                df = parse_table(stored["file"], stored["filename"])
                job = self.create_job_from_dataframe(df, stored["filename"], job_id=stored["job_id"])
                config = SearchConfig(**stored["config"])
                was_running = (stored.get("status") or {}).get("state") in ("running", "paused", None)
                await self.start_job(job.job_id, config, resume_rows=stored["rows"], run=was_running)
            except Exception:
                continue

    async def start_job(self, job_id: str, config: SearchConfig, resume_rows: Optional[List[Dict[str, Any]]] = None,
                        run: bool = True):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if resume_rows is None:
            job_store.save_config(job_id, config.model_dump())

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

        if resume_rows:
            # Resume: rows finished before the restart keep their results and are not searched again
            by_index = {r.get("row_index"): r for r in resume_rows}
            for i, row in enumerate(job.row_results):
                stored = by_index.get(row.row_index)
                if stored and stored.get("status") not in ("PENDING", "SEARCHING", None):
                    job.row_results[i] = RowResult(**stored)
            done = [r for r in job.row_results if r.status not in ("PENDING", "SEARCHING")]
            job.status.processed_rows = len(done)
            job.status.success_count = sum(1 for r in done if r.status == "SUCCESS")
            job.status.error_count = sum(1 for r in done if r.status == "ERROR")
            job.status.not_found_count = len(done) - job.status.success_count - job.status.error_count
            job.status.progress_percent = round(100 * len(done) / max(1, job.status.total_rows), 1)

        if not run:
            job.status.state = "completed"
            return job
        job.persist_status()
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
                job.persist_status()

        # Each row runs two searches at once, each downloading up to 4 images in parallel
        limits = httpx.Limits(max_keepalive_connections=config.concurrency * 4, max_connections=config.concurrency * 10)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            target_rows = specific_rows if specific_rows is not None else job.row_results

            async def process_single_row(row_res: RowResult, deep: bool = False):
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
                res_p, res_v = await asyncio.gather(*(
                    search_searxng_image(
                        client=client,
                        base_url=config.searxng_url,
                        query=row_res.variant_query if is_var else row_res.product_query,
                        brand=b_val,
                        product_name=p_val,
                        variant_label=v_val,
                        is_variant=is_var,
                        safesearch=config.safesearch,
                        time_range=config.time_range,
                        language=config.language,
                        deep=deep
                    ) for is_var in (False, True)
                ))
                apply_search_results(row_res, res_p, res_v)

                if row_res.status == "SUCCESS":
                    job.status.success_count += 1
                elif row_res.status == "ERROR":
                    job.status.error_count += 1
                else:
                    job.status.not_found_count += 1

                job.status.processed_rows += 1
                elapsed = max(0.1, time.time() - job.start_time)
                job.status.elapsed_seconds = round(elapsed, 1)
                speed = job.status.processed_rows / elapsed
                job.status.speed_rows_per_sec = round(speed, 2)
                remaining = job.status.total_rows - job.status.processed_rows
                job.status.eta_seconds = round(remaining / speed, 1) if speed > 0 else 0
                job.status.progress_percent = round((job.status.processed_rows / job.status.total_rows) * 100, 1)

                await job.broadcast("row_update", row_res.model_dump())
                job.persist_row(row_res)
                await push_status()

            async def run_pass(rows: List[RowResult], deep: bool):
                # Fixed pool of workers pulling rows from a queue (not one task per row for 1 lakh rows)
                queue: asyncio.Queue = asyncio.Queue()
                for row in rows:
                    queue.put_nowait(row)

                async def worker():
                    while not job.is_cancelled:
                        try:
                            row_res = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        try:
                            await process_single_row(row_res, deep)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            row_res.status = "ERROR"
                            row_res.error_message = str(e)
                            job.status.error_count += 1
                            job.status.processed_rows += 1
                            await job.broadcast("row_update", row_res.model_dump())
                            job.persist_row(row_res)

                await asyncio.gather(*(worker() for _ in range(config.concurrency)))

            try:
                # Rows already finished (a resumed job) are not searched again
                pending = [r for r in target_rows if r.status in ("PENDING", "SEARCHING") or specific_rows is not None]
                await run_pass(pending, deep=specific_rows is not None)

                # Automatic deep retry of every row still without both images (no button needed)
                if not job.is_cancelled and specific_rows is None:
                    failed = [r for r in target_rows if r.status in ("NOT_FOUND", "ERROR")]
                    if failed:
                        for r in failed:
                            if r.status == "ERROR":
                                job.status.error_count = max(0, job.status.error_count - 1)
                            else:
                                job.status.not_found_count = max(0, job.status.not_found_count - 1)
                            job.status.processed_rows = max(0, job.status.processed_rows - 1)
                        job.status.current_query = f"Auto-retry: {len(failed)} rows with deeper search"
                        await push_status()
                        await run_pass(failed, deep=True)
            except asyncio.CancelledError:
                job.status.state = "cancelled"
            except Exception as e:
                job.status.state = "error"

        if not job.is_cancelled:
            job.status.state = "completed"
            job.status.progress_percent = 100.0

        job.persist_status()
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
            # Confidence / source stay in the app (badges), not in the downloaded file

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
