from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SearchConfig(BaseModel):
    searxng_url: str = Field(default="http://localhost:8888")
    query_column: str = "product_name"
    additional_columns: List[str] = Field(default_factory=list)
    query_prefix: str = ""
    query_suffix: str = ""
    concurrency: int = Field(default=6, ge=1, le=30)
    delay_ms: int = Field(default=100, ge=0, le=5000)
    safesearch: int = Field(default=0, ge=0, le=2)
    time_range: str = ""
    language: str = "auto"
    limit_rows: Optional[int] = None
    output_filename: Optional[str] = None

class ImageResult(BaseModel):
    title: str = ""
    img_src: str = ""
    thumbnail_src: str = ""
    url: str = ""
    resolution: str = ""
    engine: str = ""

class RowResult(BaseModel):
    row_index: int
    original_data: Dict[str, Any]
    
    # 1. Product Image (WITHOUT size/variant)
    product_query: str = ""
    product_image_url: str = ""
    product_thumbnail_url: str = ""
    product_status: str = "PENDING"
    product_resolution: str = ""
    product_engine: str = ""
    product_alternatives: List[ImageResult] = Field(default_factory=list)

    # 2. Variant Image (WITH size/variant)
    variant_query: str = ""
    variant_image_url: str = ""
    variant_thumbnail_url: str = ""
    variant_status: str = "PENDING"
    variant_resolution: str = ""
    variant_engine: str = ""
    variant_alternatives: List[ImageResult] = Field(default_factory=list)

    # Quality of the picked images: HIGH / MEDIUM / LOW ("" until searched)
    product_confidence: str = ""
    variant_confidence: str = ""
    # Where the image came from: "" (own search), "VARIANT_IMAGE" / "PRODUCT_IMAGE" (borrowed from the other search)
    product_source: str = ""
    variant_source: str = ""
    # Message from the last Retry (e.g. no other verified image was found)
    retry_note: str = ""

    # Backward compatibility helpers
    query: str = ""
    status: str = "PENDING"  # PENDING, SEARCHING, SUCCESS, PARTIAL, NOT_FOUND, ERROR
    image_url: str = ""
    thumbnail_url: str = ""
    image_title: str = ""
    image_resolution: str = ""
    engine: str = ""
    error_message: Optional[str] = None
    alternative_images: List[ImageResult] = Field(default_factory=list)

class JobStatus(BaseModel):
    job_id: str
    filename: str
    state: str = "idle"  # idle, running, paused, completed, cancelled, error
    total_rows: int = 0
    processed_rows: int = 0
    success_count: int = 0
    not_found_count: int = 0
    error_count: int = 0
    current_query: str = ""
    speed_rows_per_sec: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    progress_percent: float = 0.0
