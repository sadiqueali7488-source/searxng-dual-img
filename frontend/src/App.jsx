import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import axios from 'axios';
import {
  Upload, Search, FileSpreadsheet, Download, Play, Pause, Square,
  CheckCircle2, XCircle, AlertCircle, RefreshCw, Eye, ExternalLink,
  Copy, Check, Settings, Sparkles, Image as ImageIcon, ArrowRight,
  Database, Zap, Clock, ShieldCheck, Layers, ChevronDown, ChevronUp,
  X, Filter, RotateCw, Tag, ShoppingBag, DollarSign, ListFilter,
  Sliders, Info, Edit3, ImagePlus, AlertTriangle,
  ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight
} from 'lucide-react';

// Memoized individual TableRow component for maximum 60fps performance
const TableRow = React.memo(function TableRow({
  row,
  index,
  columns,
  cyclingKey,
  onOpenDetail,
  onCycleImage
}) {
  return (
    <tr
      onClick={() => onOpenDetail(row)}
      className="hover:bg-slate-800/50 transition-colors group cursor-pointer border-b border-slate-800/40"
    >
      <td className="p-3 text-center text-slate-500 font-mono text-[11px]">
        {row.row_index + 1}
      </td>

      {/* 1. Product Image Thumbnail (Without Size) */}
      <td className="p-2.5 text-center border-r border-slate-800/80 bg-indigo-950/10" onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-col items-center gap-1.5">
          {row.product_image_url ? (
            <div
              onClick={() => onOpenDetail(row)}
              className="w-16 h-16 rounded-lg bg-slate-950 border border-indigo-500/40 overflow-hidden cursor-pointer hover:border-indigo-400 transition flex items-center justify-center relative shadow"
              title="Product Image (Without Size) - Click to zoom"
            >
              <img
                src={row.product_thumbnail_url || row.product_image_url}
                alt="Product"
                loading="lazy"
                decoding="async"
                className="w-full h-full object-cover hover:scale-110 transition duration-150"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://via.placeholder.com/80?text=No+Img';
                }}
              />
            </div>
          ) : (
            <div className="w-16 h-16 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-600">
              <ImageIcon className="w-5 h-5" />
            </div>
          )}

          {row.product_alternatives && row.product_alternatives.length > 0 && (
            <button
              onClick={(e) => onCycleImage(row.row_index, 'product', e)}
              disabled={cyclingKey === `${row.row_index}-product`}
              className="bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 px-1.5 py-0.5 rounded text-[9px] font-sans font-semibold transition cursor-pointer active:scale-95 flex items-center space-x-1"
              title="Cycle Product Image to next alternative"
            >
              <RotateCw className={`w-2.5 h-2.5 ${cyclingKey === `${row.row_index}-product` ? 'animate-spin' : ''}`} />
              <span>Next ({row.product_alternatives.length})</span>
            </button>
          )}
        </div>
      </td>

      {/* 2. Variant Image Thumbnail (With Size) */}
      <td className="p-2.5 text-center border-r border-slate-800/80 bg-cyan-950/10" onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-col items-center gap-1.5">
          {row.variant_image_url ? (
            <div
              onClick={() => onOpenDetail(row)}
              className="w-16 h-16 rounded-lg bg-slate-950 border border-cyan-500/40 overflow-hidden cursor-pointer hover:border-cyan-400 transition flex items-center justify-center relative shadow"
              title="Variant Image (With Size) - Click to zoom"
            >
              <img
                src={row.variant_thumbnail_url || row.variant_image_url}
                alt="Variant"
                loading="lazy"
                decoding="async"
                className="w-full h-full object-cover hover:scale-110 transition duration-150"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://via.placeholder.com/80?text=No+Img';
                }}
              />
            </div>
          ) : (
            <div className="w-16 h-16 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-600">
              <ImageIcon className="w-5 h-5" />
            </div>
          )}

          {row.variant_alternatives && row.variant_alternatives.length > 0 && (
            <button
              onClick={(e) => onCycleImage(row.row_index, 'variant', e)}
              disabled={cyclingKey === `${row.row_index}-variant`}
              className="bg-cyan-600/20 hover:bg-cyan-600 text-cyan-300 hover:text-white border border-cyan-500/40 px-1.5 py-0.5 rounded text-[9px] font-sans font-semibold transition cursor-pointer active:scale-95 flex items-center space-x-1"
              title="Cycle Variant Image to next alternative"
            >
              <RotateCw className={`w-2.5 h-2.5 ${cyclingKey === `${row.row_index}-variant` ? 'animate-spin' : ''}`} />
              <span>Next ({row.variant_alternatives.length})</span>
            </button>
          )}
        </div>
      </td>

      {/* Product Name & Queries */}
      <td className="p-3">
        <div className="font-sans font-medium text-slate-200 line-clamp-2 max-w-[240px]">
          {row.original_data?.product_name || row.query}
        </div>
        <div className="text-[10px] text-indigo-400 font-mono mt-1 truncate max-w-[240px]">
          🔹 No size: {row.product_query}
        </div>
        <div className="text-[10px] text-cyan-400 font-mono mt-0.5 truncate max-w-[240px]">
          🔸 With size: {row.variant_query}
        </div>
      </td>

      {/* Status */}
      <td className="p-3 whitespace-nowrap">
        {row.status === 'SUCCESS' && (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3 h-3" />
            <span>Found</span>
          </span>
        )}
        {row.status === 'PARTIAL' && (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <CheckCircle2 className="w-3 h-3" />
            <span>Partial</span>
          </span>
        )}
        {row.status === 'SEARCHING' && (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 animate-pulse">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span>Searching</span>
          </span>
        )}
        {row.status === 'NOT_FOUND' && (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <AlertCircle className="w-3 h-3" />
            <span>Not Found</span>
          </span>
        )}
        {row.status === 'PENDING' && (
          <span className="text-[10px] text-slate-500 font-sans">
            Queued
          </span>
        )}
      </td>

      {/* Prices with ₹ */}
      <td className="p-3 font-mono whitespace-nowrap">
        <div className="text-emerald-400 font-bold">{row.original_data?.sale_price || '—'}</div>
        <div className="text-slate-500 text-[10px] line-through">{row.original_data?.mrp || ''}</div>
      </td>

      {/* Actions */}
      <td className="p-3 text-center whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={() => onOpenDetail(row)}
          className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-2.5 py-1 rounded text-[10px] font-sans font-medium transition cursor-pointer"
        >
          View Details
        </button>
      </td>

      {/* Original columns */}
      {columns.map((col) => (
        <td key={col} className="p-3 text-slate-300 font-sans whitespace-nowrap max-w-[180px] truncate">
          {row.original_data?.[col] !== undefined ? String(row.original_data[col]) : ''}
        </td>
      ))}
    </tr>
  );
});

export default function App() {
  const [searxngUrl, setSearxngUrl] = useState('http://localhost:8888');
  const [searxngStatus, setSearxngStatus] = useState({ connected: false, loading: true, message: '' });
  const [showSettings, setShowSettings] = useState(false);

  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadData, setUploadData] = useState(null);

  const [queryColumn, setQueryColumn] = useState('');
  const [additionalColumns, setAdditionalColumns] = useState([]);
  const [queryPrefix, setQueryPrefix] = useState('');
  const [querySuffix, setQuerySuffix] = useState('');
  const [concurrency, setConcurrency] = useState(12);
  const [delayMs, setDelayMs] = useState(100);
  const [safesearch, setSafesearch] = useState(0);
  const [limitRows, setLimitRows] = useState(0);

  const [isProcessing, setIsProcessing] = useState(false);
  const [isRetryingNotFound, setIsRetryingNotFound] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);
  const [rows, setRows] = useState([]);
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [tableSearch, setTableSearch] = useState('');

  // Pagination states for super-fast 60fps rendering
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  const [detailModalRow, setDetailModalRow] = useState(null);
  const [customProductQuery, setCustomProductQuery] = useState('');
  const [customVariantQuery, setCustomVariantQuery] = useState('');
  const [retryingTarget, setRetryingTarget] = useState(null);
  const [cyclingKey, setCyclingKey] = useState(null);
  const [copiedUrl, setCopiedUrl] = useState('');

  // High-performance micro-batch update refs
  const eventSourceRef = useRef(null);
  const updateBufferRef = useRef(new Map());
  const batchTimerRef = useRef(null);

  useEffect(() => {
    checkSearxngConnection(searxngUrl);
    return () => {
      if (batchTimerRef.current) clearTimeout(batchTimerRef.current);
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, []);

  const checkSearxngConnection = async (url) => {
    setSearxngStatus(prev => ({ ...prev, loading: true }));
    try {
      const res = await axios.get(`/api/test-searxng?url=${encodeURIComponent(url)}`);
      setSearxngStatus({
        connected: res.data.ok,
        loading: false,
        message: res.data.message || (res.data.ok ? 'Connected' : 'Connection failed')
      });
    } catch (err) {
      setSearxngStatus({
        connected: false,
        loading: false,
        message: 'Could not connect to SearXNG: ' + err.message
      });
    }
  };

  const handleFileUpload = async (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadData(res.data);
      const cols = res.data.columns;
      const likelyCol = cols.find(c =>
        /^(product_name|product|name|item|title|p_name)$/i.test(c)
      ) || cols.find(c => /product|name|item|title/i.test(c)) || cols[0];
      setQueryColumn(likelyCol || '');

      const autoAdditional = [];
      const brandCol = cols.find(c => /^(brand_name|brand)$/i.test(c));
      const variantCol = cols.find(c => /^(variant_label|variant|pack_size)$/i.test(c));
      if (brandCol && brandCol !== likelyCol) autoAdditional.push(brandCol);
      if (variantCol && variantCol !== likelyCol) autoAdditional.push(variantCol);
      setAdditionalColumns(autoAdditional);

      setRows([]);
      setJobStatus(null);
      setCurrentPage(1);
    } catch (err) {
      alert('Upload failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
    }
  };

  const loadSampleData = async () => {
    try {
      setUploading(true);
      const csvSample = `sku,variant_label,product_name,product_description,product_prescription_required,brand_name,brand_description,brand_image,parent_category_name,parent_category_description,parent_category_icon,parent_category_prescription_required,parent_category_is_featured,category_name,category_description,category_icon,category_prescription_required,category_is_featured,mrp,sale_price,tax_inclusive,barcode,gtin,mpn,color_hex,product_attributes,variant_attributes,variant_axes,product_images,variant_images
AFS-24M-7GRAINMETHI-1KG-NA-001,1kg,24 Mantra Organic 7 Grain Methi Atta,24 Mantra Organic 7 Grain Methi Atta - Multigrain (Atta),FALSE,24 Mantra Organic,,,Grocery,,,FALSE,FALSE,Atta Flour and Sooji,,,FALSE,FALSE,₹130.00,₹130.00,inclusive,379,,2400,,Speciality:Multigrain|Pack Size:1kg|Type:Atta,Pack Size:1kg,Pack Size,https://old-cdn.com/old1.webp,https://old-cdn.com/old2.webp
9AM-CORNFLOUR-500G-002,500g,9AM Cornflour,9AM Cornflour pure starch for cooking,FALSE,9AM,,,Grocery,,,FALSE,FALSE,Flour and Starch,,,FALSE,FALSE,₹120.00,₹110.00,inclusive,380,,2401,,Speciality:Pure|Pack Size:500g|Type:Starch,Pack Size:500g,Pack Size,https://old-cdn.com/old3.webp,https://old-cdn.com/old4.webp
AMUL-BUTTER-500G-003,500g,Amul Pure Salted Butter,Amul Butter Delicious and Pure Dairy,FALSE,Amul,,,Dairy,,,FALSE,FALSE,Butter and Cheese,,,FALSE,FALSE,₹275.00,₹270.00,inclusive,381,,2402,,Speciality:Pure Dairy|Pack Size:500g,Pack Size:500g,Pack Size,https://old-cdn.com/old5.webp,https://old-cdn.com/old6.webp
MAGGI-MASALA-280G-004,280g,Maggi 2-Minute Masala Instant Noodles,Maggi Noodles 280g Pouch,FALSE,Nestle,,,Grocery,,,FALSE,FALSE,Instant Noodles,,,FALSE,FALSE,₹45.00,₹45.00,inclusive,382,,2403,,Speciality:Instant|Pack Size:280g,Pack Size:280g,Pack Size,https://old-cdn.com/old7.webp,https://old-cdn.com/old8.webp`;

      const blob = new Blob([csvSample], { type: 'text/csv' });
      const sampleFile = new File([blob], 'Product_Master_Sample.csv', { type: 'text/csv' });
      await handleFileUpload(sampleFile);
    } catch (e) {
      alert('Error loading sample: ' + e.message);
    } finally {
      setUploading(false);
    }
  };

  // High-performance flush: batches incoming rows every 60ms into a single React render!
  const flushRowBatch = useCallback(() => {
    if (updateBufferRef.current.size === 0) return;
    const batch = Array.from(updateBufferRef.current.values());
    updateBufferRef.current.clear();

    setRows(prevRows => {
      const next = [...prevRows];
      for (const updatedRow of batch) {
        // rows are stored in row_index order, so the index is a direct O(1) lookup
        // (findIndex over 1 lakh rows for every update froze the page)
        const i = updatedRow.row_index;
        const idx = next[i]?.row_index === i ? i : next.findIndex(r => r.row_index === i);
        if (idx >= 0) {
          next[idx] = updatedRow;
        } else {
          next.push(updatedRow);
        }
      }
      return next;
    });

    setDetailModalRow(prev => {
      if (!prev) return prev;
      const found = batch.find(r => r.row_index === prev.row_index);
      return found || prev;
    });
  }, []);

  const handleStartSearch = async () => {
    if (!uploadData?.job_id || !queryColumn) return;

    try {
      setIsProcessing(true);
      setIsPaused(false);
      setCurrentPage(1);

      const config = {
        searxng_url: searxngUrl,
        query_column: queryColumn,
        additional_columns: additionalColumns,
        query_prefix: queryPrefix,
        query_suffix: querySuffix,
        concurrency: Number(concurrency),
        delay_ms: Number(delayMs),
        safesearch: Number(safesearch),
        limit_rows: limitRows > 0 ? Number(limitRows) : null
      };

      await axios.post(`/api/start-search/${uploadData.job_id}`, config);

      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource(`/api/stream-progress/${uploadData.job_id}`);
      eventSourceRef.current = es;

      es.addEventListener('initial_rows', (e) => {
        const initialRows = JSON.parse(e.data);
        setRows(initialRows);
      });

      es.addEventListener('status_update', (e) => {
        const status = JSON.parse(e.data);
        setJobStatus(status);
        if (status.state === 'paused') setIsPaused(true);
        if (status.state === 'running') setIsPaused(false);
        if (status.state === 'completed' || status.state === 'cancelled') {
          setIsProcessing(false);
          setIsRetryingNotFound(false);
        }
      });

      es.addEventListener('row_update', (e) => {
        const updatedRow = JSON.parse(e.data);
        updateBufferRef.current.set(updatedRow.row_index, updatedRow);

        if (!batchTimerRef.current) {
          batchTimerRef.current = setTimeout(() => {
            batchTimerRef.current = null;
            flushRowBatch();
          }, 250);
        }
      });

      es.addEventListener('job_done', () => {
        flushRowBatch();
        setIsProcessing(false);
        setIsRetryingNotFound(false);
        es.close();
      });

      es.onerror = () => {};

    } catch (err) {
      alert('Failed to start search: ' + (err.response?.data?.detail || err.message));
      setIsProcessing(false);
      setIsRetryingNotFound(false);
    }
  };

  const handleRetryNotFound = async () => {
    if (!uploadData?.job_id) return;
    try {
      setIsRetryingNotFound(true);
      setIsProcessing(true);
      const res = await axios.post(`/api/job/${uploadData.job_id}/retry-not-found`);
      if (res.data.count === 0) {
        alert(res.data.message);
        setIsRetryingNotFound(false);
        setIsProcessing(false);
      }
    } catch (err) {
      alert('Retry failed: ' + (err.response?.data?.detail || err.message));
      setIsRetryingNotFound(false);
      setIsProcessing(false);
    }
  };

  const handlePause = async () => {
    if (!uploadData?.job_id) return;
    try {
      await axios.post(`/api/job/${uploadData.job_id}/pause`);
      setIsPaused(true);
    } catch (err) {
      console.error(err);
    }
  };

  const handleResume = async () => {
    if (!uploadData?.job_id) return;
    try {
      await axios.post(`/api/job/${uploadData.job_id}/resume`);
      setIsPaused(false);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCancel = async () => {
    if (!uploadData?.job_id) return;
    try {
      await axios.post(`/api/job/${uploadData.job_id}/cancel`);
      setIsProcessing(false);
      setIsRetryingNotFound(false);
      setIsPaused(false);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDownload = (format) => {
    if (!uploadData?.job_id) return;
    window.location.href = `/api/download/${uploadData.job_id}?format=${format}`;
  };

  const handleSelectAlternativeImage = useCallback(async (rowIndex, imageObj, target = 'product') => {
    const cleanUrl = imageObj.img_src || '';
    const cleanThumb = imageObj.thumbnail_src || cleanUrl;

    // 1. Instantly and optimistically update React state (0ms latency!)
    setRows((prev) => {
      const next = [...prev];
      if (next[rowIndex]) {
        const r = { ...next[rowIndex] };
        if (target === 'variant') {
          r.variant_image_url = cleanUrl;
          r.variant_thumbnail_url = cleanThumb;
          r.variant_resolution = imageObj.resolution || r.variant_resolution;
          r.variant_engine = imageObj.engine || r.variant_engine;
          r.variant_status = 'SUCCESS';
        } else {
          r.product_image_url = cleanUrl;
          r.product_thumbnail_url = cleanThumb;
          r.product_resolution = imageObj.resolution || r.product_resolution;
          r.product_engine = imageObj.engine || r.product_engine;
          r.product_status = 'SUCCESS';
        }
        r.image_url = r.product_image_url || r.variant_image_url;
        r.thumbnail_url = r.product_thumbnail_url || r.variant_thumbnail_url;
        r.status = 'SUCCESS';
        next[rowIndex] = r;
      }
      return next;
    });

    setDetailModalRow((prev) => {
      if (prev && prev.row_index === rowIndex) {
        const r = { ...prev };
        if (target === 'variant') {
          r.variant_image_url = cleanUrl;
          r.variant_thumbnail_url = cleanThumb;
          r.variant_resolution = imageObj.resolution || r.variant_resolution;
          r.variant_engine = imageObj.engine || r.variant_engine;
          r.variant_status = 'SUCCESS';
        } else {
          r.product_image_url = cleanUrl;
          r.product_thumbnail_url = cleanThumb;
          r.product_resolution = imageObj.resolution || r.product_resolution;
          r.product_engine = imageObj.engine || r.product_engine;
          r.product_status = 'SUCCESS';
        }
        r.image_url = r.product_image_url || r.variant_image_url;
        r.thumbnail_url = r.product_thumbnail_url || r.variant_thumbnail_url;
        r.status = 'SUCCESS';
        return r;
      }
      return prev;
    });

    // 2. Sync to backend in background if job exists
    if (!uploadData?.job_id) return;
    try {
      const formData = new FormData();
      formData.append('row_index', rowIndex);
      formData.append('new_image_json', JSON.stringify(imageObj));
      formData.append('target', target);

      await axios.post(`/api/job/${uploadData.job_id}/swap-image`, formData);
    } catch (err) {
      // If backend restarted or session reset, image is already cleanly swapped in UI
      console.warn('Backend sync note (image swapped locally):', err.message);
    }
  }, [uploadData]);

  const handleCycleImage = useCallback(async (rowIndex, target = 'product', e) => {
    if (e) e.stopPropagation();

    // 1. Instantly cycle from local alternatives list
    setRows((prev) => {
      const next = [...prev];
      if (next[rowIndex]) {
        const r = { ...next[rowIndex] };
        const alts = target === 'variant' ? [...(r.variant_alternatives || [])] : [...(r.product_alternatives || [])];
        if (alts.length > 0) {
          const nextImg = alts.shift();
          const currentImg = {
            title: r.original_data?.product_name || '',
            img_src: target === 'variant' ? r.variant_image_url : r.product_image_url,
            thumbnail_src: target === 'variant' ? r.variant_thumbnail_url : r.product_thumbnail_url,
            resolution: target === 'variant' ? r.variant_resolution : r.product_resolution,
            engine: target === 'variant' ? r.variant_engine : r.product_engine
          };
          if (currentImg.img_src) alts.push(currentImg);

          if (target === 'variant') {
            r.variant_image_url = nextImg.img_src;
            r.variant_thumbnail_url = nextImg.thumbnail_src || nextImg.img_src;
            r.variant_resolution = nextImg.resolution || r.variant_resolution;
            r.variant_engine = nextImg.engine || r.variant_engine;
            r.variant_alternatives = alts;
          } else {
            r.product_image_url = nextImg.img_src;
            r.product_thumbnail_url = nextImg.thumbnail_src || nextImg.img_src;
            r.product_resolution = nextImg.resolution || r.product_resolution;
            r.product_engine = nextImg.engine || r.product_engine;
            r.product_alternatives = alts;
          }
          r.image_url = r.product_image_url || r.variant_image_url;
          r.thumbnail_url = r.product_thumbnail_url || r.variant_thumbnail_url;
          next[rowIndex] = r;
        }
      }
      return next;
    });

    setDetailModalRow((prev) => {
      if (prev && prev.row_index === rowIndex) {
        const r = { ...prev };
        const alts = target === 'variant' ? [...(r.variant_alternatives || [])] : [...(r.product_alternatives || [])];
        if (alts.length > 0) {
          const nextImg = alts.shift();
          const currentImg = {
            title: r.original_data?.product_name || '',
            img_src: target === 'variant' ? r.variant_image_url : r.product_image_url,
            thumbnail_src: target === 'variant' ? r.variant_thumbnail_url : r.product_thumbnail_url,
            resolution: target === 'variant' ? r.variant_resolution : r.product_resolution,
            engine: target === 'variant' ? r.variant_engine : r.product_engine
          };
          if (currentImg.img_src) alts.push(currentImg);

          if (target === 'variant') {
            r.variant_image_url = nextImg.img_src;
            r.variant_thumbnail_url = nextImg.thumbnail_src || nextImg.img_src;
            r.variant_resolution = nextImg.resolution || r.variant_resolution;
            r.variant_engine = nextImg.engine || r.variant_engine;
            r.variant_alternatives = alts;
          } else {
            r.product_image_url = nextImg.img_src;
            r.product_thumbnail_url = nextImg.thumbnail_src || nextImg.img_src;
            r.product_resolution = nextImg.resolution || r.product_resolution;
            r.product_engine = nextImg.engine || r.product_engine;
            r.product_alternatives = alts;
          }
          r.image_url = r.product_image_url || r.variant_image_url;
          r.thumbnail_url = r.product_thumbnail_url || r.variant_thumbnail_url;
          return r;
        }
      }
      return prev;
    });

    // 2. Sync to backend
    if (!uploadData?.job_id) return;
    const key = `${rowIndex}-${target}`;
    setCyclingKey(key);
    try {
      await axios.post(`/api/job/${uploadData.job_id}/cycle-image/${rowIndex}?target=${target}`);
    } catch (err) {
      console.warn('Backend cycle sync note:', err.message);
    } finally {
      setCyclingKey(null);
    }
  }, [uploadData]);

  const handleRetrySingle = async (rowIndex, customQuery, target = 'both') => {
    if (!uploadData?.job_id) return;
    setRetryingTarget(target);

    try {
      const formData = new FormData();
      if (customQuery) {
        formData.append('custom_query', customQuery);
      }
      formData.append('target', target);
      const res = await axios.post(`/api/job/${uploadData.job_id}/retry-row/${rowIndex}`, formData);
      setRows((prev) => {
        const next = [...prev];
        next[rowIndex] = res.data.row;
        return next;
      });
      setDetailModalRow(prev => prev?.row_index === rowIndex ? res.data.row : prev);
    } catch (err) {
      alert('Retry failed: ' + err.message);
    } finally {
      setRetryingTarget(null);
    }
  };

  const openDetailModal = useCallback((row) => {
    setDetailModalRow(row);
    setCustomProductQuery(row.product_query || '');
    setCustomVariantQuery(row.variant_query || '');
  }, []);

  const copyToClipboard = (text, e) => {
    if (e) e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedUrl(text);
    setTimeout(() => setCopiedUrl(''), 2000);
  };

  // High performance memoized row filtering
  const filteredRows = useMemo(() => {
    let list = rows;
    if (filterStatus === 'SUCCESS') list = list.filter(r => r.status === 'SUCCESS');
    else if (filterStatus === 'NOT_FOUND') list = list.filter(r => r.status === 'NOT_FOUND');
    else if (filterStatus === 'ERROR') list = list.filter(r => r.status === 'ERROR');

    const q = tableSearch.trim().toLowerCase();
    if (q) {
      list = list.filter(row => {
        const matchP = row.product_query?.toLowerCase().includes(q);
        const matchV = row.variant_query?.toLowerCase().includes(q);
        const matchOrig = Object.values(row.original_data || {}).some(
          val => String(val).toLowerCase().includes(q)
        );
        return matchP || matchV || matchOrig;
      });
    }
    return list;
  }, [rows, filterStatus, tableSearch]);

  // High performance paginated rows slice
  const totalPages = useMemo(() => {
    if (pageSize === 'ALL') return 1;
    return Math.max(1, Math.ceil(filteredRows.length / Number(pageSize)));
  }, [filteredRows.length, pageSize]);

  const paginatedRows = useMemo(() => {
    if (pageSize === 'ALL') return filteredRows;
    const size = Number(pageSize);
    const start = (currentPage - 1) * size;
    return filteredRows.slice(start, start + size);
  }, [filteredRows, currentPage, pageSize]);

  const getProductPreviewQuery = () => {
    if (!uploadData || !uploadData.preview || uploadData.preview.length === 0) return '';
    const row = uploadData.preview[0];
    const brand = row['brand_name'] || row['brand'] || '';
    let name = row[queryColumn] || '';
    name = name.replace(/\b\d+(\.\d+)?\s*(kg|g|gm|gms|ml|l|ltr|pack\s+of\s+\d+|pcs|pieces)\b/gi, '').trim();
    const parts = [];
    if (queryPrefix) parts.push(queryPrefix);
    if (brand && !name.toLowerCase().includes(brand.toLowerCase())) parts.push(brand);
    if (name) parts.push(name);
    if (querySuffix) parts.push(querySuffix);
    else parts.push('front packaging pouch');
    return parts.filter(Boolean).join(' ').trim();
  };

  const getVariantPreviewQuery = () => {
    if (!uploadData || !uploadData.preview || uploadData.preview.length === 0) return '';
    const row = uploadData.preview[0];
    const brand = row['brand_name'] || row['brand'] || '';
    const name = row[queryColumn] || '';
    const variant = row['variant_label'] || row['variant'] || '';
    const parts = [];
    if (queryPrefix) parts.push(queryPrefix);
    if (brand && !name.toLowerCase().includes(brand.toLowerCase())) parts.push(brand);
    if (name) parts.push(name);
    if (variant && !name.toLowerCase().includes(variant.toLowerCase())) parts.push(variant);
    if (querySuffix) parts.push(querySuffix);
    else parts.push('front packaging pack');
    return parts.filter(Boolean).join(' ').trim();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans select-none antialiased">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-tr from-indigo-500 via-indigo-600 to-cyan-400 p-2 rounded-xl shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
                SearXNG Product Image Finder
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-full tracking-wide">
                Dual Front Packshot (AI OCR)
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              SearXNG Multi-Engine • Google, Bing & Brave • Ultra-Smooth 60fps
            </p>
          </div>
        </div>

        {/* Right Nav controls */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-800/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                searxngStatus.loading
                  ? 'bg-amber-400 animate-pulse'
                  : searxngStatus.connected
                  ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]'
                  : 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'
              }`}
            />
            <span className="text-slate-300 font-mono text-[11px] truncate max-w-[140px]">
              {searxngUrl}
            </span>
            <button
              onClick={() => checkSearxngConnection(searxngUrl)}
              title="Test connection"
              className="text-slate-400 hover:text-white transition cursor-pointer p-0.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${searxngStatus.loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="text-slate-400 hover:text-indigo-400 transition cursor-pointer p-0.5 ml-1"
              title="Settings"
            >
              <Settings className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Flyout */}
      {showSettings && (
        <div className="bg-slate-900 border-b border-slate-800 p-4 animate-in slide-in-from-top duration-200 text-xs">
          <div className="max-w-4xl mx-auto flex items-center space-x-4">
            <label className="text-slate-400 whitespace-nowrap font-medium">SearXNG Base URL:</label>
            <input
              type="text"
              value={searxngUrl}
              onChange={(e) => setSearxngUrl(e.target.value)}
              className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-white font-mono"
            />
            <button
              onClick={() => checkSearxngConnection(searxngUrl)}
              className="bg-indigo-600 hover:bg-indigo-500 px-4 py-1.5 rounded-lg text-white font-semibold transition cursor-pointer"
            >
              Save & Test
            </button>
            <span className={`text-[11px] ${searxngStatus.connected ? 'text-emerald-400' : 'text-rose-400'}`}>
              {searxngStatus.message}
            </span>
          </div>
        </div>
      )}

      {/* Main Container - Full Edge-to-Edge Width */}
      <main className="flex-1 w-full px-4 sm:px-6 lg:px-8 py-5 space-y-5">

        {/* STEP 1 & 2: Top Setup Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* STEP 1: Upload Card */}
          <div className="lg:col-span-5 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                  1
                </div>
                <h2 className="text-sm font-semibold text-white tracking-wide">
                  Upload Product Master (CSV/XLSX)
                </h2>
              </div>

              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
                }}
                className="border-2 border-dashed border-slate-700/80 hover:border-indigo-500/80 bg-slate-950/40 rounded-xl p-6 text-center transition cursor-pointer group"
                onClick={() => document.getElementById('file-upload-input')?.click()}
              >
                <input
                  id="file-upload-input"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                />
                <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition">
                  {uploading ? (
                    <RefreshCw className="w-6 h-6 animate-spin" />
                  ) : (
                    <Upload className="w-6 h-6" />
                  )}
                </div>
                <p className="text-xs font-medium text-slate-300">
                  {file ? file.name : 'Drag & drop file or browse'}
                </p>
                <p className="text-[11px] text-slate-500 mt-1">
                  Supports CSV, XLSX (Full 30-Column Product Master)
                </p>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <button
                  onClick={loadSampleData}
                  className="text-indigo-400 hover:text-indigo-300 text-xs flex items-center space-x-1 cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Load Sample 30-Column Dataset</span>
                </button>
                {uploadData && (
                  <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 rounded">
                    {uploadData.total_rows} Rows Loaded
                  </span>
                )}
              </div>
            </div>

            {uploadData && (
              <div className="mt-4 pt-3 border-t border-slate-800 text-xs">
                <span className="text-slate-400 block text-[11px] mb-1 font-semibold">
                  Detected Columns ({uploadData.columns.length}):
                </span>
                <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                  {uploadData.columns.slice(0, 10).map((col) => (
                    <span
                      key={col}
                      className="bg-slate-800 text-slate-300 text-[10px] px-2 py-0.5 rounded font-mono"
                    >
                      {col}
                    </span>
                  ))}
                  {uploadData.columns.length > 10 && (
                    <span className="text-[10px] text-slate-500 self-center">
                      +{uploadData.columns.length - 10} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* STEP 2: Smart Dual Front Packshot Card */}
          <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                  2
                </div>
                <h2 className="text-sm font-semibold text-white tracking-wide">
                  Smart Dual Front Packshot Engine (AI OCR)
                </h2>
              </div>

              {!uploadData ? (
                <div className="h-48 border border-slate-800/80 rounded-xl bg-slate-950/20 flex flex-col items-center justify-center text-slate-500 text-xs">
                  <Layers className="w-8 h-8 text-slate-600 mb-2" />
                  <span>Upload a file first to configure dual search queries</span>
                </div>
              ) : (
                <div className="space-y-4 text-xs">
                  {/* Two Query Previews */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    
                    {/* Strategy 1: Product Image (Without Size) */}
                    <div className="bg-slate-950 p-3 rounded-xl border border-indigo-500/30 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold text-indigo-300">
                          1. Product Image: Without Size
                        </span>
                        <span className="text-[9px] bg-indigo-500/20 text-indigo-300 px-1.5 py-0.2 rounded font-mono">
                          brand + name (no size)
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400">
                        Picks clean front packaging pouch without size callout
                      </p>
                      <div className="text-emerald-300 font-mono text-[11px] font-semibold truncate pt-1 border-t border-slate-800">
                        "{getProductPreviewQuery()}"
                      </div>
                    </div>

                    {/* Strategy 2: Variant Image (With Size) */}
                    <div className="bg-slate-950 p-3 rounded-xl border border-cyan-500/30 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold text-cyan-300">
                          2. Variant Image: With Size
                        </span>
                        <span className="text-[9px] bg-cyan-500/20 text-cyan-300 px-1.5 py-0.2 rounded font-mono">
                          brand + name + size
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400">
                        Inspects front pack for verified exact variant size
                      </p>
                      <div className="text-cyan-300 font-mono text-[11px] font-semibold truncate pt-1 border-t border-slate-800">
                        "{getVariantPreviewQuery()}"
                      </div>
                    </div>
                  </div>

                  {/* Primary column selection & Concurrency */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-slate-300 font-semibold mb-1">
                        Product Column:
                      </label>
                      <select
                        value={queryColumn}
                        onChange={(e) => setQueryColumn(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white font-medium cursor-pointer"
                      >
                        {uploadData.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-slate-300 font-semibold mb-1">
                        Speed (Workers: {concurrency}):
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="24"
                        value={concurrency}
                        onChange={(e) => setConcurrency(Number(e.target.value))}
                        className="w-full accent-indigo-500 cursor-pointer"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Launch Bar */}
            <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
              <span className="text-[11px] text-slate-400 font-mono">
                {uploadData ? `${uploadData.total_rows} items ready` : 'Awaiting file...'}
              </span>

              <div className="flex space-x-2">
                {isProcessing && (
                  <button
                    onClick={isPaused ? handleResume : handlePause}
                    className="bg-amber-600/30 hover:bg-amber-600/50 text-amber-300 border border-amber-500/40 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 cursor-pointer transition"
                  >
                    {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                    <span>{isPaused ? 'Resume' : 'Pause'}</span>
                  </button>
                )}

                {isProcessing && (
                  <button
                    onClick={handleCancel}
                    className="bg-rose-600/30 hover:bg-rose-600/50 text-rose-300 border border-rose-500/40 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 cursor-pointer transition"
                  >
                    <Square className="w-3.5 h-3.5" />
                    <span>Stop</span>
                  </button>
                )}

                <button
                  onClick={handleStartSearch}
                  disabled={!uploadData || isProcessing}
                  className={`bg-gradient-to-r from-indigo-600 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-bold px-5 py-1.5 rounded-xl text-xs flex items-center space-x-1.5 shadow-lg shadow-indigo-500/20 transition cursor-pointer active:scale-95 ${
                    (!uploadData || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <Play className="w-3.5 h-3.5 fill-white" />
                  <span>Start Dual Search</span>
                </button>
              </div>
            </div>
          </div>

        </div>

        {/* PROGRESS BAR & STATS PANEL */}
        {jobStatus && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <div className="flex items-center space-x-3">
                <span className="text-white font-bold">
                  Progress: {jobStatus.processed_rows} / {jobStatus.total_rows} ({jobStatus.progress_percent}%)
                </span>
                <span className="text-slate-400 truncate max-w-md">
                  Active: {jobStatus.current_query || '—'}
                </span>
              </div>
              <div className="flex items-center space-x-3 text-[11px]">
                <span className="text-emerald-400">Found: {jobStatus.success_count}</span>
                <span className="text-amber-400">Not Found: {jobStatus.not_found_count}</span>
                <span className="text-rose-400">Errors: {jobStatus.error_count}</span>
              </div>
            </div>

            <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-indigo-500 via-indigo-400 to-emerald-400 h-full transition-all duration-300"
                style={{ width: `${jobStatus.progress_percent}%` }}
              />
            </div>
          </div>
        )}

        {/* STEP 3: RESULTS DATASET TABLE */}
        {rows.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl space-y-0">
            
            {/* Table Toolbar */}
            <div className="p-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4 bg-slate-950/40">
              <div className="flex items-center space-x-3">
                
                {/* Filter buttons */}
                <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                  {['ALL', 'SUCCESS', 'NOT_FOUND', 'ERROR'].map((st) => (
                    <button
                      key={st}
                      onClick={() => {
                        setFilterStatus(st);
                        setCurrentPage(1);
                      }}
                      className={`px-3 py-1 rounded-lg font-medium transition cursor-pointer text-xs ${
                        filterStatus === st
                          ? 'bg-indigo-600 text-white shadow'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {st === 'ALL' && `All (${rows.length})`}
                      {st === 'SUCCESS' && `Found (${rows.filter(r => r.status === 'SUCCESS').length})`}
                      {st === 'NOT_FOUND' && `Not Found (${rows.filter(r => r.status === 'NOT_FOUND').length})`}
                      {st === 'ERROR' && `Errors (${rows.filter(r => r.status === 'ERROR').length})`}
                    </button>
                  ))}
                </div>

                {/* Bulk Retry for Not Found Rows */}
                {rows.filter(r => r.status === 'NOT_FOUND').length > 0 && (
                  <button
                    onClick={handleRetryNotFound}
                    disabled={isRetryingNotFound || isProcessing}
                    className="bg-amber-600/20 hover:bg-amber-600 text-amber-300 hover:text-white border border-amber-500/40 px-3 py-1 rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition cursor-pointer active:scale-95 disabled:opacity-50"
                    title="Run search only on Not Found rows"
                  >
                    <RotateCw className={`w-3.5 h-3.5 ${isRetryingNotFound ? 'animate-spin' : ''}`} />
                    <span>Retry Not Found ({rows.filter(r => r.status === 'NOT_FOUND').length})</span>
                  </button>
                )}
              </div>

              {/* Right Toolbar: Search & Export */}
              <div className="flex items-center space-x-3">
                
                {/* Instant Table Search Filter */}
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    value={tableSearch}
                    onChange={(e) => {
                      setTableSearch(e.target.value);
                      setCurrentPage(1);
                    }}
                    placeholder="Search product / query..."
                    className="bg-slate-950 border border-slate-700 rounded-xl pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 w-56 focus:outline-none focus:border-indigo-500"
                  />
                  {tableSearch && (
                    <button
                      onClick={() => {
                        setTableSearch('');
                        setCurrentPage(1);
                      }}
                      className="absolute right-2.5 top-2.5 text-slate-400 hover:text-white"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>

                {/* Page size selector */}
                <div className="flex items-center space-x-1.5 text-xs text-slate-400 bg-slate-950 px-2 py-1 rounded-xl border border-slate-800">
                  <span>Show:</span>
                  <select
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(e.target.value === 'ALL' ? 'ALL' : Number(e.target.value));
                      setCurrentPage(1);
                    }}
                    className="bg-transparent text-indigo-400 font-semibold cursor-pointer outline-none"
                  >
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                    <option value={200}>200</option>
                    <option value="ALL">All</option>
                  </select>
                </div>

                {/* Export Buttons */}
                <div className="flex space-x-1.5">
                  <button
                    onClick={() => handleDownload('csv')}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1 cursor-pointer transition active:scale-95"
                    title="Download updated CSV with pure image links"
                  >
                    <Download className="w-3.5 h-3.5 text-emerald-400" />
                    <span>CSV</span>
                  </button>
                  <button
                    onClick={() => handleDownload('xlsx')}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center space-x-1 cursor-pointer transition active:scale-95"
                    title="Download updated Excel spreadsheet"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Excel</span>
                  </button>
                </div>
              </div>
            </div>

            {/* High Performance Responsive Table */}
            <div className="overflow-x-auto max-h-[620px] overflow-y-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead className="bg-slate-950/95 sticky top-0 z-10 text-slate-400 font-semibold uppercase tracking-wider text-[10px] border-b border-slate-800">
                  <tr>
                    <th className="p-3 w-10 text-center">#</th>
                    
                    {/* DUAL IMAGE HEADERS */}
                    <th className="p-3 w-36 text-center border-r border-slate-800 bg-indigo-950/30">
                      Product Image (No Size)
                    </th>
                    <th className="p-3 w-36 text-center border-r border-slate-800 bg-cyan-950/30">
                      Variant Image (With Size)
                    </th>

                    <th className="p-3 min-w-[200px]">Product & Queries</th>
                    <th className="p-3 w-20">Status</th>
                    <th className="p-3 w-24">Prices</th>
                    <th className="p-3 w-24 text-center">Actions</th>

                    {/* All original CSV columns */}
                    {uploadData?.columns?.map((col) => (
                      <th key={col} className="p-3 min-w-[120px] whitespace-nowrap text-slate-300">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {paginatedRows.map((row) => (
                    <TableRow
                      key={row.row_index}
                      row={row}
                      index={row.row_index}
                      columns={uploadData?.columns || []}
                      cyclingKey={cyclingKey}
                      onOpenDetail={openDetailModal}
                      onCycleImage={handleCycleImage}
                    />
                  ))}
                  {paginatedRows.length === 0 && (
                    <tr>
                      <td colSpan={7 + (uploadData?.columns?.length || 0)} className="text-center py-12 text-slate-500 font-sans text-xs">
                        No rows found matching the current filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Bottom Bar */}
            <div className="p-3 border-t border-slate-800 bg-slate-950/70 flex items-center justify-between text-xs text-slate-400">
              <div className="font-mono text-[11px]">
                Showing {filteredRows.length > 0 ? (currentPage - 1) * (pageSize === 'ALL' ? filteredRows.length : pageSize) + 1 : 0} – {Math.min(currentPage * (pageSize === 'ALL' ? filteredRows.length : pageSize), filteredRows.length)} of {filteredRows.length} rows
              </div>

              {totalPages > 1 && (
                <div className="flex items-center space-x-1.5">
                  <button
                    onClick={() => setCurrentPage(1)}
                    disabled={currentPage === 1}
                    className="p-1 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    title="First page"
                  >
                    <ChevronsLeft className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="p-1 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    title="Previous page"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>

                  <span className="px-3 font-mono text-[11px] text-slate-200">
                    Page {currentPage} of {totalPages}
                  </span>

                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="p-1 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    title="Next page"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => setCurrentPage(totalPages)}
                    disabled={currentPage === totalPages}
                    className="p-1 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    title="Last page"
                  >
                    <ChevronsRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

          </div>
        )}
      </main>

      {/* DUAL PRODUCT & VARIANT DETAILS POPUP MODAL */}
      {detailModalRow && (() => {
        const orig = detailModalRow.original_data || {};
        const ignoredKeys = new Set([
          'product_images', 'variant_images', 'image_url', 'product_image_url',
          'variant_image_url', 'brand_image', 'category_icon', 'parent_category_icon'
        ]);

        const populatedFields = [];
        // Priority fields first if they have non-empty values
        const brandVal = orig.brand_name || orig.brand || orig.Brand || orig['Brand Name'] || '';
        const sizeVal = orig.variant_label || orig.variant || orig.Variant || orig.size || orig.Size || orig.pack_size || orig['Pack Size'] || '';
        const skuVal = orig.sku || orig.SKU || orig.Sku || '';
        const mrpVal = orig.mrp || orig.MRP || '';
        const priceVal = orig.sale_price || orig['Sale Price'] || orig.price || '';

        if (brandVal && brandVal !== '-' && brandVal !== '—' && brandVal.toLowerCase() !== 'nan') {
          populatedFields.push({ label: 'Brand', value: brandVal, isPrimary: true });
        }
        if (sizeVal && sizeVal !== '-' && sizeVal !== '—' && sizeVal.toLowerCase() !== 'nan') {
          populatedFields.push({ label: 'Pack Size', value: sizeVal, isSize: true });
        }
        if (skuVal && skuVal !== '-' && skuVal !== '—' && skuVal.toLowerCase() !== 'nan') {
          populatedFields.push({ label: 'SKU', value: skuVal, isMono: true });
        }
        if (mrpVal && mrpVal !== '-' && mrpVal !== '—' && mrpVal.toLowerCase() !== 'nan') {
          populatedFields.push({ label: 'MRP', value: mrpVal, isPrice: true });
        }
        if (priceVal && priceVal !== '-' && priceVal !== '—' && priceVal.toLowerCase() !== 'nan') {
          populatedFields.push({ label: 'Sale Price', value: priceVal, isPrice: true });
        }

        // Add any other non-empty columns from Excel
        const handledKeys = new Set([
          'brand_name', 'brand', 'brand name', 'product_name', 'product name',
          'variant_label', 'variant', 'size', 'pack_size', 'pack size',
          'sku', 'mrp', 'sale_price', 'sale price', 'price',
          'product_attributes', 'variant_attributes'
        ]);

        for (const [k, v] of Object.entries(orig)) {
          const kLower = k.toLowerCase().trim();
          if (ignoredKeys.has(kLower) || handledKeys.has(kLower)) continue;
          const valStr = String(v ?? '').trim();
          if (valStr && valStr !== '-' && valStr !== '—' && valStr.toLowerCase() !== 'nan' && valStr !== '0' && valStr !== '0.0') {
            const prettyLabel = k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            populatedFields.push({ label: prettyLabel, value: valStr });
          }
        }

        return (
          <div
            className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-3 sm:p-5"
            onClick={() => setDetailModalRow(null)}
          >
            <div
              className="bg-slate-900 border border-slate-800 rounded-3xl max-w-6xl w-full max-h-[94vh] overflow-hidden flex flex-col shadow-2xl animate-in zoom-in-95 duration-150"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/70">
                <div className="flex items-center space-x-3 truncate mr-4">
                  <span className="w-8 h-8 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs shrink-0 font-mono border border-indigo-500/30">
                    #{detailModalRow.row_index + 1}
                  </span>
                  <div className="truncate">
                    <h3 className="text-base font-bold text-white truncate">
                      {detailModalRow.original_data?.product_name || detailModalRow.query}
                    </h3>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 mt-1 font-sans">
                      {brandVal && (
                        <span className="bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 px-2.5 py-0.5 rounded-full font-semibold text-[11px]">
                          Brand: {brandVal}
                        </span>
                      )}
                      {sizeVal && (
                        <span className="bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 px-2.5 py-0.5 rounded-full font-semibold text-[11px]">
                          Size: {sizeVal}
                        </span>
                      )}
                      {priceVal && (
                        <span className="text-emerald-400 font-mono font-bold bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-full text-[11px]">
                          Price: {priceVal}
                        </span>
                      )}
                      {skuVal && (
                        <span className="text-slate-400 font-mono text-[10px] bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/60">
                          SKU: {skuVal}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setDetailModalRow(null)}
                  className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer shrink-0"
                  title="Close modal"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Modal Body */}
              <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-6">
                
                {/* DUAL IMAGE SIDE-BY-SIDE CARDS */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  
                  {/* 1. PRODUCT IMAGE CARD (WITHOUT SIZE) */}
                  <div className="bg-slate-950 p-4 sm:p-5 rounded-2xl border border-indigo-500/30 flex flex-col justify-between space-y-4 shadow-lg">
                    <div>
                      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                        <div>
                          <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wide flex items-center space-x-1.5">
                            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                            <span>Product Image (Without Size)</span>
                          </h4>
                        </div>
                        <span className="text-[10px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 px-2.5 py-0.5 rounded-full font-mono font-bold">
                          Clean Packshot
                        </span>
                      </div>

                      {/* Full Query Box (No Truncation) */}
                      <div className="mt-3">
                        <span className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider block mb-1">
                          Product Search Query:
                        </span>
                        <div className="text-[11px] font-mono text-slate-300 bg-slate-900/90 px-3 py-2 rounded-xl border border-slate-800 flex items-center justify-between select-text break-words">
                          <span className="break-all">{detailModalRow.product_query}</span>
                          <button
                            onClick={(e) => copyToClipboard(detailModalRow.product_query, e)}
                            className="p-1 hover:text-white text-slate-400 ml-2 shrink-0 cursor-pointer"
                            title="Copy query"
                          >
                            {copiedUrl === detailModalRow.product_query ? (
                              <Check className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Preview Image */}
                      <div className="w-full h-64 bg-slate-900/60 rounded-xl mt-3 overflow-hidden flex items-center justify-center p-3 relative group border border-slate-800/60">
                        {detailModalRow.product_image_url ? (
                          <img
                            src={detailModalRow.product_image_url}
                            alt="Product"
                            loading="lazy"
                            decoding="async"
                            className="max-h-full max-w-full object-contain hover:scale-105 transition duration-200"
                          />
                        ) : (
                          <div className="flex flex-col items-center justify-center text-slate-600">
                            <ImageIcon className="w-12 h-12 mb-2" />
                            <span className="text-xs">No Product Image Found</span>
                          </div>
                        )}
                      </div>

                      {/* URL bar */}
                      {detailModalRow.product_image_url && (
                        <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-900 px-3 py-2 rounded-xl border border-slate-800">
                          <span className="truncate max-w-[280px] select-text">{detailModalRow.product_image_url}</span>
                          <div className="flex items-center space-x-1 shrink-0 ml-2">
                            <button
                              onClick={(e) => copyToClipboard(detailModalRow.product_image_url, e)}
                              className="p-1.5 hover:text-white text-slate-400 cursor-pointer rounded-lg hover:bg-slate-800"
                              title="Copy link"
                            >
                              {copiedUrl === detailModalRow.product_image_url ? (
                                <Check className="w-3.5 h-3.5 text-emerald-400" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </button>
                            <a
                              href={detailModalRow.product_image_url}
                              target="_blank"
                              rel="noreferrer"
                              className="p-1.5 hover:text-white text-slate-400 cursor-pointer rounded-lg hover:bg-slate-800"
                              title="Open full size"
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Actions & Alternatives for Product Image */}
                    <div className="space-y-3 pt-3 border-t border-slate-800/80">
                      <div className="flex gap-2">
                        <button
                          onClick={(e) => handleCycleImage(detailModalRow.row_index, 'product', e)}
                          disabled={cyclingKey === `${detailModalRow.row_index}-product`}
                          className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-2.5 rounded-xl transition cursor-pointer flex items-center justify-center space-x-1.5 shadow-md active:scale-95"
                        >
                          <RotateCw className={`w-3.5 h-3.5 ${cyclingKey === `${detailModalRow.row_index}-product` ? 'animate-spin' : ''}`} />
                          <span>Cycle Product Image ({detailModalRow.product_alternatives?.length || 0})</span>
                        </button>
                      </div>

                      {/* Alternatives mini grid */}
                      {detailModalRow.product_alternatives && detailModalRow.product_alternatives.length > 0 && (
                        <div>
                          <div className="text-[10px] text-slate-400 mb-2 font-semibold">
                            Click to select alternative:
                          </div>
                          <div className="grid grid-cols-4 gap-2">
                            {detailModalRow.product_alternatives.slice(0, 4).map((alt, idx) => (
                              <div
                                key={idx}
                                onClick={() => handleSelectAlternativeImage(detailModalRow.row_index, alt, 'product')}
                                className="h-16 rounded-lg bg-slate-900 border border-slate-800 hover:border-indigo-400 overflow-hidden cursor-pointer p-1 flex items-center justify-center transition hover:scale-105"
                                title={alt.title}
                              >
                                <img src={alt.thumbnail_src || alt.img_src} alt="alt" loading="lazy" decoding="async" className="max-h-full max-w-full object-contain" />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 2. VARIANT IMAGE CARD (WITH SIZE) */}
                  <div className="bg-slate-950 p-4 sm:p-5 rounded-2xl border border-cyan-500/30 flex flex-col justify-between space-y-4 shadow-lg">
                    <div>
                      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                        <div>
                          <h4 className="text-xs font-bold text-cyan-300 uppercase tracking-wide flex items-center space-x-1.5">
                            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                            <span>Variant Image (With Size)</span>
                          </h4>
                        </div>
                        <span className="text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 px-2.5 py-0.5 rounded-full font-mono font-bold">
                          Exact Size Pack
                        </span>
                      </div>

                      {/* Full Query Box (No Truncation) */}
                      <div className="mt-3">
                        <span className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider block mb-1">
                          Variant Search Query:
                        </span>
                        <div className="text-[11px] font-mono text-slate-300 bg-slate-900/90 px-3 py-2 rounded-xl border border-slate-800 flex items-center justify-between select-text break-words">
                          <span className="break-all">{detailModalRow.variant_query}</span>
                          <button
                            onClick={(e) => copyToClipboard(detailModalRow.variant_query, e)}
                            className="p-1 hover:text-white text-slate-400 ml-2 shrink-0 cursor-pointer"
                            title="Copy query"
                          >
                            {copiedUrl === detailModalRow.variant_query ? (
                              <Check className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </div>
                      </div>

                      {/* Preview Image */}
                      <div className="w-full h-64 bg-slate-900/60 rounded-xl mt-3 overflow-hidden flex items-center justify-center p-3 relative group border border-slate-800/60">
                        {detailModalRow.variant_image_url ? (
                          <img
                            src={detailModalRow.variant_image_url}
                            alt="Variant"
                            loading="lazy"
                            decoding="async"
                            className="max-h-full max-w-full object-contain hover:scale-105 transition duration-200"
                          />
                        ) : (
                          <div className="flex flex-col items-center justify-center text-slate-600">
                            <ImageIcon className="w-12 h-12 mb-2" />
                            <span className="text-xs">No Variant Image Found</span>
                          </div>
                        )}
                      </div>

                      {/* URL bar */}
                      {detailModalRow.variant_image_url && (
                        <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-900 px-3 py-2 rounded-xl border border-slate-800">
                          <span className="truncate max-w-[280px] select-text">{detailModalRow.variant_image_url}</span>
                          <div className="flex items-center space-x-1 shrink-0 ml-2">
                            <button
                              onClick={(e) => copyToClipboard(detailModalRow.variant_image_url, e)}
                              className="p-1.5 hover:text-white text-slate-400 cursor-pointer rounded-lg hover:bg-slate-800"
                              title="Copy link"
                            >
                              {copiedUrl === detailModalRow.variant_image_url ? (
                                <Check className="w-3.5 h-3.5 text-emerald-400" />
                              ) : (
                                <Copy className="w-3.5 h-3.5" />
                              )}
                            </button>
                            <a
                              href={detailModalRow.variant_image_url}
                              target="_blank"
                              rel="noreferrer"
                              className="p-1.5 hover:text-white text-slate-400 cursor-pointer rounded-lg hover:bg-slate-800"
                              title="Open full size"
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Actions & Alternatives for Variant Image */}
                    <div className="space-y-3 pt-3 border-t border-slate-800/80">
                      <div className="flex gap-2">
                        <button
                          onClick={(e) => handleCycleImage(detailModalRow.row_index, 'variant', e)}
                          disabled={cyclingKey === `${detailModalRow.row_index}-variant`}
                          className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs py-2.5 rounded-xl transition cursor-pointer flex items-center justify-center space-x-1.5 shadow-md active:scale-95"
                        >
                          <RotateCw className={`w-3.5 h-3.5 ${cyclingKey === `${detailModalRow.row_index}-variant` ? 'animate-spin' : ''}`} />
                          <span>Cycle Variant Image ({detailModalRow.variant_alternatives?.length || 0})</span>
                        </button>
                      </div>

                      {/* Alternatives mini grid */}
                      {detailModalRow.variant_alternatives && detailModalRow.variant_alternatives.length > 0 && (
                        <div>
                          <div className="text-[10px] text-slate-400 mb-2 font-semibold">
                            Click to select alternative:
                          </div>
                          <div className="grid grid-cols-4 gap-2">
                            {detailModalRow.variant_alternatives.slice(0, 4).map((alt, idx) => (
                              <div
                                key={idx}
                                onClick={() => handleSelectAlternativeImage(detailModalRow.row_index, alt, 'variant')}
                                className="h-16 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-400 overflow-hidden cursor-pointer p-1 flex items-center justify-center transition hover:scale-105"
                                title={alt.title}
                              >
                                <img src={alt.thumbnail_src || alt.img_src} alt="alt" loading="lazy" decoding="async" className="max-h-full max-w-full object-contain" />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                </div>

                {/* Clean Product Metadata (Only Shows Non-Empty Fields from Excel) */}
                {populatedFields.length > 0 && (
                  <div className="bg-slate-950 p-4 sm:p-5 rounded-2xl border border-slate-800 space-y-3 text-xs shadow-lg">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <h4 className="font-bold text-white text-xs uppercase tracking-wider flex items-center space-x-2">
                        <Database className="w-3.5 h-3.5 text-indigo-400" />
                        <span>Product Excel Metadata</span>
                      </h4>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {populatedFields.length} attributes populated
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 pt-1">
                      {populatedFields.map((field, idx) => (
                        <div key={idx} className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800/70">
                          <span className="text-slate-400 block text-[10px] font-medium uppercase tracking-wider">
                            {field.label}
                          </span>
                          <span className={`font-mono text-xs mt-0.5 block truncate select-text ${
                            field.isPrice ? 'text-emerald-400 font-bold' : 
                            field.isSize ? 'text-cyan-300 font-semibold' : 
                            field.isPrimary ? 'text-indigo-300 font-semibold' : 'text-slate-200'
                          }`}>
                            {field.value}
                          </span>
                        </div>
                      ))}
                    </div>

                    {orig.product_attributes && (
                      <div className="pt-2 border-t border-slate-800/80">
                        <span className="text-slate-400 block text-[10px] mb-1 font-semibold uppercase tracking-wider">
                          Product Attributes:
                        </span>
                        <div className="text-[11px] text-slate-300 font-mono bg-slate-900/90 px-3 py-2 rounded-xl border border-slate-800 select-text break-words">
                          {orig.product_attributes}
                        </div>
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>
          </div>
        );
      })()}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 px-6 py-4 text-center text-xs text-slate-500 font-mono">
        SearXNG Product Master Finder • Product (No Size) + Variant (With Size) • Ultra-Fast Zero Lag
      </footer>
    </div>
  );
}