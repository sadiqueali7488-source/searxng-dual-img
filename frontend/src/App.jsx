import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Upload, Search, FileSpreadsheet, Download, Play, Pause, Square,
  CheckCircle2, XCircle, AlertCircle, RefreshCw, Eye, ExternalLink,
  Copy, Check, Settings, Sparkles, Image as ImageIcon, ArrowRight,
  Database, Zap, Clock, ShieldCheck, Layers, ChevronDown, ChevronUp,
  X, Filter, RotateCw, Tag, ShoppingBag, DollarSign, ListFilter,
  Sliders, Info, Edit3, ImagePlus, AlertTriangle
} from 'lucide-react';

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
  const [concurrency, setConcurrency] = useState(6);
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

  const [detailModalRow, setDetailModalRow] = useState(null);
  const [customProductQuery, setCustomProductQuery] = useState('');
  const [customVariantQuery, setCustomVariantQuery] = useState('');
  const [retryingTarget, setRetryingTarget] = useState(null); // 'product' | 'variant' | 'both'
  const [cyclingKey, setCyclingKey] = useState(null); // `${rowIndex}-${target}`

  const [quickSearchModal, setQuickSearchModal] = useState(false);
  const [quickQuery, setQuickQuery] = useState('pizza hut');
  const [quickResults, setQuickResults] = useState([]);
  const [quickSearching, setQuickSearching] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState('');

  const eventSourceRef = useRef(null);

  useEffect(() => {
    checkSearxngConnection(searxngUrl);
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

  const handleStartSearch = async () => {
    if (!uploadData?.job_id || !queryColumn) return;

    try {
      setIsProcessing(true);
      setIsPaused(false);

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
        setRows((prevRows) => {
          const index = prevRows.findIndex(r => r.row_index === updatedRow.row_index);
          if (index >= 0) {
            const next = [...prevRows];
            next[index] = updatedRow;
            return next;
          } else {
            return [...prevRows, updatedRow];
          }
        });

        setDetailModalRow(prev => {
          if (prev && prev.row_index === updatedRow.row_index) {
            return updatedRow;
          }
          return prev;
        });
      });

      es.addEventListener('job_done', () => {
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

  // Bulk Retry for Not Found Rows Only
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

  // Swap specific image (target = 'product' | 'variant')
  const handleSelectAlternativeImage = async (rowIndex, imageObj, target = 'product') => {
    if (!uploadData?.job_id) return;
    try {
      const formData = new FormData();
      formData.append('row_index', rowIndex);
      formData.append('new_image_json', JSON.stringify(imageObj));
      formData.append('target', target);

      const res = await axios.post(`/api/job/${uploadData.job_id}/swap-image`, formData);
      setRows((prev) => {
        const next = [...prev];
        next[rowIndex] = res.data.row;
        return next;
      });
      if (detailModalRow?.row_index === rowIndex) {
        setDetailModalRow(res.data.row);
      }
    } catch (err) {
      alert('Failed to swap image: ' + err.message);
    }
  };

  // Cycle image (target = 'product' | 'variant')
  const handleCycleImage = async (rowIndex, target = 'product', e) => {
    if (e) e.stopPropagation();
    if (!uploadData?.job_id) return;

    const key = `${rowIndex}-${target}`;
    setCyclingKey(key);
    try {
      const res = await axios.post(`/api/job/${uploadData.job_id}/cycle-image/${rowIndex}?target=${target}`);
      setRows((prev) => {
        const next = [...prev];
        next[rowIndex] = res.data.row;
        return next;
      });
      if (detailModalRow?.row_index === rowIndex) {
        setDetailModalRow(res.data.row);
      }
    } catch (err) {
      alert(err.response?.data?.detail || 'No more alternative images to cycle');
    } finally {
      setCyclingKey(null);
    }
  };

  // Retry single row with target ('product' | 'variant' | 'both')
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
      if (detailModalRow?.row_index === rowIndex) {
        setDetailModalRow(res.data.row);
      }
    } catch (err) {
      alert('Retry failed: ' + err.message);
    } finally {
      setRetryingTarget(null);
    }
  };

  const openDetailModal = (row) => {
    setDetailModalRow(row);
    setCustomProductQuery(row.product_query || '');
    setCustomVariantQuery(row.variant_query || '');
  };

  const copyToClipboard = (text, e) => {
    if (e) e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedUrl(text);
    setTimeout(() => setCopiedUrl(''), 2000);
  };

  const filteredRows = rows.filter((row) => {
    if (filterStatus === 'SUCCESS' && row.status !== 'SUCCESS') return false;
    if (filterStatus === 'NOT_FOUND' && row.status !== 'NOT_FOUND') return false;
    if (filterStatus === 'ERROR' && row.status !== 'ERROR') return false;
    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase();
      const matchP = row.product_query?.toLowerCase().includes(q);
      const matchV = row.variant_query?.toLowerCase().includes(q);
      const matchOrig = Object.values(row.original_data || {}).some(
        val => String(val).toLowerCase().includes(q)
      );
      return matchP || matchV || matchOrig;
    }
    return true;
  });

  const getProductPreviewQuery = () => {
    if (!uploadData || !uploadData.preview || uploadData.preview.length === 0) return '';
    const row = uploadData.preview[0];
    const brand = row['brand_name'] || row['brand'] || '';
    const name = row[queryColumn] || '';
    const parts = [];
    if (queryPrefix) parts.push(queryPrefix);
    if (brand && !name.toLowerCase().includes(brand.toLowerCase())) parts.push(brand);
    if (name) parts.push(name);
    if (querySuffix) parts.push(querySuffix);
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
                Dual Engine: Product (No Size) + Variant (With Size)
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              SearXNG 2nd Image • Google, Bing & DDG Only • Direct Clean URLs
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

      {/* SearXNG Settings */}
      {showSettings && (
        <div className="bg-slate-900 border-b border-indigo-500/30 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-4 max-w-2xl w-full">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider whitespace-nowrap">
              SearXNG Endpoint:
            </label>
            <input
              type="text"
              value={searxngUrl}
              onChange={(e) => setSearxngUrl(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white w-full focus:outline-none focus:border-indigo-500 font-mono"
              placeholder="http://localhost:8888"
            />
            <button
              onClick={() => checkSearxngConnection(searxngUrl)}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-4 py-1.5 rounded-lg font-medium transition cursor-pointer active:scale-95"
            >
              Test & Save
            </button>
          </div>
          <div className="text-xs text-slate-400 flex items-center space-x-2">
            <span>Status:</span>
            <span className={searxngStatus.connected ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
              {searxngStatus.message}
            </span>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        
        {/* Step 1 & Step 2 Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* STEP 1: Upload Card */}
          <div className="lg:col-span-5 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                    1
                  </div>
                  <h2 className="text-sm font-semibold text-white tracking-wide">
                    Upload CSV or Excel
                  </h2>
                </div>
                <button
                  onClick={loadSampleData}
                  disabled={uploading}
                  className="text-xs bg-slate-800 hover:bg-slate-700 text-indigo-300 hover:text-indigo-200 border border-slate-700/70 px-2.5 py-1 rounded-md transition cursor-pointer active:scale-95 flex items-center space-x-1"
                >
                  <Database className="w-3 h-3" />
                  <span>Load Sample 30-Cols</span>
                </button>
              </div>

              <label className="border-2 border-dashed border-slate-700 hover:border-indigo-500 bg-slate-950/40 hover:bg-slate-950/80 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition group">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  onChange={(e) => handleFileUpload(e.target.files[0])}
                  disabled={uploading}
                />
                <div className="w-12 h-12 rounded-full bg-slate-800/80 group-hover:bg-indigo-600/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6 text-slate-400 group-hover:text-indigo-400" />
                </div>
                <p className="text-sm font-medium text-slate-200 group-hover:text-indigo-300">
                  {uploading ? 'Processing file...' : 'Drop your CSV or Excel file here'}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Product Master 30 Columns • Auto Dual-Image Search
                </p>
              </label>

              {uploadData && (
                <div className="mt-4 p-3 bg-slate-950/60 border border-slate-800 rounded-xl flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2.5 truncate">
                    <FileSpreadsheet className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span className="font-medium text-white truncate max-w-[180px]">
                      {uploadData.filename}
                    </span>
                  </div>
                  <div className="flex items-center space-x-3 text-slate-400 shrink-0 font-mono">
                    <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px] text-emerald-400">
                      {uploadData.total_rows} rows
                    </span>
                    <span className="bg-slate-800 px-2 py-0.5 rounded text-[11px] text-indigo-300">
                      {uploadData.columns.length} columns
                    </span>
                  </div>
                </div>
              )}
            </div>

            {uploadData && (
              <div className="mt-4 pt-4 border-t border-slate-800/60">
                <p className="text-xs font-semibold text-slate-400 mb-2">30 Original Columns Intact:</p>
                <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
                  {uploadData.columns.map((col) => (
                    <span
                      key={col}
                      className="text-[10px] bg-slate-800/70 text-slate-300 px-2 py-0.5 rounded border border-slate-700/50 font-mono"
                    >
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* STEP 2: Dual Query Mapping Card */}
          <div className="lg:col-span-7 bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">
                  2
                </div>
                <h2 className="text-sm font-semibold text-white tracking-wide">
                  Imran Sir's Dual Image Strategy
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
                          brand + name
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400">
                        Searches general product pack without size constraints
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
                        Searches specific variant pack with exact size/weight
                      </p>
                      <div className="text-cyan-300 font-mono text-[11px] font-semibold truncate pt-1 border-t border-slate-800">
                        "{getVariantPreviewQuery()}"
                      </div>
                    </div>
                  </div>

                  {/* Primary column selection */}
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
                        max="20"
                        value={concurrency}
                        onChange={(e) => setConcurrency(e.target.value)}
                        className="w-full accent-indigo-500 cursor-pointer mt-1"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {isProcessing && (
                  <>
                    {isPaused ? (
                      <button
                        onClick={handleResume}
                        className="flex items-center space-x-1.5 bg-amber-600 hover:bg-amber-500 text-white text-xs px-3 py-2 rounded-lg font-semibold transition cursor-pointer active:scale-95"
                      >
                        <Play className="w-3.5 h-3.5" />
                        <span>Resume</span>
                      </button>
                    ) : (
                      <button
                        onClick={handlePause}
                        className="flex items-center space-x-1.5 bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/40 text-amber-300 text-xs px-3 py-2 rounded-lg font-semibold transition cursor-pointer active:scale-95"
                      >
                        <Pause className="w-3.5 h-3.5" />
                        <span>Pause</span>
                      </button>
                    )}
                    <button
                      onClick={handleCancel}
                      className="flex items-center space-x-1.5 bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/40 text-rose-300 text-xs px-3 py-2 rounded-lg font-semibold transition cursor-pointer active:scale-95"
                    >
                      <Square className="w-3.5 h-3.5" />
                      <span>Stop</span>
                    </button>
                  </>
                )}
              </div>

              <button
                onClick={handleStartSearch}
                disabled={!uploadData || isProcessing}
                className={`flex items-center space-x-2 px-6 py-2.5 rounded-xl font-bold text-xs transition cursor-pointer shadow-lg ${
                  !uploadData || isProcessing
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-indigo-500 via-indigo-600 to-indigo-700 hover:from-indigo-400 hover:to-indigo-600 text-white shadow-indigo-500/30 active:scale-95'
                }`}
              >
                <Zap className="w-4 h-4 fill-current" />
                <span>{isProcessing ? 'Processing Both Queries...' : 'Start Dual Image Search'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* STEP 3: Progress & Real-Time Stats with BULK RETRY NOT-FOUND */}
        {(jobStatus || rows.length > 0) && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <span>Processing Progress</span>
                  {jobStatus?.state === 'running' && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 animate-pulse font-sans">
                      Live Running
                    </span>
                  )}
                  {jobStatus?.state === 'completed' && (
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 font-sans">
                      Finished
                    </span>
                  )}
                </h3>
                {jobStatus?.current_query && (
                  <p className="text-xs text-slate-400 font-mono mt-0.5 truncate max-w-md">
                    Current: <span className="text-indigo-300">{jobStatus.current_query}</span>
                  </p>
                )}
              </div>

              {/* Action Toolbar: RETRY NOT FOUND + DOWNLOAD */}
              <div className="flex flex-wrap items-center gap-2">
                {/* DEDICATED RETRY ALL NOT FOUND BUTTON */}
                <button
                  onClick={handleRetryNotFound}
                  disabled={isProcessing}
                  className={`flex items-center space-x-1.5 text-xs px-3.5 py-2 rounded-lg font-bold transition shadow-md cursor-pointer active:scale-95 ${
                    isProcessing
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                      : 'bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 text-amber-300 shadow-amber-500/10'
                  }`}
                  title="Run retry only on rows that failed or were not found"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isRetryingNotFound ? 'animate-spin' : ''}`} />
                  <span>Retry All Not-Found ({jobStatus?.not_found_count || 0})</span>
                </button>

                <button
                  onClick={() => handleDownload('csv')}
                  className="flex items-center space-x-1.5 bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-emerald-500/30 text-xs px-3.5 py-2 rounded-lg font-medium transition shadow-sm cursor-pointer active:scale-95"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Clean CSV</span>
                </button>
                <button
                  onClick={() => handleDownload('xlsx')}
                  className="flex items-center space-x-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3.5 py-2 rounded-lg font-medium transition shadow-lg shadow-emerald-600/20 cursor-pointer active:scale-95"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>Download Excel (.xlsx)</span>
                </button>
              </div>
            </div>

            {/* Visual Progress Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span>
                  {jobStatus?.processed_rows || 0} of {jobStatus?.total_rows || 0} rows completed
                </span>
                <span className="font-bold text-indigo-400">
                  {jobStatus?.progress_percent || 0}%
                </span>
              </div>
              <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden border border-slate-800 p-0.5">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 rounded-full transition-all duration-300 shadow-[0_0_12px_#6366f1]"
                  style={{ width: `${jobStatus?.progress_percent || 0}%` }}
                />
              </div>
            </div>

            {/* Metric Stat Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
              <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-lg font-bold text-emerald-400 font-mono">
                    {jobStatus?.success_count || 0}
                  </div>
                  <div className="text-[11px] text-slate-400">Images Found</div>
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400">
                  <XCircle className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-lg font-bold text-rose-400 font-mono">
                    {(jobStatus?.not_found_count || 0) + (jobStatus?.error_count || 0)}
                  </div>
                  <div className="text-[11px] text-slate-400">Not Found (Retry Available)</div>
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                  <Zap className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-lg font-bold text-indigo-400 font-mono">
                    {jobStatus?.speed_rows_per_sec || 0} <span className="text-xs font-normal">/s</span>
                  </div>
                  <div className="text-[11px] text-slate-400">Speed (Rows/sec)</div>
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
                  <Clock className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-lg font-bold text-cyan-400 font-mono">
                    {jobStatus?.eta_seconds ? `${jobStatus.eta_seconds}s` : '0s'}
                  </div>
                  <div className="text-[11px] text-slate-400">Estimated ETA</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* STEP 4: Live Results Table with DUAL IMAGE PREVIEWS */}
        {rows.length > 0 && (
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl shadow-xl overflow-hidden flex flex-col">
            {/* Table Header & Filters */}
            <div className="p-4 border-b border-slate-800 bg-slate-900 flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div className="flex items-center space-x-3">
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <span>Results Table</span>
                  <span className="text-xs text-indigo-400 font-mono">({filteredRows.length} displayed)</span>
                </h3>
                <div className="flex items-center bg-slate-950 rounded-lg p-0.5 border border-slate-800 text-xs">
                  {['ALL', 'SUCCESS', 'NOT_FOUND', 'ERROR'].map((st) => (
                    <button
                      key={st}
                      onClick={() => setFilterStatus(st)}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition cursor-pointer ${
                        filterStatus === st
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {st === 'ALL' ? 'All' : st === 'SUCCESS' ? 'Found' : st === 'NOT_FOUND' ? 'Not Found' : 'Errors'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Table search filter */}
              <div className="relative max-w-xs w-full">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Filter rows by text..."
                  value={tableSearch}
                  onChange={(e) => setTableSearch(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            {/* Responsive Table Scroll Container */}
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
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
                  {filteredRows.map((row) => (
                    <tr
                      key={row.row_index}
                      onClick={() => openDetailModal(row)}
                      className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
                    >
                      <td className="p-3 text-center text-slate-500 font-mono text-[11px]">
                        {row.row_index + 1}
                      </td>

                      {/* 1. Product Image Thumbnail (Without Size) */}
                      <td className="p-3 text-center border-r border-slate-800/80 bg-indigo-950/10" onClick={(e) => e.stopPropagation()}>
                        <div className="flex flex-col items-center gap-1.5">
                          {row.product_image_url ? (
                            <div
                              onClick={() => openDetailModal(row)}
                              className="w-16 h-16 rounded-lg bg-slate-950 border border-indigo-500/40 overflow-hidden cursor-pointer hover:border-indigo-400 transition flex items-center justify-center relative shadow"
                              title="Product Image (Without Size) - Click to zoom"
                            >
                              <img
                                src={row.product_thumbnail_url || row.product_image_url}
                                alt="Product"
                                className="w-full h-full object-cover hover:scale-110 transition duration-200"
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
                              onClick={(e) => handleCycleImage(row.row_index, 'product', e)}
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
                      <td className="p-3 text-center border-r border-slate-800/80 bg-cyan-950/10" onClick={(e) => e.stopPropagation()}>
                        <div className="flex flex-col items-center gap-1.5">
                          {row.variant_image_url ? (
                            <div
                              onClick={() => openDetailModal(row)}
                              className="w-16 h-16 rounded-lg bg-slate-950 border border-cyan-500/40 overflow-hidden cursor-pointer hover:border-cyan-400 transition flex items-center justify-center relative shadow"
                              title="Variant Image (With Size) - Click to zoom"
                            >
                              <img
                                src={row.variant_thumbnail_url || row.variant_image_url}
                                alt="Variant"
                                className="w-full h-full object-cover hover:scale-110 transition duration-200"
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
                              onClick={(e) => handleCycleImage(row.row_index, 'variant', e)}
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

                      {/* Product Name & Both Queries */}
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
                      <td className="p-3">
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
                      <td className="p-3 font-mono">
                        <div className="text-emerald-400 font-bold">{row.original_data?.sale_price || '—'}</div>
                        <div className="text-slate-500 text-[10px] line-through">{row.original_data?.mrp || ''}</div>
                      </td>

                      {/* Actions */}
                      <td className="p-3 text-center" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={() => openDetailModal(row)}
                          className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-2.5 py-1 rounded text-[10px] font-sans font-medium transition cursor-pointer"
                        >
                          View Details
                        </button>
                      </td>

                      {/* Original columns */}
                      {uploadData?.columns?.map((col) => (
                        <td key={col} className="p-3 text-slate-300 font-sans whitespace-nowrap max-w-[180px] truncate">
                          {row.original_data?.[col] !== undefined ? String(row.original_data[col]) : ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* DUAL PRODUCT & VARIANT DETAILS POPUP MODAL */}
      {detailModalRow && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setDetailModalRow(null)}
        >
          <div
            className="bg-slate-900 border border-slate-800 rounded-3xl max-w-5xl w-full max-h-[92vh] overflow-hidden flex flex-col shadow-2xl animate-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
              <div className="flex items-center space-x-3 truncate">
                <span className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs shrink-0 font-mono">
                  #{detailModalRow.row_index + 1}
                </span>
                <div className="truncate">
                  <h3 className="text-base font-bold text-white truncate">
                    {detailModalRow.original_data?.product_name || detailModalRow.query}
                  </h3>
                  <div className="flex items-center space-x-2 text-xs text-slate-400 mt-0.5 font-sans">
                    {detailModalRow.original_data?.brand_name && (
                      <span className="bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 px-2 py-0.2 rounded font-semibold text-[11px]">
                        Brand: {detailModalRow.original_data.brand_name}
                      </span>
                    )}
                    {detailModalRow.original_data?.variant_label && (
                      <span className="bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 px-2 py-0.2 rounded font-semibold text-[11px]">
                        Size: {detailModalRow.original_data.variant_label}
                      </span>
                    )}
                    <span className="text-emerald-400 font-mono font-bold">
                      {detailModalRow.original_data?.sale_price}
                    </span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setDetailModalRow(null)}
                className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              {/* DUAL IMAGE SIDE-BY-SIDE CARDS */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* 1. PRODUCT IMAGE CARD (WITHOUT SIZE) */}
                <div className="bg-slate-950 p-4 rounded-2xl border border-indigo-500/30 flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <div>
                        <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wide">
                          Product Image (Without Size)
                        </h4>
                        <p className="text-[10px] text-slate-400 font-mono truncate max-w-xs mt-0.5">
                          Query: "{detailModalRow.product_query}"
                        </p>
                      </div>
                      <span className="text-[9px] bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded font-mono font-bold">
                        General Pack
                      </span>
                    </div>

                    {/* Preview Image */}
                    <div className="w-full h-56 bg-slate-900/60 rounded-xl mt-3 overflow-hidden flex items-center justify-center p-2 relative group">
                      {detailModalRow.product_image_url ? (
                        <img
                          src={detailModalRow.product_image_url}
                          alt="Product"
                          className="max-h-full max-w-full object-contain"
                        />
                      ) : (
                        <ImageIcon className="w-12 h-12 text-slate-700" />
                      )}
                    </div>

                    {/* URL bar */}
                    {detailModalRow.product_image_url && (
                      <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1.5 rounded-lg border border-slate-800">
                        <span className="truncate max-w-[240px]">{detailModalRow.product_image_url}</span>
                        <div className="flex items-center space-x-1 shrink-0">
                          <button
                            onClick={(e) => copyToClipboard(detailModalRow.product_image_url, e)}
                            className="p-1 hover:text-white cursor-pointer"
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
                            className="p-1 hover:text-white cursor-pointer"
                            title="Open full size"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Actions & Alternatives for Product Image */}
                  <div className="space-y-3 pt-2 border-t border-slate-800">
                    <div className="flex gap-2">
                      <button
                        onClick={(e) => handleCycleImage(detailModalRow.row_index, 'product', e)}
                        disabled={cyclingKey === `${detailModalRow.row_index}-product`}
                        className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-2 rounded-xl transition cursor-pointer flex items-center justify-center space-x-1.5 shadow-md active:scale-95"
                      >
                        <RotateCw className={`w-3.5 h-3.5 ${cyclingKey === `${detailModalRow.row_index}-product` ? 'animate-spin' : ''}`} />
                        <span>Cycle Product Image ({detailModalRow.product_alternatives?.length || 0})</span>
                      </button>
                    </div>

                    {/* Alternatives mini grid */}
                    {detailModalRow.product_alternatives && detailModalRow.product_alternatives.length > 0 && (
                      <div>
                        <div className="text-[10px] text-slate-400 mb-1.5 font-semibold">
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
                              <img src={alt.thumbnail_src || alt.img_src} alt="alt" className="max-h-full max-w-full object-contain" />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 2. VARIANT IMAGE CARD (WITH SIZE) */}
                <div className="bg-slate-950 p-4 rounded-2xl border border-cyan-500/30 flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <div>
                        <h4 className="text-xs font-bold text-cyan-300 uppercase tracking-wide">
                          Variant Image (With Size)
                        </h4>
                        <p className="text-[10px] text-slate-400 font-mono truncate max-w-xs mt-0.5">
                          Query: "{detailModalRow.variant_query}"
                        </p>
                      </div>
                      <span className="text-[9px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded font-mono font-bold">
                        Exact Size Pack
                      </span>
                    </div>

                    {/* Preview Image */}
                    <div className="w-full h-56 bg-slate-900/60 rounded-xl mt-3 overflow-hidden flex items-center justify-center p-2 relative group">
                      {detailModalRow.variant_image_url ? (
                        <img
                          src={detailModalRow.variant_image_url}
                          alt="Variant"
                          className="max-h-full max-w-full object-contain"
                        />
                      ) : (
                        <ImageIcon className="w-12 h-12 text-slate-700" />
                      )}
                    </div>

                    {/* URL bar */}
                    {detailModalRow.variant_image_url && (
                      <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1.5 rounded-lg border border-slate-800">
                        <span className="truncate max-w-[240px]">{detailModalRow.variant_image_url}</span>
                        <div className="flex items-center space-x-1 shrink-0">
                          <button
                            onClick={(e) => copyToClipboard(detailModalRow.variant_image_url, e)}
                            className="p-1 hover:text-white cursor-pointer"
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
                            className="p-1 hover:text-white cursor-pointer"
                            title="Open full size"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Actions & Alternatives for Variant Image */}
                  <div className="space-y-3 pt-2 border-t border-slate-800">
                    <div className="flex gap-2">
                      <button
                        onClick={(e) => handleCycleImage(detailModalRow.row_index, 'variant', e)}
                        disabled={cyclingKey === `${detailModalRow.row_index}-variant`}
                        className="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs py-2 rounded-xl transition cursor-pointer flex items-center justify-center space-x-1.5 shadow-md active:scale-95"
                      >
                        <RotateCw className={`w-3.5 h-3.5 ${cyclingKey === `${detailModalRow.row_index}-variant` ? 'animate-spin' : ''}`} />
                        <span>Cycle Variant Image ({detailModalRow.variant_alternatives?.length || 0})</span>
                      </button>
                    </div>

                    {/* Alternatives mini grid */}
                    {detailModalRow.variant_alternatives && detailModalRow.variant_alternatives.length > 0 && (
                      <div>
                        <div className="text-[10px] text-slate-400 mb-1.5 font-semibold">
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
                              <img src={alt.thumbnail_src || alt.img_src} alt="alt" className="max-h-full max-w-full object-contain" />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* Product Specifications & Full Attributes */}
              <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-3 text-xs">
                <h4 className="font-bold text-white text-xs uppercase tracking-wide">Product Metadata</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <span className="text-slate-400 block text-[10px]">SKU:</span>
                    <span className="font-mono text-slate-200">{detailModalRow.original_data?.sku || '—'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Barcode:</span>
                    <span className="font-mono text-slate-200">{detailModalRow.original_data?.barcode || '—'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">MRP:</span>
                    <span className="font-mono text-emerald-400 font-bold">{detailModalRow.original_data?.mrp || '—'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Sale Price:</span>
                    <span className="font-mono text-cyan-400 font-bold">{detailModalRow.original_data?.sale_price || '—'}</span>
                  </div>
                </div>

                {detailModalRow.original_data?.product_attributes && (
                  <div className="pt-2 border-t border-slate-800">
                    <span className="text-slate-400 block text-[10px] mb-1 font-semibold">Product Attributes:</span>
                    <div className="text-[11px] text-slate-300 font-mono bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 break-all">
                      {detailModalRow.original_data.product_attributes}
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 px-6 py-4 text-center text-xs text-slate-500 font-mono">
        SearXNG Product Master Finder • Product (No Size) + Variant (With Size) • Ultra-Fast Zero Lag
      </footer>
    </div>
  );
}
