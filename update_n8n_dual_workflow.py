import json
import subprocess

EXPECTED_COLUMNS = [
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

js_prepare_queries = """
const EXPECTED_COLUMNS = [
  'sku', 'variant_label', 'product_name', 'product_description',
  'product_prescription_required', 'brand_name', 'brand_description',
  'brand_image', 'parent_category_name', 'parent_category_description',
  'parent_category_icon', 'parent_category_prescription_required',
  'parent_category_is_featured', 'category_name', 'category_description',
  'category_icon', 'category_prescription_required', 'category_is_featured',
  'mrp', 'sale_price', 'tax_inclusive', 'barcode', 'gtin', 'mpn', 'color_hex',
  'product_attributes', 'variant_attributes', 'variant_axes', 'product_images',
  'variant_images'
];

const rows = $input.all();
if (rows.length === 0) {
  throw new Error('The uploaded file has no data rows. Add at least one product row and upload again.');
}

function cleanBrand(brand, prodName) {
  let b = String(brand || '').trim();
  if (/^0?9:00(:00)?\s*(AM|PM)?$/i.test(b) || b.includes('9:00') || /^9\s*AM$/i.test(b)) return '9AM';
  if (prodName && /^9\s*AM\b/i.test(prodName)) return '9AM';
  return b;
}

function cleanCurrency(val) {
  if (!val) return '';
  const s = String(val).replace(/,/g, '').trim();
  const match = s.match(/([0-9]+(?:\.[0-9]+)?)/);
  if (match) {
    return '₹' + parseFloat(match[1]).toFixed(2);
  }
  return s;
}

function cleanAxes(axes) {
  if (!axes) return '';
  let s = String(axes).trim().replace(/^[\[\(\{"']+|[\]\)\}"']+$/g, '').replace(/["']/g, '');
  const parts = s.split(/[,;|]/).map(p => p.trim()).filter(Boolean);
  return parts.join(' | ');
}

function cleanAttr(attr) {
  if (!attr) return '';
  let s = String(attr).replace(/\\r?\\n|\\r/g, ' ').trim();
  const parts = s.split(/[;|]/).map(p => p.trim()).filter(Boolean);
  return parts.join('|');
}

return rows.map((item, index) => {
  const row = item.json ?? {};
  const normalized = {};

  for (const col of EXPECTED_COLUMNS) {
    normalized[col] = (row[col] !== undefined && row[col] !== null) ? String(row[col]) : '';
  }

  const prodName = normalized.product_name.trim();
  normalized.brand_name = cleanBrand(normalized.brand_name, prodName);
  normalized.mrp = cleanCurrency(normalized.mrp);
  normalized.sale_price = cleanCurrency(normalized.sale_price);
  normalized.variant_axes = cleanAxes(normalized.variant_axes);
  normalized.product_attributes = cleanAttr(normalized.product_attributes);
  normalized.variant_attributes = cleanAttr(normalized.variant_attributes);
  normalized.product_images = '';
  normalized.variant_images = '';

  const brand = normalized.brand_name.trim();
  const variant = normalized.variant_label.trim();

  // 1. Product Image Query (WITHOUT SIZE) -> Brand + Product Name
  const pParts = [];
  if (brand && !prodName.toLowerCase().includes(brand.toLowerCase())) {
    pParts.push(brand);
  }
  if (prodName) pParts.push(prodName);
  const productQuery = pParts.join(' ').replace(/\\s+/g, ' ').trim() || prodName || 'product';

  // 2. Variant Image Query (WITH SIZE) -> Brand + Product Name + Variant Label
  const vParts = [];
  if (brand && !prodName.toLowerCase().includes(brand.toLowerCase())) {
    vParts.push(brand);
  }
  if (prodName) vParts.push(prodName);
  if (variant && !prodName.toLowerCase().includes(variant.toLowerCase())) {
    vParts.push(variant);
  }
  const variantQuery = vParts.join(' ').replace(/\\s+/g, ' ').trim() || productQuery;

  return {
    json: {
      ...normalized,
      product_query: productQuery,
      variant_query: variantQuery,
      row_index: index
    },
    pairedItem: { item: index }
  };
});
"""

js_pick_product_image = """
const original = $('Prepare Dual Search Queries').item.json;
let payload = $json.body ?? $json;

if (typeof payload === 'string') {
  try {
    payload = JSON.parse(payload);
  } catch (error) {}
}

if (payload && payload.data && Array.isArray(payload.data.results)) {
  payload = payload.data;
}

const results = Array.isArray(payload?.results) ? payload.results : [];

// Pick 2nd image (index 1) if available, otherwise fallback to 1st image (index 0)
let selectedImg = null;
if (results.length > 1) {
  selectedImg = results[1];
} else if (results.length === 1) {
  selectedImg = results[0];
}

let imageUrl = '';
if (selectedImg) {
  imageUrl = selectedImg.img_src || selectedImg.thumbnail_src || selectedImg.url || '';
  if (imageUrl.startsWith('//')) {
    imageUrl = 'https:' + imageUrl;
  }
}

// Pure direct URL link without array brackets or quotes
imageUrl = imageUrl.trim().replace(/^[\["']+|[\]"']+$/g, '');

const updatedRow = { ...original };
updatedRow.product_images = imageUrl;

return {
  json: updatedRow,
  pairedItem: $input.item.pairedItem
};
"""

js_pick_variant_image = """
const original = $('Pick Product Image (2nd Result)').item.json;
let payload = $json.body ?? $json;

if (typeof payload === 'string') {
  try {
    payload = JSON.parse(payload);
  } catch (error) {}
}

if (payload && payload.data && Array.isArray(payload.data.results)) {
  payload = payload.data;
}

const results = Array.isArray(payload?.results) ? payload.results : [];

// Pick 2nd image (index 1) if available, otherwise fallback to 1st image (index 0)
let selectedImg = null;
if (results.length > 1) {
  selectedImg = results[1];
} else if (results.length === 1) {
  selectedImg = results[0];
}

let imageUrl = '';
if (selectedImg) {
  imageUrl = selectedImg.img_src || selectedImg.thumbnail_src || selectedImg.url || '';
  if (imageUrl.startsWith('//')) {
    imageUrl = 'https:' + imageUrl;
  }
}

// Pure direct URL link without array brackets or quotes
imageUrl = imageUrl.trim().replace(/^[\["']+|[\]"']+$/g, '');

const finalRow = { ...original };
finalRow.variant_images = imageUrl;

// Fallback if one image is missing
if (!finalRow.product_images && finalRow.variant_images) {
  finalRow.product_images = finalRow.variant_images;
}
if (!finalRow.variant_images && finalRow.product_images) {
  finalRow.variant_images = finalRow.product_images;
}

delete finalRow.product_query;
delete finalRow.variant_query;
delete finalRow.row_index;

return {
  json: finalRow,
  pairedItem: $input.item.pairedItem
};
"""

nodes = [
  {
    "parameters": {
      "path": "b840f3a8-cd66-4159-8558-d04510ab3514",
      "formTitle": "SearXNG Product Master Image Finder (Dual Images: Without & With Size)",
      "formDescription": "Upload Product Master Excel (.xlsx) or CSV. Dual search will find Product Image (without size) and Variant Image (with size) from SearXNG picking the 2nd best image with 100% clean formatting and all 30 columns preserved.",
      "formFields": {
        "values": [
          {
            "fieldLabel": "Product Master File (.xlsx, .xls, .csv)",
            "fieldType": "file",
            "requiredField": True
          }
        ]
      },
      "options": {}
    },
    "type": "n8n-nodes-base.formTrigger",
    "typeVersion": 2.2,
    "position": [0, 0],
    "id": "e83286b2-f38b-4bfe-9875-9e4726bbd79b",
    "name": "Upload Excel",
    "webhookId": "b840f3a8-cd66-4159-8558-d04510ab3514"
  },
  {
    "parameters": {
      "operation": "fromFile",
      "binaryPropertyName": "data",
      "options": {
        "readAsString": True
      }
    },
    "type": "n8n-nodes-base.extractFromFile",
    "typeVersion": 1,
    "position": [220, 0],
    "id": "cbdd6216-9b16-43ad-8d9e-1d5be7ea529b",
    "name": "Read Excel Rows"
  },
  {
    "parameters": {
      "jsCode": js_prepare_queries
    },
    "type": "n8n-nodes-base.code",
    "typeVersion": 2,
    "position": [440, 0],
    "id": "18cfc7a7-3ce9-408a-a63e-b49eb72df3b6",
    "name": "Prepare Dual Search Queries"
  },
  {
    "parameters": {
      "url": "http://host.docker.internal:8888/search",
      "sendQuery": True,
      "queryParameters": {
        "parameters": [
          {"name": "q", "value": "={{ $json.product_query }}"},
          {"name": "category_images", "value": "1"},
          {"name": "engines", "value": "google images,bing images,duckduckgo images"},
          {"name": "format", "value": "json"}
        ]
      },
      "options": {
        "batching": {
          "batch": {
            "batchSize": 5,
            "batchInterval": 100
          }
        }
      }
    },
    "type": "n8n-nodes-base.httpRequest",
    "typeVersion": 4.2,
    "position": [680, 0],
    "id": "f5df84b0-a359-4d69-be54-c64ef5f9927f",
    "name": "Search Product Images (No Size)"
  },
  {
    "parameters": {
      "jsCode": js_pick_product_image
    },
    "type": "n8n-nodes-base.code",
    "typeVersion": 2,
    "position": [920, 0],
    "id": "c1f7ab29-4ae5-48fa-8692-749e75df4211",
    "name": "Pick Product Image (2nd Result)"
  },
  {
    "parameters": {
      "url": "http://host.docker.internal:8888/search",
      "sendQuery": True,
      "queryParameters": {
        "parameters": [
          {"name": "q", "value": "={{ $json.variant_query }}"},
          {"name": "category_images", "value": "1"},
          {"name": "engines", "value": "google images,bing images,duckduckgo images"},
          {"name": "format", "value": "json"}
        ]
      },
      "options": {
        "batching": {
          "batch": {
            "batchSize": 5,
            "batchInterval": 100
          }
        }
      }
    },
    "type": "n8n-nodes-base.httpRequest",
    "typeVersion": 4.2,
    "position": [1160, 0],
    "id": "e4299b82-938a-4933-9ce6-c67ef84a1d29",
    "name": "Search Variant Images (With Size)"
  },
  {
    "parameters": {
      "jsCode": js_pick_variant_image
    },
    "type": "n8n-nodes-base.code",
    "typeVersion": 2,
    "position": [1400, 0],
    "id": "84477218-c2b6-455b-819e-f002bf023ab8",
    "name": "Pick Variant Image (2nd Result)"
  },
  {
    "parameters": {
      "operation": "toFile",
      "fileFormat": "csv",
      "options": {
        "fileName": "Product_Master_With_SearXNG_Dual_Images.csv"
      }
    },
    "type": "n8n-nodes-base.convertToFile",
    "typeVersion": 1.1,
    "position": [1640, 0],
    "id": "a1f9dcae-0fb5-4a6c-9c98-43d78c33ec75",
    "name": "Create Result CSV"
  },
  {
    "parameters": {
      "operation": "completion",
      "respondWith": "binary",
      "dataPropertyName": "data"
    },
    "type": "n8n-nodes-base.form",
    "typeVersion": 2.2,
    "position": [1880, 0],
    "id": "b304dfa2-c7f8-48b7-846f-c1f0b09968a3",
    "name": "Download Result File"
  }
]

connections = {
  "Upload Excel": {
    "main": [[{"node": "Read Excel Rows", "type": "main", "index": 0}]]
  },
  "Read Excel Rows": {
    "main": [[{"node": "Prepare Dual Search Queries", "type": "main", "index": 0}]]
  },
  "Prepare Dual Search Queries": {
    "main": [[{"node": "Search Product Images (No Size)", "type": "main", "index": 0}]]
  },
  "Search Product Images (No Size)": {
    "main": [[{"node": "Pick Product Image (2nd Result)", "type": "main", "index": 0}]]
  },
  "Pick Product Image (2nd Result)": {
    "main": [[{"node": "Search Variant Images (With Size)", "type": "main", "index": 0}]]
  },
  "Search Variant Images (With Size)": {
    "main": [[{"node": "Pick Variant Image (2nd Result)", "type": "main", "index": 0}]]
  },
  "Pick Variant Image (2nd Result)": {
    "main": [[{"node": "Create Result CSV", "type": "main", "index": 0}]]
  },
  "Create Result CSV": {
    "main": [[{"node": "Download Result File", "type": "main", "index": 0}]]
  }
}

full_wf = {
  "name": "SearXNG - Dual Product & Variant Image Finder (Without & With Size 30-Cols)",
  "nodes": nodes,
  "connections": connections,
  "settings": {"executionOrder": "v1"},
  "staticData": None
}

with open("c:/Users/Administrator/Desktop/image search/n8n_dual_search_workflow.json", "w", encoding="utf-8") as f:
    json.dump(full_wf, f, indent=2)

print("n8n_dual_search_workflow.json written successfully!")

# Now update n8n SQLite DB for both workflow IDs: jPHVv0xMEkwPLSQb and Q3NNpfzfLfBc4Nxr
nodes_json = json.dumps(nodes)
connections_json = json.dumps(connections)
name = "SearXNG - Dual Product & Variant Image Finder (Without & With Size 30-Cols)"

update_script = f"""
const sqlite3 = require("/usr/local/lib/node_modules/n8n/node_modules/sqlite3");
const db = new sqlite3.Database("/home/node/.n8n/database.sqlite");

const name = {json.dumps(name)};
const nodes = {json.dumps(nodes_json)};
const connections = {json.dumps(connections_json)};

const wfIds = ["jPHVv0xMEkwPLSQb", "Q3NNpfzfLfBc4Nxr"];
let count = 0;
for (const id of wfIds) {{
  db.run("UPDATE workflow_entity SET name = ?, nodes = ?, connections = ? WHERE id = ?", [name, nodes, connections, id], function(err) {{
    if (err) console.error("Error updating " + id + ":", err);
    else console.log("Updated workflow " + id + ", changes: " + this.changes);
    count++;
  }});
}}
"""

res = subprocess.run(['docker', 'exec', 'N8N', 'node', '-e', update_script], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("ERR:", res.stderr)
