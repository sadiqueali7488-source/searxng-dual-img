import re
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_brand(brand, prod_name=""):
    if not brand:
        if prod_name and re.match(r"^9\s*AM\b", prod_name, re.I):
            return "9AM"
        return ""
    b = str(brand).strip()
    # Fix 9:00 AM or 09:00:00 or 9:00 AM
    b = re.sub(r"^0?([0-9]+):00(?::00)?\s*(AM|PM)$", r"\1\2", b, flags=re.I)
    if "9:00" in b:
        b = "9AM"
    if prod_name and re.match(r"^9\s*AM\b", prod_name, re.I):
        b = "9AM"
    return b

def clean_currency(val):
    if not val:
        return ""
    s = str(val).strip().replace(",", "")
    if not s or s.lower() == "nan":
        return ""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if match:
        num = float(match.group(1))
        return f"₹{num:.2f}"
    return s

def clean_axes(axes):
    if not axes:
        return ""
    s = str(axes).strip()
    # remove json array brackets and quotes
    s = re.sub(r"^[\[\(\{\"\']+|[\]\)\}\"\']+$", "", s)
    s = s.replace('"', "").replace("'", "")
    parts = [p.strip() for p in re.split(r"[,;\|]", s) if p.strip()]
    return " | ".join(parts)

def clean_attributes(attr):
    if not attr:
        return ""
    s = str(attr).strip()
    # remove newlines, replace multiple spaces
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    parts = [p.strip() for p in re.split(r"[;\|]", s) if p.strip()]
    return "|".join(parts)

def clean_image_link(img_val):
    if not img_val:
        return ""
    s = str(img_val).strip()
    # Remove json array brackets ["..."]
    s = re.sub(r"^[\[\(\{\"\']+|[\]\)\}\"\']+$", "", s)
    s = s.replace('"', "").replace("'", "")
    return s.strip()

print("Brand '9:00 AM':", clean_brand("9:00 AM"))
print("MRP '120':", clean_currency("120"))
print("MRP '₹130.00':", clean_currency("₹130.00"))
print("Axes '[\"Pack Size\"]':", clean_axes('["Pack Size"]'))
print("Attr 'Pack Size:1kg; Type:Atta':", clean_attributes("Pack Size:1kg; Type:Atta"))
print("Image '[\"https://example.com/img.jpg\"]':", clean_image_link('["https://example.com/img.jpg"]'))
