# mapillary_download_360_with_sequence_filter.py
# Requires: requests, pandas, tqdm

import os
import csv
import time
import json
import requests
from tqdm import tqdm
import pandas as pd
import urllib.parse as up

# ---------------- USER CONFIG ----------------
ACCESS_TOKEN = "MLY|25984759771124636|f4bba18fac5061e39514c35a5fb9b318"  
# If you want to download all images in a specific sequence, set SEQUENCE_KEY to that key (string).
# Otherwise set SEQUENCE_KEY = None and the script will only use the BBOX search.
SEQUENCE_KEY = None  # e.g. "s-ABCD1234..." or None

# bbox = lon_min, lat_min, lon_max, lat_max
BBOX = (-102.4486111, 44.0658333, -102.4422222, 44.0725000) 
OUT_DIR = "mapillary_images"
CSV_PATH = os.path.join(OUT_DIR, "mapillary_images_metadata.csv")

# API base
BASE_URL = "https://graph.mapillary.com"

# We include common EXIF and computed fields; the code will also capture any other keys present in the returned JSON (stored as raw_json).
FIELDS = ",".join([
    "id",
    "key",
    "sequence_key",
    "captured_at",
    "captured_by",
    "camera_type",
    "thumb_2048_url",
    "thumb_original_url",
    "original_url",
    "computed_geometry",
    "computed_rotation",
    "compass_angle",
    "exif_orientation",
    "exif_make",
    "exif_model",
    "exif_lens_model",
    "exif_focal_length",
    "exif_aperture",
    "exif_iso",
    "height",
    "width",
    "file_size"
])

os.makedirs(OUT_DIR, exist_ok=True)

def fetch_images_page(access_token, bbox=None, fields=FIELDS, limit=100, after=None):
    """
    Fetch a page of images using bbox search on /images endpoint.
    Returns the JSON dict.
    """
    params = {
        "access_token": access_token,
        "fields": fields,
        "limit": limit
    }
    if bbox:
        params["bbox"] = ",".join(map(str, bbox))
    if after:
        params["after"] = after
    resp = requests.get(f"{BASE_URL}/images", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_sequence_images(access_token, sequence_key, fields=FIELDS, limit=100, after=None):
    """
    Fetch images that belong to a sequence by querying the sequences endpoint.
    Mapillary sequences expose an images edge; we'll request images{<fields>} using Graph API.
    We'll use the sequences endpoint to iterate pages if needed.
    """
    # The sequences endpoint supports fields; request images with nested fields
    # Example: /{sequence_key}?access_token=...&fields=images{<fields>}
    params = {
        "access_token": access_token,
        "fields": f"images{{{fields}}}",
        "limit": limit
    }
    if after:
        params["after"] = after
    resp = requests.get(f"{BASE_URL}/{sequence_key}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def pick_image_url(item):
    for key in ("original_url", "thumb_original_url", "thumb_2048_url"):
        if key in item and item.get(key):
            return item[key]
    for k,v in item.items():
        if isinstance(v, str) and k.lower().endswith("_url"):
            return v
    return None

def get_coordinates(item):
    geom = item.get("computed_geometry") or item.get("geometry") or {}
    if isinstance(geom, dict):
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
    if "lat" in item and "lon" in item:
        return float(item["lon"]), float(item["lat"])
    return None, None

def download_file(url, out_path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get('content-length')
        if total is None:
            with open(out_path, "wb") as f:
                f.write(r.content)
        else:
            total = int(total)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

def flatten_item(item):
    """
    Turn item dict into a flat dict suitable for CSV:
    - Top-level scalar keys copied as-is
    - Nested dicts (like computed_geometry) serialized as JSON strings
    - Always include sequence_key, id, and raw_json
    """
    flat = {}
    for k, v in item.items():
        if isinstance(v, (dict, list)):
            try:
                flat[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                flat[k] = str(v)
        else:
            flat[k] = v
    # computed_geometry convenience: add lon/lat columns if available
    lon, lat = get_coordinates(item)
    flat["lon"] = lon
    flat["lat"] = lat
    flat["raw_json"] = json.dumps(item, ensure_ascii=False)
    return flat

def main():
    access_token = ACCESS_TOKEN
    bbox = BBOX
    fields = FIELDS
    sequence_key = SEQUENCE_KEY

    all_rows = []
    page_count = 0

    if sequence_key:
        print(f"Querying Mapillary for images in sequence: {sequence_key}")
        # Use sequences endpoint and fetch images edge
        after = None
        while True:
            page_count += 1
            try:
                data = fetch_sequence_images(access_token, sequence_key, fields=fields, limit=100, after=after)
            except requests.HTTPError as e:
                print("HTTP error:", e)
                break
            except Exception as e:
                print("Error fetching sequence images:", e)
                break

            # In sequence responses, images are nested: data.get('images', {}).get('data')
            images_block = data.get("images") or {}
            items = images_block.get("data") or []
            if not items:
                print("No more images returned on page", page_count)
                break

            for item in items:
                img_id = item.get("id")
                camera_type = item.get("camera_type")
                lon, lat = get_coordinates(item)
                img_url = pick_image_url(item)

                # optional: filter spherical only (if camera_type is present)
                if camera_type and camera_type != "spherical":
                    continue

                if not img_url:
                    continue

                fname = f"{img_id}.jpg"
                out_path = os.path.join(OUT_DIR, fname)
                if not os.path.exists(out_path):
                    try:
                        download_file(img_url, out_path)
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"Failed to download {img_id}: {e}")
                        continue

                row = {
                    "id": img_id,
                    "filename": fname,
                    "lon": lon,
                    "lat": lat,
                    "camera_type": camera_type,
                    "image_url": img_url,
                    "sequence_key": item.get("sequence_key"),
                    "raw_item": item
                }
                all_rows.append(row)

            # paging for nested images - look for images.paging.cursors.after
            paging = images_block.get("paging") or {}
            cursors = paging.get("cursors") or {}
            after = cursors.get("after")
            if not after:
                next_url = paging.get("next")
                if next_url:
                    q = up.urlparse(next_url).query
                    qp = up.parse_qs(q)
                    if "after" in qp:
                        after = qp["after"][0]
                else:
                    break

            if page_count >= 500:
                print("Page limit reached; stopping.")
                break

    else:
        print("Querying Mapillary API for images in bbox:", bbox)
        after = None
        page_count = 0
        while True:
            page_count += 1
            try:
                data = fetch_images_page(access_token, bbox=bbox, fields=fields, limit=100, after=after)
            except requests.HTTPError as e:
                print("HTTP error:", e)
                break
            except Exception as e:
                print("Error fetching images:", e)
                break

            items = data.get("data") or []
            if not items:
                print("No more images returned on page", page_count)
                break

            for item in items:
                img_id = item.get("id")
                camera_type = item.get("camera_type")
                lon, lat = get_coordinates(item)
                img_url = pick_image_url(item)

                if camera_type:
                    if camera_type != "spherical":
                        continue

                if not img_url:
                    continue

                fname = f"{img_id}.jpg"
                out_path = os.path.join(OUT_DIR, fname)
                if not os.path.exists(out_path):
                    try:
                        download_file(img_url, out_path)
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"Failed to download {img_id}: {e}")
                        continue

                row = {
                    "id": img_id,
                    "filename": fname,
                    "lon": lon,
                    "lat": lat,
                    "camera_type": camera_type,
                    "image_url": img_url,
                    "sequence_key": item.get("sequence_key"),
                    "raw_item": item
                }
                all_rows.append(row)

            paging = data.get("paging") or {}
            cursors = paging.get("cursors") or {}
            after = cursors.get("after")
            if not after:
                next_url = paging.get("next")
                if not next_url:
                    break
                q = up.urlparse(next_url).query
                qp = up.parse_qs(q)
                if "after" in qp:
                    after = qp["after"][0]
                else:
                    break

            if page_count >= 200:
                print("Page limit reached; stopping.")
                break

    # If we collected results, flatten and write CSV with all available keys
    if all_rows:
        # Build flattened rows and union of columns
        flat_rows = []
        columns = set()
        for r in all_rows:
            item = r.pop("raw_item", {})
            # start with basic columns we have
            base = {
                "id": r.get("id"),
                "filename": r.get("filename"),
                "image_url": r.get("image_url"),
                "camera_type": r.get("camera_type"),
                "sequence_key": r.get("sequence_key"),
                "lon": r.get("lon"),
                "lat": r.get("lat")
            }
            # flatten the API item (this will include any additional fields returned)
            flattened_item = flatten_item(item)
            # Merge base and flattened_item (flattened_item may include duplicates; keep flattened_item values)
            merged = {**base, **flattened_item}
            flat_rows.append(merged)
            columns.update(merged.keys())

        # Ensure consistent column order (put common columns first)
        preferred = ["id", "filename", "image_url", "sequence_key", "camera_type", "lon", "lat"]
        other_cols = sorted([c for c in columns if c not in preferred])
        final_columns = preferred + other_cols

        # Write CSV
        df = pd.DataFrame(flat_rows, columns=final_columns)
        df.to_csv(CSV_PATH, index=False)
        print(f"Downloaded {len(flat_rows)} images. Metadata saved to {CSV_PATH}")
    else:
        print("No images downloaded. Check BBOX/SEQUENCE_KEY, token, and API fields.")

if __name__ == "__main__":
    main()
