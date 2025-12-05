import os
import glob
import cv2
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm
import pandas as pd
import json
import piexif

# -------- CONFIG --------
SRC_DIR = "image"          # source equirectangular images
OUT_DIR = "faces"          # outputs (faces, csv, geojson)
FACE_SIZE = None           # if None -> H (image height)
METADATA_CSV = os.path.join(OUT_DIR, "metadata.csv")
GEOJSON_PATH = os.path.join(OUT_DIR, "images_footprints_epsg4326.geojson")

# Face generation configuration - set to False to skip generating that face
FACE_CONFIG = {
    "posx": True,   # Right face
    "negx": True,   # Left face
    "posy": False,   # Top face
    "negy": True,   # Bottom face
    "posz": True,   # Front face
    "negz": False    # Back face
}

# Only generate faces that are enabled
FACE_NAMES = [face for face, enabled in FACE_CONFIG.items() if enabled]

# Target projection for GeoJSON
SRC_CRS = "EPSG:4326"   # input lat/lon WGS84
DST_CRS = "EPSG:4326"   # Output also in WGS84 lat/lon

# -------- EXIF / GPS helpers --------

def _rational_to_float(r):
    try:
        return float(r[0]) / float(r[1])
    except Exception:
        try:
            return float(r)
        except Exception:
            return None

def _dms_to_deg(dms):
    # dms -> decimal degrees, dms as ((d_num,d_den),(m_num,m_den),(s_num,s_den))
    d = _rational_to_float(dms[0])
    m = _rational_to_float(dms[1])
    s = _rational_to_float(dms[2])
    if d is None or m is None or s is None:
        return None
    return d + (m / 60.0) + (s / 3600.0)

def _deg_to_dms_rational(dec_deg):
    """Convert decimal degrees to rational DMS tuples for EXIF (num,den)"""
    sign = 1
    if dec_deg < 0:
        sign = -1
        dec_deg = -dec_deg
    d = int(dec_deg)
    m_full = (dec_deg - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    # return as tuples of (num, den)
    return ((d, 1), (m, 1), (int(round(s * 10000)), 10000))  # seconds with 1/10000 precision

def get_gps_from_exif(path):
    """
    Returns (lat, lon, altitude) in decimal degrees/meters if available, else (None, None, None).
    """
    try:
        img = Image.open(path)
        exif_raw = img._getexif()
        if not exif_raw:
            return None, None, None

        exif = {}
        for tag_id, value in exif_raw.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif[tag] = value

        gps_info = exif.get("GPSInfo")
        if not gps_info:
            return None, None, None

        gps = {}
        for key in gps_info.keys():
            name = ExifTags.GPSTAGS.get(key, key)
            gps[name] = gps_info[key]

        lat = None; lon = None; alt = None
        if "GPSLatitude" in gps and "GPSLatitudeRef" in gps:
            lat_deg = _dms_to_deg(gps["GPSLatitude"])
            if lat_deg is not None and gps["GPSLatitudeRef"] in ("S", "s"):
                lat_deg = -lat_deg
            lat = lat_deg

        if "GPSLongitude" in gps and "GPSLongitudeRef" in gps:
            lon_deg = _dms_to_deg(gps["GPSLongitude"])
            if lon_deg is not None and gps["GPSLongitudeRef"] in ("W", "w"):
                lon_deg = -lon_deg
            lon = lon_deg

        if "GPSAltitude" in gps:
            a = _rational_to_float(gps["GPSAltitude"])
            if a is not None:
                ref = gps.get("GPSAltitudeRef", 0)
                try:
                    ref_val = int(ref)
                except Exception:
                    ref_val = 0
                alt = -a if ref_val == 1 else a

        return lat, lon, alt
    except Exception:
        return None, None, None

# -------- cubemap math (CORRECTED coordinate assignments) --------

def get_face_coords(face, size):
    rng = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    a = np.tile(rng.reshape(1, size), (size, 1))
    b = np.tile(rng.reshape(size, 1), (1, size))
    b = np.flipud(b)

    # CORRECTED: Fixed coordinate assignments so face names match actual directions
    if face == "posx":
        x =  np.ones_like(a); y =  b; z =  a  # Changed: z = a (was -a)
    elif face == "negx":
        x = -np.ones_like(a); y =  b; z = -a  # Changed: z = -a (was a)
    elif face == "posy":
        x =  a; y =  np.ones_like(a); z =  b  # Changed: z = b (was -b)
    elif face == "negy":
        x =  a; y = -np.ones_like(a); z = -b  # Changed: z = -b (was b)
    elif face == "posz":
        x = -a; y =  b; z =  np.ones_like(a)  # Unchanged
    elif face == "negz":
        x = a; y =  b; z = -np.ones_like(a)  # Unchanged
    else:
        raise ValueError("Unknown face: " + str(face))

    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm; y /= norm; z /= norm
    return x, y, z

def sph_from_vec(x, y, z):
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return lon, lat

def equirect_to_face(equ, face, face_size):
    H, W = equ.shape[:2]
    x, y, z = get_face_coords(face, face_size)
    lon, lat = sph_from_vec(x, y, z)
    u = (lon + np.pi) / (2 * np.pi) * (W - 1)
    v = (np.pi/2 - lat) / np.pi * (H - 1)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    face_img = cv2.remap(equ, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return face_img

# -------- EXIF embedding --------

def create_gps_ifd(lat, lon, alt):
    """
    Build GPS IFD dict for piexif using decimal lat/lon/alt.
    Returns {} if lat/lon None.
    """
    if lat is None or lon is None:
        return {}

    lat_ref = b'N' if lat >= 0 else b'S'
    lon_ref = b'E' if lon >= 0 else b'W'
    lat_dms = _deg_to_dms_rational(abs(lat))
    lon_dms = _deg_to_dms_rational(abs(lon))

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: lon_dms,
    }

    if alt is not None:
        # altitude reference: 0 = above sea level, 1 = below sea level
        if alt < 0:
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 1
            alt_val = (int(round(-alt * 100)), 100)  # store with 2 decimals
        else:
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0
            alt_val = (int(round(alt * 100)), 100)
        gps_ifd[piexif.GPSIFD.GPSAltitude] = alt_val

    return gps_ifd

# -------- metadata CSV / GeoJSON helpers --------

def append_metadata_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)

def write_geojson(features, path):
    # features: list of GeoJSON feature dicts (in EPSG:4326 lon/lat coords)
    fc = {
        "type": "FeatureCollection",
        "crs": { "type": "name", "properties": { "name": DST_CRS } },
        "features": features
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

# -------- main processing (combines everything) --------

def process_image(path, out_dir, face_size=None, geo_features_accum=None):
    basename = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        print("Failed to read:", path)
        return
    H, W = img.shape[:2]
    if face_size is None:
        face_size = H

    # extract GPS before any conversions
    lat, lon, alt = get_gps_from_exif(path)

    face_folder = os.path.join(out_dir, basename)
    os.makedirs(face_folder, exist_ok=True)

    # create GPS IFD once per source image
    gps_ifd = create_gps_ifd(lat, lon, alt)

    # Only generate enabled faces
    for face in FACE_NAMES:
        face_img = equirect_to_face(img, face, face_size)
        # Convert BGR->RGB for PIL
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        out_fname = f"{basename}_{face}.jpg"
        out_path = os.path.join(face_folder, out_fname)

        # prepare EXIF dict with GPS IFD (leave other tags empty)
        exif_dict = {"0th":{}, "Exif":{}, "GPS": gps_ifd, "1st":{}, "thumbnail": None}
        exif_bytes = piexif.dump(exif_dict)

        # save with EXIF via PIL
        pil_img.save(out_path, "JPEG", quality=95, exif=exif_bytes)

    # append metadata csv row (one per source image)
    row = {
        "basename": basename,
        "source_path": path,
        "face_size": face_size,
        "latitude": lat,
        "longitude": lon,
        "altitude_m": alt,
        "faces_generated": ", ".join(FACE_NAMES)
    }
    append_metadata_row(METADATA_CSV, row)

    # add GeoJSON feature (point) in EPSG:4326 if GPS exists
    if lat is not None and lon is not None and geo_features_accum is not None:
        # For EPSG:4326, use lon/lat directly (no transformation needed)
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON standard: [longitude, latitude]
            },
            "properties": {
                "basename": basename,
                "source_path": path,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "crs": DST_CRS,
                "faces_generated": ", ".join(FACE_NAMES)
            }
        }
        geo_features_accum.append(feat)

    enabled_faces = ", ".join(FACE_NAMES)
    print(f"Processed {os.path.basename(path)} -> faces ({enabled_faces}) in {face_folder}  (GPS: {lat}, {lon}, {alt})")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")) + glob.glob(os.path.join(SRC_DIR, "*.jpeg")) + glob.glob(os.path.join(SRC_DIR, "*.png")))

    if not img_paths:
        print("No images found in", SRC_DIR)
        return

    # reset metadata CSV & geojson
    if os.path.exists(METADATA_CSV):
        os.remove(METADATA_CSV)
    if os.path.exists(GEOJSON_PATH):
        os.remove(GEOJSON_PATH)

    # Print face configuration
    print("Face generation configuration:")
    for face, enabled in FACE_CONFIG.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {face}: {status}")
    print()

    geo_feats = []
    print(f"Found {len(img_paths)} images. Converting to cubemap faces and embedding GPS EXIF...")
    for p in tqdm(img_paths):
        process_image(p, OUT_DIR, face_size=FACE_SIZE, geo_features_accum=geo_feats)

    # write geojson (features are in EPSG:4326 coords)
    write_geojson(geo_feats, GEOJSON_PATH)

    print("Done.")
    print("Faces with embedded GPS EXIF saved under:", OUT_DIR)
    print("Metadata CSV:", METADATA_CSV)
    print("GeoJSON (EPSG:4326):", GEOJSON_PATH)

if __name__ == "__main__":
    main()