import os
import pandas as pd
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_DIR = r"C:\Users\jsilw\Desktop\mapillary\code\mapillary\mapillary_images"      # where your downloaded images are
OUT_DIR = r"C:\Users\jsilw\Desktop\mapillary\code\mapillary\mapillary_faces"         # where cubemap face folders will be created
FACE_SIZE = None                  # if None, will default to image height // 1 (recommended when input is 2:1 equirectangular)

# Face generation configuration - set to False to skip generating that face
FACE_CONFIG = {
    "posx": True,   # Right face
    "negx": True,   # Left face
    "posy": False,   # Top face
    "negy": False,   # Bottom face
    "posz": True,   # Front face
    "negz": True    # Back face
}

# Only generate faces that are enabled
FACE_NAMES = [face for face, enabled in FACE_CONFIG.items() if enabled]

def get_face_coords(face, size):
    """
    Return normalized 3D direction vectors for cube face pixels.
    face: one of "posx","negx","posy","negy","posz","negz"
    size: face resolution (width=height=size)
    Returns: (dirs_x, dirs_y, dirs_z) each shape (size, size)
    
    CORRECTED: Fixed coordinate assignments so face names match actual directions
    """
    # create a,b in [-1,1], a increases to the right, b increases upward
    # note: j is row (y), i is col (x)
    rng = (np.arange(size) + 0.5) / size * 2.0 - 1.0  # centers
    a = np.tile(rng.reshape(1, size), (size, 1))      # shape (size, size) horizontal (x_face)
    b = np.tile(rng.reshape(size, 1), (1, size))      # shape (size, size) vertical (y_face)
    # but b needs to be flipped because image y increases downward; we want b positive = up
    b = np.flipud(b)

    #coordinate assignments
    # Standard cube map coordinate assignments
    if face == "posx":  # +X face (right)
        x =  np.ones_like(a); y = b; z = -a
    elif face == "negx":  # -X face (left)
        x = -np.ones_like(a); y = b; z =  a
    elif face == "posy":  # +Y face (top)
        x =  a; y =  np.ones_like(a); z =  b
    elif face == "negy":  # -Y face (bottom)
        x =  a; y = -np.ones_like(a); z = b
    elif face == "posz":  # +Z face (front)
        x =  a; y = b; z =  np.ones_like(a)
    elif face == "negz":  # -Z face (back)
        x = -a; y = b; z = -np.ones_like(a)

    # normalize
    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def sph_from_vec(x, y, z):
    """
    Convert 3D direction vector to spherical coordinates (lon, lat):
    lon in [-pi, pi], lat in [-pi/2, pi/2]
    Using convention:
      lon = atan2(X, Z)
      lat = asin(Y)
    where X-right, Y-up, Z-forward.
    """
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return lon, lat

def equirect_to_face(equ, face, face_size):
    """
    equ: input equirectangular image as numpy array (H, W, 3), BGR or RGB whichever passed
    face: face name
    face_size: integer
    returns: face image as numpy array same dtype as equ
    """
    H, W = equ.shape[:2]
    x, y, z = get_face_coords(face, face_size)
    lon, lat = sph_from_vec(x, y, z)

    # map to equirectangular pixel coordinates
    # u (x pixel) corresponds to longitude: -pi -> 0, +pi -> W
    # v (y pixel) corresponds to latitude: +pi/2 -> 0, -pi/2 -> H
    u = (lon + np.pi) / (2 * np.pi) * (W - 1)
    v = (np.pi/2 - lat) / np.pi * (H - 1)

    # cv2.remap wants single-channel float32 maps (x map -> cols, y map -> rows)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    # remap; use BORDER_WRAP horizontally and BORDER_REFLECT vertically via manual wrap of map_x
    # cv2.remap supports borderMode; for equirectangular horizontally wrap is desired.
    face_img = cv2.remap(equ, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return face_img

def process_image(path, out_dir, face_size=None):
    basename = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        print("Failed to read:", path)
        return
    H, W = img.shape[:2]
    if face_size is None:
        # typical equirectangular: W = 2*H; using face_size = H // 1 yields square faces roughly H x H
        face_size = H
    face_folder = os.path.join(out_dir, basename)
    os.makedirs(face_folder, exist_ok=True)

    # Only generate enabled faces
    for face in FACE_NAMES:
        face_img = equirect_to_face(img, face, face_size)
        out_path = os.path.join(face_folder, f"{basename}_{face}_{face_size}.jpg")
        # save with decent quality
        cv2.imwrite(out_path, face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    enabled_faces = ", ".join(FACE_NAMES)
    print(f"Processed {os.path.basename(path)} -> faces ({enabled_faces}) in {face_folder}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Print face configuration
    print("Face generation configuration:")
    for face, enabled in FACE_CONFIG.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {face}: {status}")
    print()
    
    img_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")) + glob.glob(os.path.join(SRC_DIR, "*.png")))
    if not img_paths:
        print("No images found in", SRC_DIR)
        return
    print(f"Found {len(img_paths)} images. Converting to cubemap faces...")
    for p in tqdm(img_paths):
        process_image(p, OUT_DIR, face_size=FACE_SIZE)
    print("Done. Faces saved to", OUT_DIR)

if __name__ == "__main__":
    main()


# Create metadata - Import ALL columns from original metadata
metadata_path = os.path.join("mapillary_images", "mapillary_images_metadata.csv")
if os.path.exists(metadata_path):
    meta = pd.read_csv(metadata_path)
    records = []
    
    for _, row in meta.iterrows():
        img_id = str(row["id"])
        
        # For each face, create a new record with all original metadata
        for face in FACE_NAMES:
            # Create a copy of the entire row as a dictionary
            record = row.to_dict()
            
            # Add face-specific information
            record["face"] = face
            record["filename"] = f"{img_id}/{img_id}_{face}_1024.jpg"
            
            records.append(record)
    
    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "cubemap_faces_metadata.csv")
    df.to_csv(out_csv, index=False)
    print(f"Cubemap metadata saved to {out_csv}")
    print(f"Total records: {len(df)}, Columns: {list(df.columns)}")
else:
    print("Could not find mapillary_images_metadata.csv to link coordinates.")