  <h1>Mapillary 360° → Cubemap Face Generator</h1>
  <p>
    This script converts <strong>360° equirectangular Mapillary images</strong> into 
    <strong>cubemap faces</strong> (<code>posx</code>, <code>negx</code>, <code>posy</code>, <code>negy</code>, <code>posz</code>, <code>negz</code>)
    for use in machine learning, AR/VR, environment mapping, and scene reconstruction.
    It also generates a metadata CSV linking each cubemap face to the original Mapillary metadata.
  </p>

  <h2>Features</h2>
  <ul>
    <li>Converts 2:1 equirectangular spherical images into cubemap faces.</li>
    <li>Select which cube faces to generate via <code>FACE_CONFIG</code>.</li>
    <li>Automatically computes 3D direction vectors and correct spherical coordinates.</li>
    <li>Uses <code>cv2.remap</code> for high-quality sampling and proper wrap-around.</li>
    <li>Exports a <strong>cubemap_faces_metadata.csv</strong> file containing:
      <ul>
        <li>full original Mapillary metadata</li>
        <li>face name</li>
        <li>face filename</li>
      </ul>
    </li>
  </ul>

  <h2>Requirements</h2>
  <pre><code>pip install numpy opencv-python pandas pillow tqdm</code></pre>

  <h2>Configuration</h2>
  <p>Set the input/output directories and options in the script:</p>

  <pre><code>SRC_DIR = r"C:\...\mapillary_images"
OUT_DIR = r"C:\...\mapillary_faces"
FACE_SIZE = None  # defaults to image height (recommended)
</code></pre>

  <h3>Face enabling</h3>
  <p>Choose which cube faces to generate:</p>

  <pre><code>FACE_CONFIG = {
    "posx": True,   # Right
    "negx": True,   # Left
    "posy": False,  # Top
    "negy": False,  # Bottom
    "posz": True,   # Front
    "negz": True    # Back
}
</code></pre>

  <h2>Usage</h2>
  <p>Run the script:</p>

  <pre><code>python mapillary_to_cubemap.py</code></pre>

  <h2>Output</h2>
  <ul>
    <li>Cubemap faces stored in:
      <br><code>mapillary_faces/&lt;image-id&gt;/</code></li>
    <li>Metadata CSV stored at:
      <br><code>mapillary_faces/cubemap_faces_metadata.csv</code></li>
  </ul>

  <h2>Cubemap Metadata Contents</h2>
  <p>Each record includes:</p>
  <ul>
    <li><strong>All original Mapillary columns</strong></li>
    <li><strong>face</strong> (posx, negx, posz, etc.)</li>
    <li><strong>filename</strong> (path to the generated face)</li>
    <li><strong>id</strong> (Mapillary image ID)</li>
  </ul>

  <h2>Notes</h2>
  <div class="note">
    <ul>
      <li>The script assumes the input images are valid 360° equirectangular with aspect ratio 2:1.</li>
      <li>Only faces enabled in <code>FACE_CONFIG</code> are produced.</li>
      <li>Output face resolution defaults to the height of the equirectangular image unless overridden.</li>
      <li>This script integrates directly with <code>mapillary_images_metadata.csv</code> from the Mapillary downloader tool.</li>
    </ul>
  </div>
