
<body>
  <h1>Mapillary 360° Image Downloader</h1>
  <p>A Python script to download <strong>360° (spherical)</strong> Mapillary images either within a geographic bounding box (BBOX) or from a specific Mapillary sequence. The script saves image files and a flattened metadata CSV for easy analysis.</p>

  <h2>Features</h2>
  <ul>
    <li>Downloads only <strong>spherical (360°)</strong> images.</li>
    <li>Supports two modes:
      <ul>
        <li><strong>BBOX mode</strong> — search by longitude/latitude bounding box.</li>
        <li><strong>Sequence mode (THIS DOESN'T WORK)</strong> — download all images from a sequence key (set <code>SEQUENCE_KEY</code>).</li>
      </ul>
    </li>
    <li>Saves:
      <ul>
        <li>Full-resolution or fallback thumbnail URLs</li>
        <li>Image files (<code>.jpg</code>)</li>
        <li>Complete flattened metadata CSV (<code>mapillary_images_metadata.csv</code>)</li>
      </ul>
    </li>
    <li>Automatically flattens Mapillary's nested JSON fields and stores the original raw JSON per item.</li>
  </ul>

  <h2>Requirements</h2>
  <p>Install the required Python packages:</p>
  <pre><code>pip install requests pandas tqdm</code></pre>

  <h2>Configuration</h2>
  <p>Open the script and edit the <strong>USER CONFIG</strong> section:</p>
  <pre><code>ACCESS_TOKEN = "YOUR_MAPILLARY_ACCESS_TOKEN"
SEQUENCE_KEY = None  # or e.g. "s-ABC123..."
BBOX = (-102.4486, 44.0658, -102.4422, 44.0725)
OUT_DIR = "mapillary_images"</code></pre>
  <ul>
    <li><code>ACCESS_TOKEN</code> — your Mapillary API token.</li>
    <li><code>SEQUENCE_KEY</code> — set to a sequence key to download a single sequence; set to <code>None</code> to use BBOX mode.</li>
    <li><code>BBOX</code> — bounding box in the format <code>(lon_min, lat_min, lon_max, lat_max)</code>.</li>
    <li><code>OUT_DIR</code> — folder where images and CSV will be saved.</li>
  </ul>

  <h2>Usage</h2>
  <p>Run the script from the command line:</p>
  <pre><code>python mapillary_download_360_with_sequence_filter.py</code></pre>

  <h2>Output</h2>
  <p>Files created by the script:</p>
  <ul>
    <li>Images saved to: <code>mapillary_images/</code></li>
    <li>Metadata CSV saved to: <code>mapillary_images/mapillary_images_metadata.csv</code></li>
  </ul>

  <h2>CSV Contents</h2>
  <p>Each CSV row includes:</p>
  <ul>
    <li><strong>ID</strong>, <strong>filename</strong>, <strong>latitude/longitude</strong>, <strong>sequence key</strong></li>
    <li>The selected download URL used for the image</li>
    <li>EXIF and other Mapillary fields (flattened)</li>
    <li><code>raw_json</code> — the full API response for the image</li>
  </ul>

  <h2>Notes</h2>
  <div class="note">
    <ul>
      <li>Only images where <code>camera_type == "spherical"</code> are downloaded.</li>
      <li>The script automatically pages through API results until completion (subject to the script's page limits).</li>
      <li>If <code>SEQUENCE_KEY</code> is set, sequence mode takes precedence over the BBOX search.</li>
      <li>Mapillary API rate limits may apply; the script includes small delays between downloads.</li>
    </ul>
  </div>

 
