import os
import numpy as np
from feature_extraction import extract_features

# ---- CONFIG ----
IMAGES_DIR = 'sample_images'    # or 'sample_images' if you’re prototyping
OUTPUT_DIR = 'features'
# ------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(IMAGES_DIR):
    if not fname.lower().endswith('.jpg'):
        continue

    img_id   = os.path.splitext(fname)[0]
    img_path = os.path.join(IMAGES_DIR, fname)

    # 1) extract features
    feats = extract_features(img_path)  # should be shape (1,8,8,2048)

    # 2) skip if empty
    if feats.size == 0:
        print(f"⚠️  Skipping {fname}: extracted feature array is empty.")
        continue

    # 3) save inside a try/except to catch write errors
    out_path = os.path.join(OUTPUT_DIR, f"{img_id}.npy")
    try:
        np.save(out_path, feats)
    except OSError as e:
        print(f"❌  Failed to write {out_path}: {e}")
        # you can choose to break or continue
        continue

print("✅  Feature extraction complete.")
