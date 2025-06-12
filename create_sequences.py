import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Config ----
FEATURES_DIR = 'features'
OUTPUT_DIR   = 'training_data'
PREPARED_DIR = 'prepared_data'
# ----------------

# 1) Load cleaned captions dict (from prepare_captions.py)
#    This requires you have `clean_caps` available as a Python object.
#    If you stored it to disk, you could un-pickle it here instead.
from prepare_captions import clean_caps  

# 2) Load tokenizer & metadata
with open(os.path.join(PREPARED_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)
with open(os.path.join(PREPARED_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
max_length = meta['max_length']

# 3) Prepare output lists
X_image, X_seq, y_word = [], [], []

# 4) Loop through each image ID and its captions
for img_id, captions in clean_caps.items():
    # Derive the exact feature filename:
    fname = f"COCO_train2014_{int(img_id):012d}.npy"
    feature_path = os.path.join(FEATURES_DIR, fname)
    if not os.path.exists(feature_path):
        # Skip if feature file wasn't generated
        continue

    # Load the pre-extracted feature array
    feats = np.load(feature_path)  # shape (1, 8, 8, 2048)

    # For each cleaned caption, build (image_feature, text_sequence, next_word) samples
    for cap in captions:
        seq = tokenizer.texts_to_sequences([cap])[0]  # e.g. [1, 34, 56, ..., 2]
        for i in range(1, len(seq)):
            in_seq  = seq[:i]      # input words so far
            out_word = seq[i]       # next word to predict

            # pad the input sequence to max_length
            in_seq_padded = pad_sequences([in_seq], maxlen=max_length)[0]

            # one-hot encode the output word
            out_vec = np.zeros(vocab_size, dtype='float32')
            out_vec[out_word] = 1.0

            # append to our lists
            X_image.append(feats)         # shape (1,8,8,2048)
            X_seq.append(in_seq_padded)   # shape (max_length,)
            y_word.append(out_vec)        # shape (vocab_size,)

# 5) Convert lists to arrays
if not X_image:
    raise RuntimeError("No training samples were created—check FEATURES_DIR and clean_caps IDs.")

# Stack image features (drop that extra leading dimension)
X_image = np.vstack([x.reshape(-1) for x in X_image])  
# Now shape (n_samples, 8*8*2048); you can reshape later if needed

X_seq   = np.vstack(X_seq)     # shape (n_samples, max_length)
y_word  = np.vstack(y_word)    # shape (n_samples, vocab_size)

# 6) Save the training data
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, 'X_image.npy'), X_image)
np.save(os.path.join(OUTPUT_DIR, 'X_seq.npy'),   X_seq)
np.save(os.path.join(OUTPUT_DIR, 'y_word.npy'),  y_word)

print(f"Saved training data: \n"
      f"  X_image.npy (shape {X_image.shape})\n"
      f"  X_seq.npy   (shape {X_seq.shape})\n"
      f"  y_word.npy  (shape {y_word.shape})")
