import os
import glob
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add
from feature_extraction import extract_features  # your pooling='avg' version

# ---- Config ----
SAMPLE_DIR     = 'sample_images'
TOKENIZER_PATH = 'prepared_data/tokenizer.pkl'
META_PATH      = 'prepared_data/meta.pkl'
WEIGHTS_PATH   = 'image_caption_model.h5'
# ----------------

# 1) Gather all sample images dynamically
image_paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.jpg')))
if not image_paths:
    raise FileNotFoundError(f"No .jpg files found in {SAMPLE_DIR}")
print(f"Found {len(image_paths)} images in {SAMPLE_DIR}.")

# 2) Load tokenizer and metadata
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# 3) Rebuild the same model architecture
# Image branch
img_input = Input(shape=(2048,), name='image_input')
img_dense = Dense(256, activation='relu', name='image_dense')(img_input)

# Sequence branch
seq_input = Input(shape=(max_length,), name='seq_input')
seq_embed = Embedding(input_dim=vocab_size,
                      output_dim=256,
                      mask_zero=True,
                      name='seq_embedding')(seq_input)
seq_lstm  = LSTM(256, name='seq_lstm')(seq_embed)

# Combine and output
merged = Add(name='add_features')([img_dense, seq_lstm])
output = Dense(vocab_size, activation='softmax', name='output')(merged)

model = Model(inputs=[img_input, seq_input], outputs=output, name='image_caption_model')
model.load_weights(WEIGHTS_PATH)

# 4) Caption generator
def generate_caption(model, tokenizer, image_path, max_length):
    feats = extract_features(image_path).reshape(1, -1)  # (1, 2048)
    seq = [tokenizer.word_index['startseq']]
    for _ in range(max_length):
        yhat = model.predict([feats, np.array(seq).reshape(1, -1)], verbose=0)
        word_id = int(np.argmax(yhat))
        word = tokenizer.index_word.get(word_id)
        if not word or word == 'endseq':
            break
        seq.append(word_id)
    # convert IDs to words (skip startseq/endseq)
    return ' '.join(tokenizer.index_word[i] 
                    for i in seq 
                    if i not in (tokenizer.word_index['startseq'],
                                 tokenizer.word_index['endseq']))

# 5) Run on each image
for img_path in image_paths:
    caption = generate_caption(model, tokenizer, img_path, max_length)
    print(f"\nImage: {os.path.basename(img_path)}")  
    print(f"Caption: {caption}")
