import numpy as np
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add

# ---- Config Paths ----
PREPARED_DIR   = 'prepared_data'
TRAINING_DIR   = 'training_data'
MODEL_OUT_PATH = 'image_caption_model.h5'
# -----------------------

# 1) Load metadata
with open(f'{PREPARED_DIR}/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
max_length = meta['max_length']

# 2) Load training data
X_image = np.load(f'{TRAINING_DIR}/X_image.npy')   # shape (n_samples, 2048)
X_seq   = np.load(f'{TRAINING_DIR}/X_seq.npy')     # shape (n_samples, max_length)
y_word  = np.load(f'{TRAINING_DIR}/y_word.npy')    # shape (n_samples, vocab_size)

# 3) Define the captioning model
# 3a) Image feature input branch
img_input = Input(shape=(X_image.shape[1],), name='image_input')
img_dense = Dense(256, activation='relu', name='image_dense')(img_input)

# 3b) Text sequence input branch
seq_input = Input(shape=(max_length,), name='seq_input')
seq_embed = Embedding(input_dim=vocab_size,
                      output_dim=256,
                      mask_zero=True,
                      name='seq_embedding')(seq_input)
seq_lstm  = LSTM(256, name='seq_lstm')(seq_embed)

# 3c) Combine branches and output
merged   = Add(name='add_features')([img_dense, seq_lstm])
output   = Dense(vocab_size, activation='softmax', name='output')(merged)

model = Model(inputs=[img_input, seq_input], outputs=output, name='image_caption_model')
model.compile(optimizer=Adam(), loss='categorical_crossentropy')
model.summary()

# 4) Train the model
# Use fewer epochs if you're on small sample set
model.fit(
    [X_image, X_seq],
    y_word,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# 5) Save the model
model.save(MODEL_OUT_PATH)
print(f'Model saved to {MODEL_OUT_PATH}')
