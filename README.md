# 🖼️📜  Image Captioning

An end‑to‑end pipeline that turns images into captions using a **pre‑trained InceptionV3** encoder and an **LSTM** decoder.  
The repository is kept intentionally light: only code and **100 sample JPEGs** are included so anyone can run the full pipeline quickly.

---

## ✨ Features
| Stage | Script | Purpose |
|-------|--------|---------|
| **Feature extraction** | `feature_extraction.py` | Loads InceptionV3 (`pooling="avg"`) and produces a 2048‑dim feature vector. |
| **Batch extract** | `extract_all_features.py` | Runs the extractor on every image in `sample_images/` and stores `.npy` files in `features/`. |
| **Caption prep** | `prepare_captions.py` | Cleans & tokenizes COCO captions → `tokenizer.pkl`, `meta.pkl`. |
| **Sequence build** | `create_sequences.py` | Builds `X_image.npy`, `X_seq.npy`, `y_word.npy`. |
| **Model train** | `train_model.py` | Trains a Dense + LSTM captioning model. |
| **Inference / test** | `test_caption.py` | Loads weights and generates captions for images. |

---

## 📂 Directory Tree
```
image-captioning/
├── feature_extraction.py
├── extract_all_features.py
├── prepare_captions.py
├── create_sequences.py
├── train_model.py
├── test_caption.py
├── sample_images/          # 100 tiny JPEGs
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/your-username/image-captioning.git
cd image-captioning

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the pipeline
```bash
# 1. Extract features
python extract_all_features.py
# 2. Prepare & tokenize captions
python prepare_captions.py
# 3. Create training sequences
python create_sequences.py
# 4. Train the model
python train_model.py
```

### Test / Generate Captions
```bash
python test_caption.py                 # captions every image in sample_images/
python test_caption.py --image sample_images/your_image.jpg   # single image
```

---
---

## 📝 Requirements
```
tensorflow>=2.10
numpy>=1.23
pandas>=1.5
nltk>=3.7
opencv-python>=4.7
```

Install with `pip install -r requirements.txt`.

---

## 🖋️ License
MIT © Janu Singh
