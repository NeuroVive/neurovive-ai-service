# NeuroSpiral — Image-Based PD Detection

## Overview

**NeuroSpiral** classifies hand-drawn spiral and wave images as either Healthy Control (`HC`) or Parkinson's Disease (`PD`). It uses a **hybrid architecture** that fuses deep CNN features from EfficientNet-B0 with classical handcrafted features (HOG + LBP), followed by a fitted feature reduction pipeline (VarianceThreshold → StandardScaler → PCA).

---

## Why Hybrid Features?

| Feature type                              | What it captures                          | Why it helps                                             |
| ----------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| **EfficientNet-B0 CNN**                   | Global semantic structure, shape patterns | Learns high-level drawing quality from data              |
| **HOG (Histogram of Oriented Gradients)** | Edge direction distributions, local shape | Directly encodes tremor-induced irregularity in strokes  |
| **LBP (Local Binary Pattern)**            | Fine-grained texture regularity           | Captures micro-texture degradation from motor impairment |

Combining both streams consistently outperforms either alone on small medical image datasets.

---

## Model Architecture

```
Input Image (any size)
        │
   cv2.resize → (224, 224, 3)
        │
   ┌────┴──────────────────────────────┐
   │                                   │
   │  Image Branch                     │  Feature Branch
   │  EfficientNet-B0                  │  HOG + LBP extraction
   │  (pretrained, global_pool="avg")  │  → 6,094-d raw vector
   │  → 1280-d embedding               │      │
   │                                   │  VarianceThreshold → StandardScaler → PCA
   │                                   │  → D-d reduced vector
   │                                   │      │
   │                                   │  MLP: D → 256 → 128
   │                                   │  (BN + GELU + Dropout × 2)
   └────────────┬──────────────────────┘
                │  Gating: gate = sigmoid(cat(img, feat))
                │  fused  = cat(img, feat * gate)
                │
           Fusion Head
       Linear(1280+128 → 256) → GELU → Dropout(0.5)
       Linear(256 → 64)        → GELU → Dropout(0.5)
       Linear(64  → 1)
                │
           Raw Logit → sigmoid → label
```

---

## `NeuroSpiralPredictor` — Inference Class

**File:** `src/inference/neurospiral_predictor.py`

### Constructor

```python
import joblib
from src.inference.neurospiral_predictor import NeuroSpiralPredictor

reducers = joblib.load("checkpoint/feature_reducers.pkl")

predictor = NeuroSpiralPredictor(
    model_path="checkpoint/spiral_best_model.onnx",
    variance_selector=reducers["variance_selector"],
    scaler=reducers["scaler"],
    pca=reducers["pca"],
)
```

Initializes an `onnxruntime.InferenceSession` and stores the fitted feature reduction objects. The reducers **must** be the same objects produced during training via `fit_feature_reducers()`.

Raises `RuntimeError` if the ONNX file cannot be loaded.

### `predict(img)`

```python
label, probability = predictor.predict(img)  # img: BGR np.ndarray
```

| Parameter | Type         | Description                                                   |
| --------- | ------------ | ------------------------------------------------------------- |
| `img`     | `np.ndarray` | BGR image from `cv2.imread` or `cv2.imdecode`, any resolution |

**Returns:** `Tuple[str, float]`

- `label` → `"HC"` if `probability ≥ 0.5`, `"PD"` if `probability < 0.5`
- `probability` → sigmoid of raw logit, rounded to 4 decimal places

---

## Preprocessing Pipeline (Inside `predict`)

### Step 1 — Resize

```python
img_input = cv2.resize(img, IMAGE_SIZE)  # (224, 224)
```

### Step 2 — Grayscale guard

```python
if img_input.ndim == 2:
    img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
```

### Step 3 — Image tensor

```python
img_tensor = img_input.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 224, 224)
img_tensor = (img_tensor / 255.0).astype(np.float32)
```

### Step 4 — HOG features

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, IMAGE_SIZE)

hog_features = hog(
    gray,
    orientations=HOG_ORIENTATIONS,       # 9
    pixels_per_cell=HOG_PIXELS_PER_CELL, # (16, 16)
    cells_per_block=HOG_CELLS_PER_BLOCK, # (2, 2)
    feature_vector=True,
)
# Output: 1D array of length 6,084
```

### Step 5 — LBP features

```python
lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_HIST_BINS, range=LBP_HIST_RANGE)
# bins=10, range=(0, 10)
lbp_hist = lbp_hist.astype("float32") / (lbp_hist.sum() + 1e-6)
# Output: 1D array of length 10
```

### Step 6 — Feature Reduction Pipeline

```python
raw_features = np.concatenate([hog_features, lbp_hist]).reshape(1, -1)
# shape: (1, 6094)

reduced = variance_selector.transform(raw_features)  # removes low-variance features
reduced = scaler.transform(reduced)                  # StandardScaler normalization
reduced = pca.transform(reduced)                     # PCA(n_components=0.95)
# shape: (1, D)  — D determined at training time
```

> The reduction pipeline must be loaded from `feature_reducers.pkl`, which is saved during training. D varies depending on training data.

### Step 7 — ONNX forward pass

```python
inputs = {
    input_names[0]: img_tensor,  # (1, 3, 224, 224)
    input_names[1]: reduced,     # (1, D)
}
logit = session.run([output_name], inputs)[0].item()
probability = 1 / (1 + np.exp(-logit))
label = "PD" if probability < 0.5 else "HC"
```

---

## Feature Dimensions

| Feature                  | Computation                              | Length    |
| ------------------------ | ---------------------------------------- | --------- |
| HOG                      | 9 orientations × 13 × 13 × 4 cells/block | 6,084     |
| LBP histogram            | 10 uniform bins, L1-normalized           | 10        |
| **Raw total**            |                                          | **6,094** |
| After VarianceThreshold  | removes features with variance < 0.01    | < 6,094   |
| After StandardScaler     | zero mean, unit variance                 | same      |
| **After PCA (95% var.)** | D — determined at training time          | **D**     |

---

## ONNX Model Details

| Property      | Value                                        |
| ------------- | -------------------------------------------- |
| Input 0 name  | `"image"`                                    |
| Input 0 shape | `(batch_size, 3, 224, 224)`                  |
| Input 1 name  | `"math_features"`                            |
| Input 1 shape | `(batch_size, D)` — D set by PCA at training |
| Output name   | `"logit"`                                    |
| Output shape  | `(batch_size, 1)`                            |
| Dynamic axes  | batch dimension for all inputs/output        |
| Opset         | 18                                           |

---

## Required Checkpoint Files

```
checkpoint/
├── spiral_best_model.onnx   ← ONNX exported model
└── feature_reducers.pkl     ← fitted VarianceThreshold / StandardScaler / PCA
```

The `feature_reducers.pkl` is saved during training:

```python
import joblib

joblib.dump({
    "variance_selector": vt,
    "scaler": scaler,
    "pca": pca,
}, REDUCERS_PATH)
```

---

## Usage Example

```python
import cv2
import joblib
from src.inference.neurospiral_predictor import NeuroSpiralPredictor

reducers = joblib.load("checkpoint/feature_reducers.pkl")

predictor = NeuroSpiralPredictor(
    model_path="checkpoint/spiral_best_model.onnx",
    variance_selector=reducers["variance_selector"],
    scaler=reducers["scaler"],
    pca=reducers["pca"],
)

img = cv2.imread("patient_spiral.png")
label, probability = predictor.predict(img)

print(f"Label      : {label}")        # HC or PD
print(f"Probability: {probability}")  # e.g. 0.8734
```

---

## Performance

| Metric             | Value             |
| ------------------ | ----------------- |
| Best Validation F1 | 93.58% (Epoch 13) |
| Test Accuracy      | 87.80%            |
| Test F1 Score      | 87.80%            |
| Training Epochs    | 15                |
