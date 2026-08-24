# Inference Guide

## Overview

Both predictors can be used directly in Python without running the API server. This is useful for batch processing, scripting, or integrating NeuroVive into a larger pipeline.

| Predictor              | Input                   | Class                                    |
| ---------------------- | ----------------------- | ---------------------------------------- |
| `NeuroSpiralPredictor` | BGR image (NumPy array) | `src/inference/neurospiral_predictor.py` |
| `NeuroVoxPredictor`    | Path to `.wav` file     | `src/inference/neurovox_predictor.py`    |

---

## Prerequisites

Both ONNX model files and the fitted feature reducers must exist before running inference:

```
checkpoint/
├── spiral_best_model.onnx
├── voice_best_model.onnx
└── feature_reducers.pkl      ← fitted VarianceThreshold / StandardScaler / PCA
```

The `feature_reducers.pkl` file is generated at the end of NeuroSpiral training:

```python
import joblib

joblib.dump({
    "variance_selector": vt,
    "scaler": scaler,
    "pca": pca,
}, "checkpoint/feature_reducers.pkl")
```

---

## Quick Start

### Image Inference (NeuroSpiral)

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

img = cv2.imread("spiral_drawing.png")   # BGR NumPy array
label, probability = predictor.predict(img)

print(f"Label      : {label}")           # 'HC' or 'PD'
print(f"Probability: {probability:.4f}") # e.g. 0.8734
```

### Voice Inference (NeuroVox)

```python
from src.inference.neurovox_predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/voice_best_model.onnx")

label, probability = predictor.predict("voice_recording.wav")

print(f"Label      : {label}")           # 'HC' or 'PD'
print(f"Probability: {probability:.4f}") # e.g. 0.3102
```

---

## Batch Processing

### Batch — Images

```python
import os
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

image_dir = "data/spirals/"
results = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(os.path.join(image_dir, fname))
        label, prob = predictor.predict(img)
        results.append({"file": fname, "label": label, "probability": prob})

for r in results:
    print(f"{r['file']:<35} → {r['label']}  ({r['probability']:.4f})")
```

### Batch — Audio

```python
import os
from src.inference.neurovox_predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/voice_best_model.onnx")

audio_dir = "data/recordings/"
results = []

for fname in os.listdir(audio_dir):
    if fname.lower().endswith(".wav"):
        path = os.path.join(audio_dir, fname)
        label, prob = predictor.predict(path)
        results.append({"file": fname, "label": label, "probability": prob})

for r in results:
    print(f"{r['file']:<35} → {r['label']}  ({r['probability']:.4f})")
```

---

## Multi-Modal Prediction

Use both models on the same patient and compare or combine their outputs:

```python
import cv2
import joblib
from src.inference.neurospiral_predictor import NeuroSpiralPredictor
from src.inference.neurovox_predictor import NeuroVoxPredictor

reducers = joblib.load("checkpoint/feature_reducers.pkl")

spiral_model = NeuroSpiralPredictor(
    model_path="checkpoint/spiral_best_model.onnx",
    variance_selector=reducers["variance_selector"],
    scaler=reducers["scaler"],
    pca=reducers["pca"],
)
voice_model = NeuroVoxPredictor("checkpoint/voice_best_model.onnx")

img = cv2.imread("patient_spiral.png")
spiral_label, spiral_prob = spiral_model.predict(img)
voice_label,  voice_prob  = voice_model.predict("patient_voice.wav")

print(f"Spiral → {spiral_label} ({spiral_prob:.4f})")
print(f"Voice  → {voice_label}  ({voice_prob:.4f})")

# Simple majority vote
votes = [spiral_label, voice_label]
final = "PD" if votes.count("PD") >= 1 else "HC"
print(f"Combined → {final}")
```

---

## Execution Providers

Both predictors use the providers configured in `src/constant/constant.py`:

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

ONNX Runtime tries each provider in order and falls back automatically. To force CPU:

```python
import onnxruntime as ort
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

---

## Output Reference

### NeuroSpiralPredictor

| `probability` | `label` | Interpretation                                  |
| ------------- | ------- | ----------------------------------------------- |
| ≥ 0.5         | `HC`    | Model is confident subject is healthy           |
| < 0.5         | `PD`    | Model detects Parkinson's indicators in drawing |

### NeuroVoxPredictor

| `probability` | `label` | Interpretation                                |
| ------------- | ------- | --------------------------------------------- |
| > 0.5         | `PD`    | Model detects vocal biomarkers of Parkinson's |
| ≤ 0.5         | `HC`    | Model predicts healthy vocal pattern          |

> The polarity difference between the two models reflects their training label encoding. Both return the sigmoid of the raw logit; the threshold and direction differ by design.

---

## Error Handling

```python
# NeuroSpiralPredictor — check for decode failure upstream
img = cv2.imread("file.png")
if img is None:
    raise ValueError("Image could not be read")
label, prob = predictor.predict(img)

# NeuroVoxPredictor — returns ("Error", 0.0) on audio load failure
label, prob = predictor.predict("file.wav")
if label == "Error":
    print("Audio file could not be loaded — check format and path")
```

---

## Troubleshooting

| Problem                              | Likely Cause                               | Fix                                                                      |
| ------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------ |
| `RuntimeError: Failed to load model` | ONNX file missing or corrupt               | Check `checkpoint/` path; re-export model                                |
| `feature_reducers.pkl not found`     | Reducers not saved after training          | Re-run training and call `joblib.dump(...)` to save reducers             |
| `NeuroVox returns ("Error", 0.0)`    | Unreadable WAV (corrupt, non-PCM)          | Re-encode: `ffmpeg -i in.wav -ar 22050 -ac 1 out.wav`                    |
| `cv2.imdecode` returns `None`        | Image is corrupt or unsupported format     | Use `cv2.imread` with a valid path; check file integrity                 |
| Slow CPU inference                   | No GPU / large batch                       | Install `onnxruntime-gpu`; process in parallel with `ThreadPoolExecutor` |
| Wrong predictions on short audio     | Audio < 6 seconds gets zero-padded heavily | Ensure recordings are at least 3–4 seconds of actual speech              |
