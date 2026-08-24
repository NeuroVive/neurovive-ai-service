# NeuroVox — Voice-Based PD Detection

## Overview

**NeuroVox** classifies `.wav` voice recordings as either Healthy Control (`HC`) or Parkinson's Disease (`PD`). It converts raw audio into **Mel-Spectrogram** images and processes them with a fine-tuned **ResNet-18** backbone adapted for single-channel spectrogram input.

---

## Why Mel-Spectrograms + ResNet?

| Choice                        | Rationale                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Mel-Spectrogram**           | Converts 1D audio into a 2D time-frequency image; perceptually weighted frequency axis emphasizes clinically relevant vocal bands                                  |
| **ResNet-18**                 | Pretrained ImageNet weights encode general edge/texture patterns that transfer well to spectrograms; residual connections ensure stable training on small datasets |
| **Single-channel adaptation** | Spectrograms are grayscale; the first conv layer is modified from `3→64` to `1→64` to accept mono input without information duplication                            |

---

## Model Architecture

```
Input: .wav file
        │
   librosa.load → resample to 22050 Hz, mono
        │
   Normalize to 6 seconds (pad or truncate)
        │
   Mel-Spectrogram: (40 mels × ~517 frames)
        │
   ┌────▼──────────────────────────────────────┐
   │  ResNet-18 Backbone                       │
   │  conv1: Conv2d(1→64, kernel=(3,7),        │
   │          stride=(1,2))  ← 1-ch adapted    │
   │  layers 1–4 (residual blocks)             │
   │  AdaptiveAvgPool2d → 512-d embedding      │
   └────┬──────────────────────────────────────┘
        │
   Custom Classifier Head
   Linear(512 → 256) → BatchNorm1d → SiLU → Dropout(0.5) → Linear(256 → 1)
        │
   Raw Logit → sigmoid → label
```

---

## `NeuroVoxPredictor` — Inference Class

**File:** `src/inference/neurovox_predictor.py`

### Constructor

```python
predictor = NeuroVoxPredictor(model_path="checkpoint/voice_best_model.onnx")
```

Initializes an `onnxruntime.InferenceSession`, caches input/output node names, and precomputes `target_length = duration × sample_rate = 132,300` samples.

Raises `RuntimeError` if the ONNX file cannot be loaded.

### `predict(audio_path)`

```python
label, probability = predictor.predict("recording.wav")
```

| Parameter    | Type  | Description                                                    |
| ------------ | ----- | -------------------------------------------------------------- |
| `audio_path` | `str` | Path to a `.wav` file (any sample rate — resampled internally) |

**Returns:** `Tuple[str, float]`

- `label` → `"PD"` if `probability > 0.5`, `"HC"` if `probability ≤ 0.5`
- `probability` → sigmoid of raw logit, rounded to 4 decimal places
- Returns `("Error", 0.0)` if audio loading fails

---

## Preprocessing Pipeline (Inside `predict`)

### Step 1 — Load & Resample

```python
audio, _ = librosa.load(audio_path, sr=22050, mono=True)
# Any input sample rate is resampled to 22,050 Hz; stereo is mixed to mono
```

### Step 2 — Normalize Length

```python
target_length = 6 × 22050 = 132,300 samples

if len(audio) < target_length:
    audio = librosa.util.fix_length(audio, size=target_length)  # zero-pad
else:
    audio = audio[:target_length]                               # truncate
```

### Step 3 — Mel-Spectrogram

```python
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=22050,
    n_fft=1024,
    hop_length=256,
    n_mels=40,
)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
# Output shape: (40, ~517)
```

### Step 4 — Format for ONNX

```python
spec_tensor = mel_db[np.newaxis, np.newaxis, :, :]  # (1, 1, 40, 517)
spec_tensor = spec_tensor.astype(np.float32)
```

### Step 5 — Inference & Sigmoid

```python
logits = session.run([output_name], {input_name: spec_tensor})[0]
probability = 1 / (1 + np.exp(-logits.item()))
label = "PD" if probability > 0.5 else "HC"
```

---

## Audio Parameters

| Parameter         | Value           | Description                           |
| ----------------- | --------------- | ------------------------------------- |
| `sample_rate`     | 22,050 Hz       | Target resampling rate                |
| `duration`        | 6 seconds       | Fixed input length                    |
| `target_length`   | 132,300 samples | `duration × sample_rate`              |
| `n_fft`           | 1,024           | FFT window size                       |
| `hop_length`      | 256             | `n_fft // 4` — frames per second ≈ 86 |
| `n_mel`           | 40              | Number of Mel filter banks            |
| Spectrogram shape | `(40, ~517)`    | Mel bands × time frames               |

---

## ONNX Model Details

| Property     | Value                                |
| ------------ | ------------------------------------ |
| Input name   | `"input"`                            |
| Input shape  | `(batch_size, 1, 40, T)` — dynamic T |
| Output name  | `"outputs"`                          |
| Output shape | `(batch_size, 1)` — raw logit        |
| Dynamic axes | batch dimension for input and output |
| Opset        | 11                                   |

---

## Usage Example

```python
from src.inference.neurovox_predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/voice_best_model.onnx")

label, probability = predictor.predict("patient_voice.wav")

print(f"Label      : {label}")        # HC or PD
print(f"Probability: {probability}")  # e.g. 0.3102 → PD
```

---

## Performance

| Metric                   | Value            |
| ------------------------ | ---------------- |
| Best Validation Accuracy | 97.80% (Epoch 6) |
| Best Validation F1       | 97.37% (Epoch 6) |
| Final Training Accuracy  | 100.00%          |
| Training Epochs          | 10               |
| Valid Audio Files        | 1,274            |
