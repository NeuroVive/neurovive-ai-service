# API Reference

## Overview

NeuroVive exposes **two prediction endpoints** under a single FastAPI service — one for image-based spiral/wave analysis and one for voice-based analysis. Both endpoints share the same response schema.

|                      |                                      |
| -------------------- | ------------------------------------ |
| **Base URL**         | `http://localhost:8000`              |
| **Interactive Docs** | `http://localhost:8000/docs`         |
| **ReDoc**            | `http://localhost:8000/redoc`        |
| **OpenAPI JSON**     | `http://localhost:8000/openapi.json` |

---

## Shared Response Schema

Both endpoints return the same Pydantic model:

```python
class PredictionResponse(BaseModel):
    label: str        # "HC" or "PD"
    probability: float
```

```json
{
  "label": "HC",
  "probability": 0.8734
}
```

**Interpretation:**

| `label` | `probability` range | Meaning                                        |
| ------- | ------------------- | ---------------------------------------------- |
| `HC`    | > 0.5               | Predicted Healthy Control                      |
| `PD`    | ≤ 0.5               | Predicted Parkinson's Disease _(spiral model)_ |
| `PD`    | > 0.5               | Predicted Parkinson's Disease _(voice model)_  |

> **Note on thresholds:** The spiral model (`NeuroSpiralPredictor`) labels `PD` when `probability < 0.5` and `HC` when `probability ≥ 0.5`. The voice model (`NeuroVoxPredictor`) is the inverse: `PD` when `probability > 0.5`. Both models use sigmoid output — the probability always reflects confidence in the `HC` class for spiral, and `PD` class for voice.

---

## `POST /predict/image`

Classify a hand-drawn spiral or wave image as HC or PD.

### Request

| Field   | Type               | Required | Description                                   |
| ------- | ------------------ | -------- | --------------------------------------------- |
| `image` | `file` (form-data) | ✅       | Any standard image format (JPEG, PNG, BMP, …) |

**Content-Type:** `multipart/form-data`

### Processing Steps

1. Validate `Content-Type` starts with `image/`
2. Decode bytes → NumPy BGR array via `cv2.imdecode`
3. Offload `NeuroSpiralPredictor.predict(img)` to `ThreadPoolExecutor`
4. Return `PredictionResponse`

### Status Codes

| Code                         | Condition                                       |
| ---------------------------- | ----------------------------------------------- |
| `200 OK`                     | Successful prediction                           |
| `415 Unsupported Media Type` | File is not an image (wrong MIME type)          |
| `400 Bad Request`            | Image is corrupt or cannot be decoded by OpenCV |
| `500 Internal Server Error`  | Inference failure                               |

### Examples

**curl:**

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "image=@spiral_drawing.png"
```

**Python:**

```python
import requests

with open("spiral_drawing.png", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict/image",
        files={"image": ("spiral_drawing.png", f, "image/png")},
    )

print(r.json())
# {'label': 'HC', 'probability': 0.8734}
```

**JavaScript:**

```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict/image", {
  method: "POST",
  body: formData,
});
console.log(await res.json());
// { label: "HC", probability: 0.8734 }
```

---

## `POST /predict/voice`

Classify a voice recording as HC or PD.

### Request

| Field   | Type               | Required | Description         |
| ------- | ------------------ | -------- | ------------------- |
| `audio` | `file` (form-data) | ✅       | A `.wav` audio file |

**Content-Type:** `multipart/form-data`

> Only `.wav` files are accepted. The extension is checked via `audio.filename.lower().endswith(".wav")`.

### Processing Steps

1. Validate filename ends with `.wav`
2. Stream file to `tmp/<uuid>.wav` in 1 MB chunks
3. Offload `NeuroVoxPredictor.predict(tmp_path)` to `ThreadPoolExecutor`
4. Register `remove_file(tmp_path)` as a `BackgroundTask` (runs after response is sent)
5. Return `PredictionResponse`

### File Cleanup

Temporary files are deleted **after** the HTTP response is delivered using FastAPI's `BackgroundTasks`. If inference fails, the temp file is deleted immediately before raising the exception.

### Status Codes

| Code                         | Condition                             |
| ---------------------------- | ------------------------------------- |
| `200 OK`                     | Successful prediction                 |
| `415 Unsupported Media Type` | File is not a `.wav`                  |
| `500 Internal Server Error`  | Audio processing or inference failure |

### Examples

**curl:**

```bash
curl -X POST http://localhost:8000/predict/voice \
  -F "audio=@voice_sample.wav"
```

**Python:**

```python
import requests

with open("voice_sample.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict/voice",
        files={"audio": ("voice_sample.wav", f, "audio/wav")},
    )

print(r.json())
# {'label': 'PD', 'probability': 0.3102}
```

---

## Error Response Format

All errors follow FastAPI's standard structure:

```json
{ "detail": "Human-readable error message" }
```

**Complete error reference:**

| Endpoint         | Code | `detail`                                                            |
| ---------------- | ---- | ------------------------------------------------------------------- |
| `/predict/image` | 415  | `"Uploaded file must be an image (JPEG, PNG, …)."`                  |
| `/predict/image` | 400  | `"Could not decode image. The file may be corrupt or unsupported."` |
| `/predict/image` | 500  | `"Inference error: <exception>"`                                    |
| `/predict/voice` | 415  | `"Only .wav audio files are supported."`                            |
| `/predict/voice` | 500  | `"Processing failed"` or `"Internal Server Error: <exception>"`     |

---

## Running the Server

```bash
# Standard
python -m src.api.main

# Custom port
uvicorn src.api.endpoint:app --host 0.0.0.0 --port 8080

# Development with auto-reload
uvicorn src.api.endpoint:app --reload
```

---

## Concurrency

Both endpoints offload blocking inference to a shared `ThreadPoolExecutor` with `max_workers=4`, keeping the async event loop free. Up to 4 simultaneous inference requests are processed in parallel — additional requests are queued.

```python
result = await loop.run_in_executor(app.state.executor, model.predict, input)
```

Both models (`app.state.voice_model` and `app.state.spiral_model`) are loaded once at startup and are thread-safe for concurrent reads.
