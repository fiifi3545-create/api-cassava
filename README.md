# Cassava Guard API

Disease classification for cassava leaf photos, serving the **CustomCNN** trained in
`nn-final-project (3).ipynb`. Backend for the
[cassava_guard](https://github.com/KelpyShades/cassava_guard) Flutter app.

Inference runs on ONNX Runtime, not PyTorch — the int8-quantized graph is 26 MB and the
service holds ~125 MB of RAM, so it fits a free-tier instance.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/analyze-cassava` | multipart field `file` = leaf image → prediction |
| `POST` | `/analyze-pest` | legacy alias of the above |
| `GET` | `/health` | service + model status |
| `GET` | `/classes` | the five classes the model predicts |
| `GET` | `/docs` | Swagger UI — upload an image and see the result |

Response:

```json
{
  "disease_class": "Cassava mosaic disease",
  "analysis": "... Model confidence: 91.4%.",
  "suggestions": "Remove and destroy infected plants ...",
  "confidence": 0.9138,
  "probabilities": { "Cassava bacterial blight": 0.021, "...": 0.0 },
  "pest": "Cassava mosaic disease"
}
```

`pest` duplicates `disease_class` for older app builds.

## Classes

Index order is the Kaggle `label_num_to_disease_map.json` used by the training notebook —
it is the meaning of the model's five output neurons, so **do not reorder it**:

| Index | Label |
| --- | --- |
| 0 | Cassava bacterial blight |
| 1 | Cassava brown streak disease |
| 2 | Cassava green mottle |
| 3 | Cassava mosaic disease |
| 4 | Healthy |

The network only knows these five, so it will still answer confidently on a photo of
something else. Predictions below `CASSAVA_MIN_CONFIDENCE` are reported as
`Unclear / not a cassava leaf`.

## Run locally

```bash
pip install -r requirements_api.txt
uvicorn main:app --reload --port 8000
curl -X POST -F "file=@leaf.jpg" http://127.0.0.1:8000/analyze-cassava
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CASSAVA_MODEL_PATH` | `./cassava_customcnn.int8.onnx` | graph to load |
| `CASSAVA_MODEL_URL` | — | download the graph at boot if the file is missing |
| `CASSAVA_MIN_CONFIDENCE` | `0.40` | below this → `Unclear / not a cassava leaf` |
| `CASSAVA_USE_FP32` | — | `1` loads the full-precision graph instead |
| `CASSAVA_ORT_THREADS` | `1` | ONNX Runtime intra-op threads |

## Regenerating the model

`best_CustomCNN.pth` (104 MB) is **not** in the repo — it is over GitHub's file limit, and
only the ONNX graph is needed to serve. To rebuild from the checkpoint:

```bash
pip install torch onnx onnxruntime onnxscript
python export_onnx.py
```

That writes `cassava_customcnn.onnx` (fp32, 104 MB, gitignored) and
`cassava_customcnn.int8.onnx` (26 MB, committed), then checks both against PyTorch —
fp32 matches exactly, int8 agrees on every argmax with ~0.04 max logit drift.

## Deploy

`render.yaml` deploys as a Python service on Render's free plan. `Dockerfile` is
equivalent if you prefer a container. The health check is `/health`, which reports
whether the model actually loaded.
