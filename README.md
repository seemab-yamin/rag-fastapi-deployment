# fastapi-model-service

YOLO-style ONNX detection service template, with a documented path for other detectors and tasks. It includes a FastAPI server, Docker image, GitHub Actions CI, tests, and an AI skill that guides code generation.

Out of the box, `config.yaml` points at a YOLOv8n object detector (`model.url`), which CI downloads and bakes into the image. Other YOLOv8-layout ONNX detectors need only `config.yaml` changes. Detectors with a different layout need the pipeline's pre- and postprocessing changed, and other tasks (classification, NLP, tabular) need a new pipeline. Both paths are documented in `docs/first-time-setup.md`.

## Supported and tested models

| Model | Task | Status | What it takes |
|---|---|---|---|
| YOLOv8 ONNX export (output `(1, 4+classes, N)`) | Detection | Reference, tested: unit tests, CI integration tests, local `/predict` | `config.yaml` only |
| RF-DETR Nano ONNX (outputs `pred_boxes`, `pred_logits`) | Detection | Worked example, tested: [rfdetr-model-deployment](https://github.com/seemab-yamin/rfdetr-model-deployment) (unit tests, CI integration tests, local `/predict`) `config.yaml` (including `mean`/`std`), `preprocess.py` (resize), `postprocess.py`, and the detection pipeline class. See "Other detectors: RF-DETR" in `docs/first-time-setup.md` |

Other layouts (YOLOv5 with an objectness column, SSD, exports with NMS built in) follow the same path as RF-DETR. Other tasks follow "Adding a new task".

## What you get

- FastAPI server that loads the model once at startup and never crashes on bad setup: problems are listed in `/health`
- Task pipeline registry (`app/pipeline/`) with a YOLOv8-style detection reference, on a task-agnostic ONNX runtime, so other detectors and tasks plug in without touching the runtime
- `/predict` endpoint with input validation and a typed response
- `/health` (info) and `/ping` (readiness) endpoints
- Dockerfile that bakes the model into the image at build time (CI downloads it into `models/` first)
- GitHub Actions: unit tests, model download, build, integration tests against the live container, push
- Images tagged with the commit SHA and `latest`
- Unit tests and integration tests (integration tests skip unless `BASE_URL` is set)
- A `SKILL.md` that tells an AI coding assistant how to adapt the template safely

## Quick start

1. Click **Use this template** on GitHub and name your new repo.
2. Clone it and install dependencies (`requirements-dev.txt` adds the test tools):

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

3. Download the model to `model.path` (models are not committed to git):

```bash
curl -fL -o models/yolov8n.onnx https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx
```

4. Run the server:

```bash
uvicorn app.main:app --reload
```

5. Open `http://localhost:8000/docs` and try `/predict` with an image.
6. Run the tests (no model file needed):

```bash
pytest tests/unit -v
```

Without a model file the server still starts: `/ping` and `/predict` return 503, and `/health` lists what to fix. See `docs/first-time-setup.md`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and info. Returns 200 when the process is up, with `status`, `model_loaded`, `model_path`, `model_io` (real input and output shapes, active and requested providers), and `setup_errors`. |
| GET | `/ping` | Readiness. Returns 503 until the model is loaded, then 200. Point platform health checks here. |
| POST | `/predict` | Runs the model on an uploaded image (`multipart/form-data`, field `file`) and returns JSON. |

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ping
```

### `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

Response:

```json
{
  "filename": "bus.jpg",
  "image": {"width": 810, "height": 1080},
  "count": 1,
  "detections": [
    {"class_id": 5, "class": "bus", "confidence": 0.84, "bbox": [30.9, 231.2, 800.7, 777.7]}
  ],
  "inference_ms": 24.2
}
```

`bbox` is `[x1, y1, x2, y2]` in original-image pixels. No detections returns `count: 0` and an empty list with status 200. `inference_ms` covers the model pipeline only.

| Status | Cause |
|---|---|
| 400 | Empty file, corrupt or truncated image, or a side under 16 px |
| 413 | File larger than `inference.max_upload_mb` (default 10) |
| 415 | Content type or real file format is not `image/jpeg`, `image/png`, or `image/webp` |
| 422 | No `file` field in the form |
| 500 | Pipeline error (message says what to fix, for example a labels count mismatch) or an internal error (generic message, traceback in the server log) |
| 503 | Model not loaded. The body has `setup_errors` listing what to fix |

## Project structure

```
app/
  main.py          # lifespan, app.state, setup_errors, router registration
  config.py        # loads and validates config.yaml (unknown keys rejected)
  routers/         # thin HTTP layer (health.py, predict.py)
  pipeline/        # task registry: base.py, detection/ (preprocess.py, postprocess.py)
  services/        # onnx_runtime.py: task-agnostic ONNX model wrapper
  schemas/         # request and response models
  utils/           # upload and image validation helpers
config.yaml        # model and inference settings
models/            # .onnx goes here (git keeps only .gitkeep)
optional/annotate/ # opt-in /annotate route (not in the image until copied into app/)
tests/
  unit/            # run on every push, no model needed
  integration/     # run against a live container, skip unless BASE_URL is set
docs/first-time-setup.md
Dockerfile
.dockerignore
requirements.txt       # runtime dependencies
requirements-dev.txt   # test tools
.github/workflows/ci.yml
.claude/skills/fastapi-model-service/
  SKILL.md
  references/
```

## Configuration

Settings live in `config.yaml`. Each key is commented there, and unknown keys are rejected so typos fail loudly.

| Key | Default in template | Meaning |
|---|---|---|
| `model.path` | `models/yolov8n.onnx` | Where the app loads the model. Must be under `models/` for the image |
| `model.url` | YOLOv8n on Hugging Face | Direct link CI downloads to `model.path`. The app never downloads it |
| `model.sha256` | SHA-256 of that YOLOv8n file | Optional. CI fails the download if the file differs. Update it together with `model.url` |
| `model.task` | `detection` | Pipeline to use from `app/pipeline/` |
| `model.input_size` | `[640, 640]` | Height, width. Used only if the model input is dynamic |
| `model.channel_order` | `RGB` | `RGB` or `BGR` |
| `model.normalize` | `true` | Divide pixel values by 255 |
| `model.mean`, `model.std` | unset | Optional per-channel mean and std, applied after `normalize` (for example ImageNet values for RF-DETR) |
| `model.labels` | 80 COCO names | Class names in model order |
| `inference.providers` | `[CPUExecutionProvider]` | ONNX Runtime execution providers, in order. Ones this build lacks are skipped with a warning, and CPU is always the fallback |
| `inference.confidence_threshold` | `0.25` | Minimum score to keep a detection |
| `inference.iou_threshold` | `0.45` | NMS overlap threshold |
| `inference.max_upload_mb` | `10` | `/predict` rejects larger uploads with 413 |

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port the server listens on in the container |
| `CONFIG_PATH` | `config.yaml` | Path to the config file |

Secrets (API keys, registry tokens) come from env vars or CI secrets. Never commit them and never bake them into the image.

## Swap in a different model

Change only these:

- `config.yaml`: `model.path`, `model.url`, `model.labels`, `model.input_size`, `model.channel_order`, `model.normalize`, thresholds
- `app/pipeline/<task>/preprocess.py`: if the model's input handling differs (resize, color order, scaling)
- `app/pipeline/<task>/postprocess.py`: if the output layout differs from YOLOv8 (`(1, 4+classes, N)`)
- `app/schemas/`: only if the response shape changes
- `tests/`: update tests and fakes to match

Check `/health` `model_io` to see the model's real input and output shapes. For a new task type (NLP, tabular), see "Adding a new task" in `docs/first-time-setup.md`.

Before any change, find it in "Changing the model or task" in `docs/first-time-setup.md`, and afterwards run its "Before you push" checklist. Some mistakes pass startup, and some raise no error at all. "What breaks and where it shows" lists them.

Do not change the lifespan loading, the setup-errors flow, the readiness logic, or the workflow step order unless you have a clear reason. `app/services/onnx_runtime.py` stays task-agnostic: model-specific code goes in the pipeline, while runtime-wide changes (execution providers, GPU fallback) are fine.

## Optional features

`optional/` holds opt-in code that is not part of the core service or the Docker image:

- `optional/annotate/`: a `/annotate` route that streams the uploaded image back with boxes drawn on it. See its README to enable it.

## Use the AI skill

The skill lives in `.claude/skills/fastapi-model-service/`. Open the repo root in Claude Code and it loads automatically when your request matches its description.

Example prompts:

- "Adapt this template to serve a ViT image classifier. Return the top 5 labels with scores."
- "Enable the optional `/annotate` endpoint from `optional/annotate/`."
- "The container keeps restarting after deploy. Help me debug it."

After the AI finishes, review the diff. It should touch only the swap points. If it edits the lifespan code or the workflow, reject the change and tighten the wording in `SKILL.md`.

When you hit a new class of failure, add a general pattern to `references/failure-patterns.md`.

## CI/CD

The workflow (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

1. Installs dependencies and runs unit tests (always)
2. Reads `model.url`, `model.sha256` and `model.path` from `config.yaml`, downloads the model with `wget`, verifies its checksum, and builds the Docker image
3. Starts the container, waits for `/ping`, and runs integration tests against it
4. Pushes the image to Docker Hub, tagged with the commit SHA and `latest` (push to `main` only, never on pull requests)

The model is set in `config.yaml`:

```yaml
model:
  path: models/yolov8n.onnx   # CI downloads to this path. Must be under models/
  url: https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx
  sha256: 65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5   # optional, pins the file
```

`model.url` must be a direct file link (a GitHub release asset or a Hugging Face `resolve/main/...` URL, not a web page). Models are never committed to git.

Registry settings go under **Settings > Secrets and variables > Actions**:

| Name | Kind | Value |
|---|---|---|
| `DOCKERHUB_USERNAME` | secret | Your registry username |
| `DOCKERHUB_TOKEN` | secret | A registry access token |
| `MODEL_URL` | repository variable (optional) | Overrides `model.url`, for example to test another model without a commit |
| `MODEL_SHA256` | repository variable (optional) | Checksum for the `MODEL_URL` file. Used only together with `MODEL_URL` |

| Configured | What runs |
|---|---|
| No `model.url` | Unit tests only. The run is green and the job summary says what to set. |
| `model.url`, no secrets | Unit tests, download, build, integration tests. Push is skipped. |
| `model.url` and both secrets, push to `main` | Everything, and the image is pushed as `<username>/<repo>:<sha>` and `:latest`. |

The job summary lists each stage's result, where the model came from, and why any stage was skipped.

The Docker Hub image name is `<DOCKERHUB_USERNAME>/<GitHub repo name>`, lowercased. Docker Hub creates the repository on the first push (public unless your account default is private). The token needs Read & Write access.

CI stops once the image is pushed. Deploying to a platform is a per-project step.

## Deploy

Any platform that runs a container works. See `.claude/skills/fastapi-model-service/references/deployment.md` for the requirements.

In short:

- Deploy the pinned SHA tag, not `latest`
- Point the platform health check at `/ping`
- Set required env vars on the platform
- Roll back by redeploying an earlier SHA tag
- Record platform quirks in your project's README, not in the template

## Known limits

This is a template for serving one YOLO-style ONNX detection model. It is
not a general model server.

- **Detectors:** the reference pipeline targets YOLOv8-style output
  `(1, 4+classes, N)`. Other detectors (RF-DETR, YOLOv5, SSD, DETR, models
  with built-in NMS) need edits to `postprocess.py` and sometimes
  `preprocess.py`, `config.yaml`, and the pipeline class (for models with
  several outputs). RF-DETR is the worked example.
- **Other tasks:** classification, NLP, or tabular models need a new pipeline
  class, a router, and schemas. See "Adding a new task" in
  `docs/first-time-setup.md`.
- **One model per service.** No multi-model serving or hot reload.
- **No authentication.** Add it before exposing a service publicly.
- **No throughput features:** no batching, no ONNX Runtime thread tuning,
  no request timeouts. Concurrent requests each run inference in their own
  thread.
- **No observability:** no metrics, request IDs, or structured logs.
- **Integration tests check the response shape only.** They do not check
  that detections are correct, because the template cannot know the model.
- **Model download:** set `model.sha256` (or `MODEL_SHA256` with
  `MODEL_URL`) to verify the file. Without it, a changed upstream file is not
  detected.

For high throughput, several models, or uptime guarantees, use a dedicated
serving tool such as Triton, BentoML, or Ray Serve.

## License

MIT. See `LICENSE`. The default model is Ultralytics YOLOv8n, which Ultralytics releases under AGPL-3.0. Check the model card and the license of any model you swap in.