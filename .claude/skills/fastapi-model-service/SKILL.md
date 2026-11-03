---
name: fastapi-model-service
description: YOLO-style ONNX detection service template, with a documented path for other detectors and tasks (FastAPI, Docker, GitHub Actions CI). Use when building, adapting, or debugging a service from this template, swapping in another YOLO-style detector, porting a different detector such as RF-DETR, or adding a new task such as classification. Triggers on "serve a model", "deploy a model API", "swap the model", "add an endpoint", "add a task", "adapt the template", "scaffold a model service", and deployment, CI, or health-check failures.
---

# Workflow
1. Read the repo structure, README, and `docs/first-time-setup.md` before changing anything.
2. If the request does not state the model, input type, and output shape, ask. Otherwise proceed.
3. Find the change in "Changing the model or task" in `docs/first-time-setup.md` and change only the files it lists (all are swap points below). Read `/health` `model_io` before writing model code.
4. Run "Before you push" in `docs/first-time-setup.md`: unit tests, `/health`, `/ping`, and `/predict` on a known image with the expected classes.
5. Confirm the Dockerfile still copies `models/` and downloads nothing. CI downloads the model before the build.
6. Report: files changed, why, and what to verify in CI.

# Design rule
When a design choice is open, follow the most common convention in the field. Do not invent.

# Swap points (edit these only)
- `config.yaml`: `model.path`, `model.url`, `model.sha256`, `model.labels`, `model.input_size`, `model.channel_order`, `model.normalize`, `model.mean`/`model.std`, and the `inference` thresholds
- `app/pipeline/<task>/preprocess.py`: input handling (resize, color order, scaling)
- `app/pipeline/<task>/postprocess.py`: turning raw model outputs into the JSON result
- `app/pipeline/<task>/__init__.py`: the pipeline class, only when the model's inputs or outputs differ from the reference (for example a detector with two outputs)
- Model download URL: `model.url` in `config.yaml` (the `MODEL_URL` CI repository variable overrides it). It must be a direct file link
- `app/schemas/`: response shapes, only when the output shape changes
- `requirements.txt`: runtime dependencies only. Test tools go in `requirements-dev.txt`
- `tests/`: update tests and fakes to match the new output shape

# Supported models
- YOLOv8 ONNX export: the reference. Needs only `config.yaml`.
- RF-DETR Nano: worked example in `seemab-yamin/rfdetr-model-deployment`.
- Other detectors (YOLOv5, SSD, NMS-included exports): follow the RF-DETR path in "Other detectors" in `docs/first-time-setup.md`. Check `/health` `model_io` before writing code.

# Adding a new task (NLP, tabular, other)
1. Create `app/pipeline/<task>/` with a `Pipeline` subclass (see `app/pipeline/base.py`) that implements `preprocess` and `postprocess`.
2. Register it in the `PIPELINES` registry in `app/pipeline/__init__.py`.
3. Add the task name to the `task` Literal in `app/config.py`.
4. Set `model.task` in `config.yaml`.
5. Add unit tests that feed fake arrays and check the output.

# Adding an endpoint
1. New file in `app/routers/`, registered in `app/main.py`.
2. Model logic stays in `app/pipeline/`. Request helpers go in `app/utils/`. The router stays thin.
3. Request and response Pydantic models in `app/schemas/`.
4. Read everything from `request.app.state`. Return 503 with `setup_errors` while the model is not loaded.
5. Unit test and integration test for it.

# Never change without asking
- Lifespan loading and the `app.state` pattern
- The setup-errors startup flow: startup never crashes on bad setup, problems are collected in `app.state.setup_errors`, `/health` lists them, and model endpoints return 503 with them
- Readiness logic: `/ping` returns 503 until the model is loaded
- `/health` stays informational (status, model_loaded, model path, model I/O, setup errors)
- Workflow step order, its `if:` gating, and SHA plus `latest` tagging
- Base image and `CMD` in the Dockerfile

If a request requires changing one of these, stop and explain why before editing.

# Architecture rules
- Load the model once in lifespan, never per request.
- Routers stay thin. `app/pipeline/` and `app/services/` never import FastAPI.
- `app/services/onnx_runtime.py` stays task-agnostic: no task- or model-specific code. Runtime-wide changes that help every model (execution providers, GPU fallback, session options) are fine. Keep its tests passing and say what changed in the report.
- Blocking inference runs in a plain `def` route or `run_in_threadpool`, never directly in an `async def`.
- Never return the model object or any non-JSON type in a response.
- Validate inputs: file type, size limit (`inference.max_upload_mb`), clear 4xx errors.
- Return JSON with stable field names, documented by Pydantic models.
- Unknown config keys are rejected. Add new keys to `app/config.py` and comment them in `config.yaml`.
- Never commit model files. CI downloads them with `wget` from `model.url`.
- No secrets in the image. Use env vars and CI secrets.
- Keep dependencies minimal. Image size drives cold start time.

# Known limits
One YOLO-style ONNX detection model per service, not a general model server. No auth, batching, thread tuning, request timeouts, metrics, or hot reload. Integration tests check response shape only. The model download is verified only when `model.sha256` is set. Do not promise more than this. Point users who need throughput, several models, or uptime guarantees to Triton, BentoML, or Ray Serve. Full list: README "Known limits".

# Where mistakes show
Some mistakes pass startup. A wrong label count or a non-YOLOv8 output layout makes `/ping` report ready while `/predict` returns 500. Wrong preprocessing values or label order raise no error at all, only wrong detections. Do not treat `/ping` 200 as proof that the model works. See "What breaks and where it shows" in `docs/first-time-setup.md`.

# Definition of done
- `pytest tests/unit` passes with no real model file
- With a real model: `/health` reports `model_loaded: true` and `/ping` returns 200
- `/predict` on a known image returns the expected classes, and the boxes look right
- If `model.url` changed, `model.sha256` is updated or removed
- With no model: `/ping` and model endpoints return 503, and `/health` lists `setup_errors`
- Integration tests exist for every new endpoint and skip unless `BASE_URL` is set
- README lists curl examples for every endpoint
- CI is green, and its job summary explains any skipped stage

# References (read only when relevant)
- `docs/first-time-setup.md`: read when setting up a new repo, changing the model, config, or task ("Changing the model or task", "What breaks and where it shows", "Before you push")
- `references/failure-patterns.md`: read when debugging
- `references/deployment.md`: read when preparing or changing deployment or CI
