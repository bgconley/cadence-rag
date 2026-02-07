# P620 Runbook: Triton + ONNX + Qwen3-Embedding-4B (End-to-End)

This guide is a full, copy/paste walkthrough for standing up the exact embedding stack now running on the Lenovo P620:

- `Triton Inference Server` in Docker (`nvcr.io/nvidia/tritonserver:26.01-py3`)
- `Qwen3-Embedding-4B` ONNX model served by Triton
- `FastAPI embedding gateway` that performs tokenization, last-token pooling, L2 normalization, and enforces a `1024` output vector contract for this RAG project

This is written for Ubuntu `24.04.3` with NVIDIA GPU and minimal prior experience.

## 1. Final Architecture
> Why this step exists: You need a mental model of request flow before touching infrastructure.
> What this enables: Faster debugging because you can localize failures to client, gateway, Triton, or model.

```text
Client / RAG app
   |
   | POST /embed
   v
Embed Gateway (FastAPI, port 8101)
   |  - tokenizes text
   |  - calls Triton model
   |  - last-token pooling
   |  - truncates 2560 -> 1024
   |  - L2 normalize
   v
Triton Server (ports 8200/8201/8202)
   |
   | loads ONNX model from model_repository
   v
Qwen3-Embedding-4B ONNX
```

Project contract alignment:
- Embedding dimensionality exposed by gateway: `1024`
- Internal model hidden size: `2560` (from Qwen3-Embedding-4B)

## 2. Preflight Checklist
> Why this step exists: Most setup failures are missing prerequisites, not bad model code.
> What this enables: You eliminate environment blockers early (permissions, disk, GPU, network, credentials).

Before starting, confirm:

1. You are SSH'ed into the P620 Ubuntu host.
2. You have `sudo` privileges.
3. You have internet access.
4. You have an NGC API key (for pulling `nvcr.io` images).
5. You have at least ~20 GB free disk (model + containers + cache).

Check disk and GPU quickly:

```bash
df -h /
nvidia-smi
```

Expected GPU output should show your NVIDIA card (for example RTX 3090).

## 3. Set Common Variables (Do This First)
> Why this step exists: Hard-coded paths create subtle copy/paste mistakes.
> What this enables: Reusable commands and consistent file locations across the full runbook.

Use these variables in all commands below:

```bash
export HOST_USER="$USER"
export BASE="/home/$HOST_USER/nvidia"
export MODEL_SRC_DIR="$BASE/models/onnx/qwen3-embedding-4b-zhiqing"
export TRITON_REPO="$BASE/triton/model_repository"
export GATEWAY_DIR="$BASE/gateways/embed"

mkdir -p "$BASE"
```

Verify:

```bash
echo "$HOST_USER"
echo "$BASE"
```

## 4. Install Host Dependencies (Ubuntu 24.04.3)
> Why this step exists: Triton and GPU serving depend on host-level plumbing (driver, Docker, NVIDIA runtime).
> What this enables: Containers can run with GPU access and required tooling is present for verification.

### 4.1 System Packages
> Why: These tools are the minimum foundation for package installs, JSON inspection, model download, and git-lfs artifacts.
> What for: Ensures every later command in this runbook can execute successfully.

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  jq \
  git \
  git-lfs

git lfs install
```

### 4.2 NVIDIA Driver Check
> Why: The host driver is the dependency that exposes CUDA capability to containers.
> What for: Confirms the GPU is usable before you invest time in Triton setup.

If `nvidia-smi` already works, skip this subsection.

If it does not:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, SSH back in and verify:

```bash
nvidia-smi
```

### 4.3 Docker Engine Install (Official Docker Repo)
> Why: Triton and the gateway are containerized, so container runtime is mandatory.
> What for: Provides build/run capabilities and reproducible service lifecycle management.

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and log back in (or reboot) so group changes apply.

Verify Docker:

```bash
docker --version
docker run --rm hello-world
```

### 4.4 NVIDIA Container Toolkit (Docker GPU Access)
> Why: Standard Docker cannot use NVIDIA GPUs without the NVIDIA runtime integration.
> What for: Enables `--gpus all` so Triton can execute inference on the GPU.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, do not continue until this works.

## 5. NGC Login and Pull Triton Image
> Why this step exists: Triton images live in NGC, and tag-only pulls are not fully reproducible.
> What this enables: Authenticated pulls from `nvcr.io` and immutable digest pinning for repeatable deployments.

### 5.1 Login to `nvcr.io`
> Why: NVIDIA Triton production images are hosted in NGC private registry.
> What for: Grants pull access to required Triton images.

Use your NGC API key.

```bash
docker login nvcr.io
```

When prompted:
- Username: `$oauthtoken`
- Password: `<YOUR_NGC_API_KEY>`

### 5.2 Choose and Pull Image Tags
> Why: You must select a known release family and download it locally.
> What for: Ensures you run a specific, testable Triton version.

This runbook uses:
- Triton: `nvcr.io/nvidia/tritonserver:26.01-py3`

Pull it:

```bash
docker pull nvcr.io/nvidia/tritonserver:26.01-py3
```

### 5.3 Record Image Digests (Pinning)
> Why: Tags can move over time; digests do not.
> What for: Locks your environment for reproducibility and future rollback.

```bash
mkdir -p "$BASE"
{
  echo "# NVIDIA image digests"
  echo "# generated $(date -Is)"
  echo
  echo "TRITON_IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' nvcr.io/nvidia/tritonserver:26.01-py3)"
} > "$BASE/VERSIONS.local.md"

cat "$BASE/VERSIONS.local.md"
```

Notes:
- Tag is human-friendly (`26.01-py3`).
- Digest is immutable pin (`sha256:...`).

## 6. Create Directory Layout
> Why this step exists: Triton requires a strict model-repository structure.
> What this enables: Predictable paths for model assets, server repository, and gateway source.

```bash
mkdir -p "$BASE/models/onnx"
mkdir -p "$TRITON_REPO"
mkdir -p "$GATEWAY_DIR"
```

Expected layout:

```text
/home/<user>/nvidia/
  models/onnx/
  triton/model_repository/
  gateways/embed/
```

## 7. Download Qwen3-Embedding-4B ONNX Files
> Why this step exists: Triton cannot serve the model until ONNX weights and tokenizer files are local.
> What this enables: A complete local model package the gateway and Triton can both consume.

This guide uses a pre-exported ONNX repository:
- `https://huggingface.co/zhiqing/Qwen3-Embedding-4B-ONNX`

Clone with Git LFS:

```bash
cd "$BASE/models/onnx"
git clone https://huggingface.co/zhiqing/Qwen3-Embedding-4B-ONNX qwen3-embedding-4b-zhiqing
```

Verify key files:

```bash
ls -lah "$MODEL_SRC_DIR"
```

You must see at least:
- `model.onnx`
- `model.onnx_data`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

If files are owned by `root` (can happen if downloaded inside root-run container), fix ownership:

```bash
sudo chown -R "$HOST_USER":"$HOST_USER" "$MODEL_SRC_DIR"
```

## 8. Create Triton Model Repository Entry
> Why this step exists: Triton discovers models only through repository folders plus `config.pbtxt`.
> What this enables: Triton can load the model with correct input/output names, datatypes, batching, and instance policy.

### 8.1 Copy Model Files Into Triton Repo
> Why: Triton only loads models that exist in its mounted repository hierarchy.
> What for: Makes the ONNX graph and external weights discoverable to Triton.

```bash
mkdir -p "$TRITON_REPO/qwen3_embed_4b_onnx/1"
cp -f "$MODEL_SRC_DIR/model.onnx" "$TRITON_REPO/qwen3_embed_4b_onnx/1/model.onnx"
cp -f "$MODEL_SRC_DIR/model.onnx_data" "$TRITON_REPO/qwen3_embed_4b_onnx/1/model.onnx_data"
```

### 8.2 Create `config.pbtxt`
> Why: Triton needs explicit model interface and scheduling metadata.
> What for: Defines how clients call the model and how Triton batches/places execution.

```bash
cat > "$TRITON_REPO/qwen3_embed_4b_onnx/config.pbtxt" <<'PBTXT'
name: "qwen3_embed_4b_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "position_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP16
    dims: [ -1, 2560 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
PBTXT
```

Important:
- Keep output datatype as `TYPE_FP16` to match the ONNX model.
- Do not set this output to `TYPE_FP32` unless the model actually outputs FP32.

## 9. Start Triton Container
> Why this step exists: This is the serving control plane for ONNX inference.
> What this enables: Live HTTP/gRPC/metrics endpoints and GPU-backed inference execution.

If a previous `triton` container exists, remove it first:

```bash
docker rm -f triton 2>/dev/null || true
```

Run Triton:

```bash
docker run -d \
  --name triton \
  --gpus all \
  --restart unless-stopped \
  --shm-size=1g \
  -p 8200:8000 \
  -p 8201:8001 \
  -p 8202:8002 \
  -v "$TRITON_REPO":/models \
  nvcr.io/nvidia/tritonserver:26.01-py3 \
  tritonserver --model-repository=/models
```

## 10. Validate Triton and Model Load
> Why this step exists: A running container does not guarantee the model loaded correctly.
> What this enables: Early detection of dtype mismatches, missing files, and bad model metadata before app integration.

### 10.1 Health Checks
> Why: Live and ready probes validate server process state and serving readiness.
> What for: Separates startup failures from request-path/model failures.

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8200/v2/health/live
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8200/v2/health/ready
```

Both should return `200`.

### 10.2 Model Index
> Why: Server readiness alone does not imply model readiness.
> What for: Confirms the specific model is loaded and in `READY` state.

```bash
curl -s -X POST http://localhost:8200/v2/repository/index | jq
```

Expected state includes:
- `qwen3_embed_4b_onnx` with `"state": "READY"`

### 10.3 Model Metadata
> Why: Client code must match exact input/output names, types, and shapes.
> What for: Prevents mismatched request payloads and runtime infer errors.

```bash
curl -s http://localhost:8200/v2/models/qwen3_embed_4b_onnx | jq
```

Expected highlights:
- inputs: `input_ids`, `attention_mask`, `position_ids`
- output: `last_hidden_state`
- output shape ending in `2560`

If not ready, inspect logs:

```bash
docker logs -n 300 triton
```

## 11. Build Embedding Gateway (FastAPI)
> Why this step exists: Raw model output is token-level hidden states, not app-ready embeddings.
> What this enables: Tokenization plus pooling plus normalization plus dimension enforcement to satisfy your `1024` embedding contract.

The gateway exists because:
1. Triton serves token-level hidden states, not directly pooled sentence vectors.
2. We need project-compatible `1024` vectors.
3. We need stable request/response API for app integration.

### 11.1 Create Gateway Files
> Why: This service adapts raw model output into your app contract.
> What for: Produces normalized, pooled, 1024-d vectors from Qwen3 4B hidden states.

#### `Dockerfile`

```bash
cat > "$GATEWAY_DIR/Dockerfile" <<'DOCKERFILE'
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/app.py
EXPOSE 8101
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8101"]
DOCKERFILE
```

#### `requirements.txt`

```bash
cat > "$GATEWAY_DIR/requirements.txt" <<'REQ'
fastapi==0.116.1
uvicorn[standard]==0.35.0
numpy==2.3.2
requests==2.32.5
transformers==4.55.4
tritonclient[http]==2.65.0
REQ
```

#### `app.py`

```bash
cat > "$GATEWAY_DIR/app.py" <<'PY'
from __future__ import annotations

import os
from urllib.parse import urlparse
from typing import Dict, List, Optional

import numpy as np
import requests
import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8200").rstrip("/")
TRITON_MODEL = os.getenv("TRITON_MODEL", "qwen3_embed_onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/models/qwen3-embedding-0.6b-community")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "1024"))
EMBED_OUTPUT_DIM = int(os.getenv("EMBED_OUTPUT_DIM", "1024"))
TIMEOUT_S = float(os.getenv("EMBED_TIMEOUT_S", "120"))


class EmbedRequest(BaseModel):
    texts: List[str] = Field(default_factory=list)
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str


app = FastAPI(title="Qwen3 Embedding Gateway")

tokenizer = None
TRITON_CLIENT = None
MODEL_INPUTS: Dict[str, Dict] = {}
MODEL_OUTPUT_NAME = "last_hidden_state"
MODEL_HIDDEN_DIM = 0
REQUIRES_PAST_KV = False
PAST_INPUT_NAMES: List[str] = []


def _normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


def _build_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    pos = np.cumsum(attention_mask, axis=1) - 1
    pos = np.where(attention_mask > 0, pos, 0)
    return pos.astype(np.int64)


def _triton_client_url(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    return parsed.netloc or parsed.path


def _numpy_dtype(datatype: str) -> np.dtype:
    mapping = {
        "BOOL": np.bool_,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
    }
    if datatype not in mapping:
        raise RuntimeError(f"Unsupported Triton datatype: {datatype}")
    return mapping[datatype]


def _parse_model_signature() -> None:
    global MODEL_INPUTS
    global MODEL_OUTPUT_NAME
    global MODEL_HIDDEN_DIM
    global REQUIRES_PAST_KV
    global PAST_INPUT_NAMES

    r = requests.get(f"{TRITON_URL}/v2/models/{TRITON_MODEL}", timeout=TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch model metadata for {TRITON_MODEL}: {r.status_code} {r.text[:400]}"
        )

    meta = r.json()
    inputs = meta.get("inputs", [])
    outputs = meta.get("outputs", [])

    MODEL_INPUTS = {i.get("name"): i for i in inputs if i.get("name")}
    PAST_INPUT_NAMES = sorted([name for name in MODEL_INPUTS if name.startswith("past_key_values.")])
    REQUIRES_PAST_KV = len(PAST_INPUT_NAMES) > 0

    if not outputs:
        raise RuntimeError(f"Model {TRITON_MODEL} has no outputs")

    output_names = [o.get("name") for o in outputs if o.get("name")]
    MODEL_OUTPUT_NAME = "last_hidden_state" if "last_hidden_state" in output_names else output_names[0]

    chosen_output = next((o for o in outputs if o.get("name") == MODEL_OUTPUT_NAME), None)
    if chosen_output is None:
        raise RuntimeError(f"Could not find output metadata for {MODEL_OUTPUT_NAME}")

    shape = chosen_output.get("shape") or []
    if not shape:
        raise RuntimeError(f"Output {MODEL_OUTPUT_NAME} has empty shape metadata")
    hidden_dim = int(shape[-1])
    if hidden_dim <= 0:
        raise RuntimeError(f"Output {MODEL_OUTPUT_NAME} hidden dim is not static: {shape}")
    MODEL_HIDDEN_DIM = hidden_dim

    if EMBED_OUTPUT_DIM > MODEL_HIDDEN_DIM:
        raise RuntimeError(
            f"Configured EMBED_OUTPUT_DIM={EMBED_OUTPUT_DIM} exceeds model hidden dim {MODEL_HIDDEN_DIM}"
        )


def _shape_for_empty_past(shape: List[int], batch: int) -> List[int]:
    resolved: List[int] = []
    used_batch = False
    used_zero = False
    for d in shape:
        if d == -1 and not used_batch:
            resolved.append(batch)
            used_batch = True
        elif d == -1 and not used_zero:
            resolved.append(0)
            used_zero = True
        elif d == -1:
            resolved.append(1)
        else:
            resolved.append(int(d))
    if not used_zero and len(resolved) >= 3:
        resolved[-2] = 0
    return resolved


def _to_infer_input(name: str, arr: np.ndarray, datatype: str) -> httpclient.InferInput:
    infer_input = httpclient.InferInput(name, list(arr.shape), datatype)
    infer_input.set_data_from_numpy(arr)
    return infer_input


def _triton_infer(input_ids: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
    batch = int(input_ids.shape[0])

    triton_inputs: List[httpclient.InferInput] = [
        _to_infer_input("input_ids", input_ids, "INT64"),
        _to_infer_input("attention_mask", attention_mask, "INT64"),
    ]

    if "position_ids" in MODEL_INPUTS:
        triton_inputs.append(_to_infer_input("position_ids", position_ids, "INT64"))

    if REQUIRES_PAST_KV:
        for name in PAST_INPUT_NAMES:
            spec = MODEL_INPUTS[name]
            dtype = spec.get("datatype", "FP32")
            shape = _shape_for_empty_past(spec.get("shape", []), batch)
            empty = np.empty(shape, dtype=_numpy_dtype(dtype))
            triton_inputs.append(_to_infer_input(name, empty, dtype))

    try:
        result = TRITON_CLIENT.infer(
            model_name=TRITON_MODEL,
            inputs=triton_inputs,
            outputs=[httpclient.InferRequestedOutput(MODEL_OUTPUT_NAME, binary_data=True)],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triton infer failed: {exc}") from exc

    out = result.as_numpy(MODEL_OUTPUT_NAME)
    if out is None:
        raise HTTPException(status_code=502, detail=f"Triton output missing {MODEL_OUTPUT_NAME}")
    return out


@app.on_event("startup")
def _startup() -> None:
    global tokenizer
    global TRITON_CLIENT

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    TRITON_CLIENT = httpclient.InferenceServerClient(
        url=_triton_client_url(TRITON_URL),
        verbose=False,
    )
    _parse_model_signature()


@app.get("/health")
def health() -> dict:
    r = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=10)
    return {
        "status": "ok" if r.status_code == 200 else "degraded",
        "triton_status": r.status_code,
        "triton_model": TRITON_MODEL,
        "tokenizer_path": TOKENIZER_PATH,
        "embed_output_dim": EMBED_OUTPUT_DIM,
        "model_hidden_dim": MODEL_HIDDEN_DIM,
        "requires_past_kv": REQUIRES_PAST_KV,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    texts = [t for t in req.texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="texts must contain at least one non-empty string")

    encoded = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=EMBED_MAX_LENGTH,
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    position_ids = _build_position_ids(attention_mask)

    hidden = _triton_infer(input_ids, attention_mask, position_ids)

    last_token_idx = attention_mask.sum(axis=1) - 1
    last_token_idx = np.clip(last_token_idx, 0, hidden.shape[1] - 1)
    emb = hidden[np.arange(hidden.shape[0]), last_token_idx]

    if emb.shape[1] < EMBED_OUTPUT_DIM:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned {emb.shape[1]} dims, smaller than EMBED_OUTPUT_DIM={EMBED_OUTPUT_DIM}",
        )
    if emb.shape[1] > EMBED_OUTPUT_DIM:
        emb = emb[:, :EMBED_OUTPUT_DIM]

    emb = _normalize(emb.astype(np.float32))
    return EmbedResponse(embeddings=emb.tolist(), model=req.model or EMBED_MODEL_ID)
PY
```

## 12. Build and Run Gateway Container
> Why this step exists: Packaging the gateway in Docker makes runtime consistent with deployment.
> What this enables: A stable `/embed` service restartable by policy (`unless-stopped`) with explicit env-based config.

Remove old container if present:

```bash
docker rm -f embed-gateway 2>/dev/null || true
```

Build image:

```bash
docker build -t qwen3-embed-gateway:latest "$GATEWAY_DIR"
```

Run container:

```bash
docker run -d \
  --name embed-gateway \
  --network host \
  --restart unless-stopped \
  -v "$BASE/models/onnx":/models:ro \
  -e TRITON_URL=http://localhost:8200 \
  -e TRITON_MODEL=qwen3_embed_4b_onnx \
  -e TOKENIZER_PATH=/models/qwen3-embedding-4b-zhiqing \
  -e EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B \
  -e EMBED_MAX_LENGTH=1024 \
  -e EMBED_OUTPUT_DIM=1024 \
  -e EMBED_TIMEOUT_S=180 \
  qwen3-embed-gateway:latest
```

Why `--network host`:
- Simplifies localhost connectivity from gateway to Triton.

## 13. Validate Gateway End-to-End
> Why this step exists: You need proof the full path works, not just individual components.
> What this enables: Confidence that requests return correct shape, model identity, and normalized vectors.

### 13.1 Health Endpoint
> Why: Confirms gateway startup and Triton connectivity in one call.
> What for: Quick status check for runbooks, monitoring, and incident triage.

```bash
curl -s http://localhost:8101/health | jq
```

Expected:
- `status: ok`
- `triton_model: qwen3_embed_4b_onnx`
- `model_hidden_dim: 2560`
- `embed_output_dim: 1024`

### 13.2 Real Embedding Request
> Why: Synthetic health probes do not prove inference correctness.
> What for: Verifies true request/response behavior and output dimensions.

```bash
cat > /tmp/embed_req.json <<'JSON'
{"texts":["hello world","postgres paradedb retrieval"]}
JSON

curl -s -X POST http://localhost:8101/embed \
  -H 'content-type: application/json' \
  -d @/tmp/embed_req.json | jq '{model,rows:(.embeddings|length),dim:(.embeddings[0]|length),sample:(.embeddings[0][0:5])}'
```

Expected:
- `model: "Qwen/Qwen3-Embedding-4B"`
- `rows: 2`
- `dim: 1024`

### 13.3 Verify L2 Norm = 1.0
> Why: The gateway contract includes normalized embeddings for consistent similarity math.
> What for: Ensures cosine/dot-product retrieval behavior is stable.

```bash
curl -s -X POST http://localhost:8101/embed \
  -H 'content-type: application/json' \
  -d @/tmp/embed_req.json > /tmp/embed_resp.json

python3 - <<'PY'
import json, math
with open('/tmp/embed_resp.json', 'r') as f:
    data = json.load(f)
for i, vec in enumerate(data['embeddings']):
    n = math.sqrt(sum(x*x for x in vec))
    print(f'vec{i}_norm={n:.6f}')
PY
```

Expected each norm close to `1.000000`.

## 14. Operational Commands
> Why this step exists: Day-2 operations are where most time is lost without a runbook.
> What this enables: Fast inspect/restart/redeploy cycles during active development and troubleshooting.

### 14.1 Status
> Why: You need a fast snapshot of service state while iterating.
> What for: Identifies whether failures are due to stopped/restarting containers.

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

### 14.2 Logs
> Why: Logs contain root-cause information for startup and inference failures.
> What for: Enables targeted fixes instead of guesswork.

```bash
docker logs -n 200 triton
docker logs -n 200 embed-gateway
```

### 14.3 Restart
> Why: Some config changes only apply after process restart.
> What for: Reinitializes services without full teardown.

```bash
docker restart triton
docker restart embed-gateway
```

### 14.4 Stop/Start
> Why: Useful for controlled maintenance windows or resource management.
> What for: Temporarily free GPU/CPU resources and restore services cleanly.

```bash
docker stop triton embed-gateway
docker start triton embed-gateway
```

### 14.5 Remove and Recreate
> Why: Container state can drift during rapid iteration.
> What for: Resets runtime to a known-good declaration from this runbook.

```bash
docker rm -f triton embed-gateway
```

Then rerun the container start commands from sections 9 and 12.

## 15. Common Failure Modes and Fixes
> Why this step exists: These are high-probability issues in this exact stack.
> What this enables: Shorter mean time to recovery with known symptoms and concrete fixes.

### Error: `sending FP16 data via JSON is not supported`
> Why this happens: Triton JSON infer path does not serialize FP16 outputs in plain JSON mode.
> What to do: Use `tritonclient` binary output mode, exactly as implemented in this guide.

Cause:
- Triton HTTP JSON infer path cannot return FP16 output payload in plain JSON mode.

Fix:
- Use `tritonclient` with binary output (`InferRequestedOutput(..., binary_data=True)`), as in this guide.

### Model not `READY` after Triton start
> Why this happens: Most often datatype mismatch, missing ONNX external data, or bad config metadata.
> What to do: Inspect logs, reconcile config with model metadata, and reload.

Cause examples:
- `config.pbtxt` datatype mismatch (e.g. config says FP32 but model emits FP16)
- missing `model.onnx_data`

Fix:
1. `docker logs -n 300 triton`
2. Correct `config.pbtxt`
3. Ensure both `model.onnx` and `model.onnx_data` exist
4. Restart Triton

### Gateway container keeps restarting
> Why this happens: Startup exception in app initialization (paths, metadata calls, or code errors).
> What to do: Read `embed-gateway` logs, patch, rebuild, and recreate.

Cause:
- startup exception in `app.py` (bad env path, bad model metadata call, syntax bug)

Fix:
1. `docker logs -n 200 embed-gateway`
2. Patch file
3. Rebuild image
4. Recreate container

### OOM or memory pressure
> Why this happens: Model size and request batching exceed current memory envelope.
> What to do: Reduce batch size/concurrency or unload unused models.

Fixes:
- lower Triton `max_batch_size`
- reduce request concurrency
- remove unused models from Triton repo

### Downloaded files are root-owned
> Why this happens: Files were created by a root-run process/container.
> What to do: Reassign ownership so normal user operations (edit/copy/rebuild) succeed.

Fix:

```bash
sudo chown -R "$HOST_USER":"$HOST_USER" "$BASE"
```

## 16. Verify GPU Utilization and Memory
> Why this step exists: You need to verify work is actually on GPU and understand memory headroom.
> What this enables: Better tuning decisions for batch size, concurrency, and model loading strategy.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
```

Use this while embedding requests are running to verify active GPU usage.

## 17. Integrating with Your RAG App
> Why this step exists: Infrastructure only matters once your app can consume it.
> What this enables: Correct API contract wiring from your RAG pipeline into the embedding service.

When your app needs embeddings, configure it to call:

- `http://<P620-IP>:8101/embed`

Request body:

```json
{"texts": ["text 1", "text 2"]}
```

Response shape:
- `embeddings`: list of vectors
- each vector length: `1024`

## 18. Exact Versions Used in This Working Setup
> Why this step exists: Version drift is one of the biggest causes of non-reproducible behavior.
> What this enables: Recreating the same working environment later or on another host with fewer surprises.

- OS: Ubuntu `24.04.3`
- GPU: RTX `3090` (24 GB)
- Triton image: `nvcr.io/nvidia/tritonserver:26.01-py3`
- Triton model name: `qwen3_embed_4b_onnx`
- Gateway image base: `python:3.11-slim`
- Gateway deps:
  - fastapi `0.116.1`
  - uvicorn `0.35.0`
  - numpy `2.3.2`
  - requests `2.32.5`
  - transformers `4.55.4`
  - tritonclient[http] `2.65.0`
- Tokenizer path: `/models/qwen3-embedding-4b-zhiqing`
- Output contract: `1024` dims (from hidden size `2560`)

## 19. One-Command Smoke Test (After Setup)
> Why this step exists: You need a fast go/no-go check after any change.
> What this enables: Quick regression detection after upgrades, reboots, or container rebuilds.

```bash
set -euo pipefail
curl -s -o /dev/null -w "triton_ready=%{http_code}\n" http://localhost:8200/v2/health/ready
curl -s -o /dev/null -w "gateway_health=%{http_code}\n" http://localhost:8101/health
cat > /tmp/embed_req.json <<'JSON'
{"texts":["smoke test"]}
JSON
curl -s -X POST http://localhost:8101/embed -H 'content-type: application/json' -d @/tmp/embed_req.json | jq '{dim:(.embeddings[0]|length),model}'
```

Expected final output includes `dim: 1024`.

---

If this runbook is followed exactly, you should end up with the same working 4B embedding service topology currently running on the P620.

## 20. Command-by-Command Intent Reference
> Why this section exists: A beginner should understand each command before executing it.
> What this enables: You can reason about failures and safely adapt commands to your own environment.

This section explains what each command block in the runbook is doing.

### 20.1 Preflight commands

- `df -h /`
  - Shows free/used space on root filesystem in human-readable units.
  - Used to verify enough storage for model files and Docker layers.
- `nvidia-smi`
  - Queries NVIDIA driver + GPU state.
  - Confirms driver is installed and GPU is visible to host.

### 20.2 Variable setup commands

- `export HOST_USER="$USER"`
  - Stores current Linux username into `HOST_USER` variable.
- `export BASE="/home/$HOST_USER/nvidia"`
  - Defines root working directory for all NVIDIA assets.
- `export MODEL_SRC_DIR=...`
  - Defines where ONNX model files should exist.
- `export TRITON_REPO=...`
  - Defines Triton model repository directory path.
- `export GATEWAY_DIR=...`
  - Defines local source directory for embedding gateway files.
- `mkdir -p "$BASE"`
  - Creates base directory if missing; does nothing if it already exists.
- `echo "$HOST_USER"` / `echo "$BASE"`
  - Prints resolved values to confirm variables are correct.

### 20.3 Host dependency commands

- `sudo apt-get update`
  - Refreshes apt package index metadata.
- `sudo apt-get install -y ...`
  - Installs required system packages without interactive prompts.
- `git lfs install`
  - Initializes Git LFS hooks so large files (like model artifacts) download correctly.
- `sudo ubuntu-drivers autoinstall`
  - Installs recommended proprietary NVIDIA driver automatically.
- `sudo reboot`
  - Reboots to activate newly installed kernel modules/driver.

### 20.4 Docker repository + engine commands

- `sudo install -m 0755 -d /etc/apt/keyrings`
  - Creates keyring directory with proper permissions.
- `curl -fsSL ... | sudo gpg --dearmor -o ...`
  - Downloads Docker GPG key and stores in binary keyring format.
- `sudo chmod a+r /etc/apt/keyrings/docker.gpg`
  - Makes keyring readable by apt process.
- `echo "deb ..." | sudo tee /etc/apt/sources.list.d/docker.list`
  - Registers Docker apt repository on this host.
- `sudo apt-get install -y docker-ce ...`
  - Installs Docker engine, CLI, Buildx, Compose plugin.
- `sudo usermod -aG docker "$USER"`
  - Adds current user to Docker group for non-sudo Docker commands.
- `docker --version`
  - Prints installed Docker version.
- `docker run --rm hello-world`
  - Smoke test that Docker can run containers.

### 20.5 NVIDIA container runtime commands

- `curl ... nvidia-container-toolkit gpgkey | sudo gpg --dearmor ...`
  - Adds NVIDIA toolkit signing key.
- `curl ... nvidia-container-toolkit.list | sed ... | sudo tee ...`
  - Adds NVIDIA toolkit apt repository with keyring binding.
- `sudo apt-get install -y nvidia-container-toolkit`
  - Installs NVIDIA runtime integration for Docker.
- `sudo nvidia-ctk runtime configure --runtime=docker`
  - Writes Docker runtime config for NVIDIA GPUs.
- `sudo systemctl restart docker`
  - Reloads Docker daemon with new runtime config.
- `docker run --rm --gpus all nvidia/cuda:... nvidia-smi`
  - Confirms GPU visibility inside Docker, not just on host.

### 20.6 NGC auth + image pinning commands

- `docker login nvcr.io`
  - Authenticates Docker client to NVIDIA NGC registry.
- `docker pull nvcr.io/nvidia/tritonserver:26.01-py3`
  - Downloads Triton server image with chosen release tag.
- `docker inspect --format='{{index .RepoDigests 0}}' ...`
  - Extracts immutable digest for reproducibility.
- `mkdir -p "$BASE"; { ... } > "$BASE/VERSIONS.local.md"`
  - Writes local version/digest manifest file.
- `cat "$BASE/VERSIONS.local.md"`
  - Verifies pinned digest file content.

### 20.7 Directory and model artifact commands

- `mkdir -p "$BASE/models/onnx" "$TRITON_REPO" "$GATEWAY_DIR"`
  - Creates all required top-level directories in one go.
- `cd "$BASE/models/onnx"`
  - Moves into model storage directory.
- `git clone https://huggingface.co/... qwen3-embedding-4b-zhiqing`
  - Downloads ONNX model repository (with LFS files).
- `ls -lah "$MODEL_SRC_DIR"`
  - Lists model files to verify required assets exist.
- `sudo chown -R "$HOST_USER":"$HOST_USER" "$MODEL_SRC_DIR"`
  - Fixes file ownership if artifacts were created by root.

### 20.8 Triton model repository commands

- `mkdir -p "$TRITON_REPO/qwen3_embed_4b_onnx/1"`
  - Creates Triton model folder with version subfolder.
- `cp -f ...model.onnx .../1/model.onnx`
  - Copies ONNX graph file into Triton model version directory.
- `cp -f ...model.onnx_data .../1/model.onnx_data`
  - Copies external tensor weights file required by model.
- `cat > .../config.pbtxt <<'PBTXT' ...`
  - Writes Triton model configuration file.

### 20.9 Triton container lifecycle commands

- `docker rm -f triton 2>/dev/null || true`
  - Removes old triton container if present; ignores error when absent.
- `docker run -d --name triton --gpus all ...`
  - Starts Triton in background with GPU access, mounted model repo, and published ports.
- `--restart unless-stopped`
  - Auto-restarts container after reboot/crash unless manually stopped.
- `--shm-size=1g`
  - Allocates larger shared memory region for model runtime stability.
- `-p 8200:8000 -p 8201:8001 -p 8202:8002`
  - Maps host ports to Triton HTTP/gRPC/metrics ports.
- `-v "$TRITON_REPO":/models`
  - Mounts host model repository at `/models` inside container.
- `tritonserver --model-repository=/models`
  - Launches Triton process against mounted model repository.

### 20.10 Triton validation commands

- `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8200/v2/health/live`
  - Checks if Triton process is alive.
- `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8200/v2/health/ready`
  - Checks if Triton is ready to serve requests.
- `curl -s -X POST http://localhost:8200/v2/repository/index | jq`
  - Lists models and states from repository index.
- `curl -s http://localhost:8200/v2/models/qwen3_embed_4b_onnx | jq`
  - Displays exact model metadata (inputs/outputs/types/shapes).
- `docker logs -n 300 triton`
  - Prints recent Triton logs for debugging load failures.

### 20.11 Gateway file creation commands

- `cat > "$GATEWAY_DIR/Dockerfile" <<'DOCKERFILE' ...`
  - Creates Dockerfile for FastAPI gateway image.
- `cat > "$GATEWAY_DIR/requirements.txt" <<'REQ' ...`
  - Creates pinned Python dependency list.
- `cat > "$GATEWAY_DIR/app.py" <<'PY' ...`
  - Creates gateway API implementation.

### 20.12 Gateway build/run commands

- `docker rm -f embed-gateway 2>/dev/null || true`
  - Removes prior gateway container safely.
- `docker build -t qwen3-embed-gateway:latest "$GATEWAY_DIR"`
  - Builds gateway image from local Dockerfile and source.
- `docker run -d --name embed-gateway --network host ...`
  - Starts gateway container in background.
- `--network host`
  - Makes container use host network namespace (gateway can reach `localhost:8200`).
- `-v "$BASE/models/onnx":/models:ro`
  - Mounts tokenizer/model assets read-only into container.
- `-e TRITON_URL=http://localhost:8200`
  - Points gateway to Triton HTTP endpoint.
- `-e TRITON_MODEL=qwen3_embed_4b_onnx`
  - Selects model name gateway should call.
- `-e TOKENIZER_PATH=/models/qwen3-embedding-4b-zhiqing`
  - Selects local tokenizer path inside container.
- `-e EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B`
  - Sets model ID returned in API response.
- `-e EMBED_MAX_LENGTH=1024`
  - Sets tokenizer truncation max length.
- `-e EMBED_OUTPUT_DIM=1024`
  - Enforces output vector length expected by app.
- `-e EMBED_TIMEOUT_S=180`
  - Sets metadata/health timeout window used by gateway.

### 20.13 Gateway validation commands

- `curl -s http://localhost:8101/health | jq`
  - Verifies gateway is running and sees Triton/model settings.
- `cat > /tmp/embed_req.json <<'JSON' ...`
  - Writes test request payload for consistent reuse.
- `curl -s -X POST http://localhost:8101/embed -H 'content-type: application/json' -d @/tmp/embed_req.json | jq ...`
  - Performs real embedding call and prints key response fields.
- `curl ... > /tmp/embed_resp.json`
  - Saves full embed response for offline checks.
- `python3 - <<'PY' ...`
  - Calculates L2 norms of returned vectors to verify normalization.

### 20.14 Operational command intent

- `docker ps --format 'table ...'`
  - Shows container status quickly.
- `docker logs -n 200 triton` / `docker logs -n 200 embed-gateway`
  - Shows recent logs for targeted service.
- `docker restart triton` / `docker restart embed-gateway`
  - Restarts service process in-place.
- `docker stop ...` / `docker start ...`
  - Controlled shutdown/startup.
- `docker rm -f ...`
  - Force removes containers so they can be recreated cleanly.

### 20.15 GPU and integration command intent

- `nvidia-smi --query-gpu=... --format=csv,noheader`
  - Prints GPU memory and utilization metrics in parseable form.
- `set -euo pipefail`
  - Shell safety mode for smoke tests:
  - `-e`: exit on first failing command.
  - `-u`: fail on unset variables.
  - `-o pipefail`: fail pipelines when any command fails.
- `curl ... -w "..."`
  - Uses format output to print concise HTTP status success/failure.

### 20.16 How to adapt commands safely

- Replace only variables at top (`HOST_USER`, `BASE`, ports, model name) rather than editing every command manually.
- Keep `config.pbtxt` datatype and shape aligned with actual ONNX metadata.
- If changing ports, update both container port mappings and any `TRITON_URL`/gateway endpoints.
- If changing model family, re-check tokenizer path and output hidden size, then adjust `EMBED_OUTPUT_DIM` logic as needed.
