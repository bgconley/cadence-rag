# NVIDIA Immersion Track (Cadence RAG)

This is a human-run, step-by-step plan to learn NVIDIA's inference + tooling ecosystem
by deploying GPU endpoints that this repo will call (embeddings, rerank, LLM), then
adding profiling and telemetry.

This plan is intentionally separate from the canonical product docs:
- `APP_SPEC.md` (contracts)
- `IMPLEMENTATION_PLAN.md` (non-negotiables)
- `PHASED_PLAN.md` (phase ordering)

## How To Use This Plan (Triton/TensorRT-First)
This version of the immersion track intentionally skips "Path A (NIM-first)" and goes straight
to Triton/TensorRT so you can host BYO models (Qwen3 embeddings + Qwen3 reranker) while learning
the NVIDIA stack deeply.

1. Do Milestone 0 to confirm the RAG stack works (CPU-only).
2. Do Milestone 1 to confirm GPU-in-Docker works.
3. Do Milestone 2 to stand up Triton and learn the model repository + serving primitives.
4. Do Milestone 3 to serve embeddings behind Triton and expose a stable HTTP `/embed` gateway.
5. Do Milestone 4 to serve reranking behind Triton and expose a stable HTTP `/rerank` gateway.
6. Do Milestone 5 to serve an LLM (TensorRT-LLM) behind an OpenAI-compatible gateway.
7. Do Milestone 6 and 7 to add profiling (Nsight) and telemetry (DCGM + NVML).
8. Do Milestone 8 to wire this repo (Phase 3/4/5) to your endpoints via env vars.

Optional later:
- Use NIM as a comparison point (curated models, fastest time-to-first-token), but it is not required.

## NVIDIA Components (Quick Mental Model)
This is the glossary that makes the rest of the plan make sense.

- NIM (NVIDIA Inference Microservices): pre-packaged inference containers for specific, curated models
  (plus their serving stack). Operationally easy, but you only get the models NVIDIA publishes as NIMs.
  This is why NIM can feel "misaligned" if your repo expects specific models (like Qwen3 embed + rerank)
  and a NIM for those exact models is not available.
- Triton Inference Server: a general-purpose NVIDIA model server. You bring models (ONNX, TensorRT engines,
  Python backend, etc.). Triton handles: multi-model serving, dynamic batching, multiple backends, metrics,
  and standard inference APIs.
- TensorRT: an optimizer/compiler that turns models (often ONNX) into highly optimized GPU engines.
  In Triton, those engines are served via the TensorRT backend (`platform: "tensorrt_plan"`).
- TensorRT-LLM: NVIDIA's LLM-specific inference stack. Often deployed behind Triton (TensorRT-LLM backend),
  or as its own server. In this plan we keep an OpenAI-compatible gateway in front either way.
- Nsight Systems / Nsight Compute: profiling tools to find latency hotspots and GPU inefficiencies.
- DCGM + NVML: telemetry/health (GPU utilization, memory, power, temps). Useful for debugging and later
  optional integration into `GET /diagnostics`.

Why we're skipping "NIM-first" here:
- This repo's GPU profiles (in `APP_SPEC.md`) choose Qwen3 embeddings/rerank models and require `dim=1024`.
- NIM is not a generic "run any HF model" wrapper; it's a productized, model-specific container.
- Triton + TensorRT is the NVIDIA-native "BYO model" path, so it stays aligned with our pipeline choices.

## Repo Non-Negotiables
- Postgres (ParadeDB) remains the canonical store.
- Retrieval is deterministic and server-orchestrated (no LLM tool loops).
- Evidence-pack answering uses strict citation gating (Phase 5).
- Embedding dimension is standardized to 1024.

## Assumptions (P620 + RTX 3080 -> RTX 3090)
- OS: Ubuntu 24.04.3 LTS.
- Current GPU: RTX 3080 (10GB VRAM). Near-term upgrade: RTX 3090 (24GB VRAM).
- Plan for success on the 3080 first (smaller batches, shorter max lengths), then re-tune on the 3090.
- You will run inference endpoints as separate processes/containers.

## Suggested Local Port Map
- Postgres: `5432`
- RAG API: `8001`
- Embeddings: `8101`
- Rerank: `8102`
- LLM (OpenAI-compatible): `8103`
- DCGM exporter: `9400`
- Triton HTTP: `8200` -> `8000` (container)
- Triton gRPC: `8201` -> `8001` (container)
- Triton metrics: `8202` -> `8002` (container)

## Serving Contracts (What The RAG API Will Call)
Embeddings (Phase 3 client will call this):
```json
POST /embed
{"texts":["..."],"model":null}
-> {"embeddings":[[0.1,0.2,...]],"model":"..."}
```
Rerank (Phase 4 client will call this):
```json
POST /rerank
{"query":"...","documents":["...","..."]}
-> {"scores":[0.9,0.2],"order":[0,1],"model":"..."}
```
LLM (Phase 5 will call this): OpenAI-compatible `POST /v1/chat/completions`.

Important:
- These contracts are for your gateway services (thin adapters).
- Triton itself exposes `/v2/models/.../infer` and does not handle tokenization/pooling; the gateways do.

## Architecture Overview (How Triton Fits The RAG Pipeline)
This repo is a deterministic RAG orchestrator. The GPU services are "dumb" inference endpoints:
they do exactly what they are asked (embed, rerank, generate) and do not decide retrieval steps.

Why we split "RAG API" from "GPU inference":
- Determinism: retrieval and evidence-pack construction must be server-orchestrated.
- Swapability: Triton/TensorRT engines can evolve without rewriting RAG logic.
- Reproducibility: we can pin container digests and record model IDs/configs.

### Component Diagram
```mermaid
flowchart LR
  Client[Client] -->|HTTP| API[FastAPI RAG API]

  API -->|POST /embed| EG[Embeddings Gateway]
  EG -->|Infer| TRITON[Triton Inference Server]
  TRITON -->|TensorRT engine| GPU[(GPU)]

  API -->|POST /rerank| RG[Rerank Gateway]
  RG -->|Infer| TRITON

  API -->|POST /v1/chat/completions| LG[LLM Gateway (OpenAI-compat)]
  LG -->|Infer| TRTLLM[TensorRT-LLM (often behind Triton)]
  TRTLLM --> GPU

  API -->|SQL| PG[(Postgres + ParadeDB)]

  DCGM[DCGM Exporter] -. GPU metrics .-> GPU
```

### How This Maps To The RAG Data Flow
Ingestion flow (Phase 1 + Phase 3 backfill):
1. Transcript/analysis is ingested into Postgres (`utterances`, `chunks`, `analysis_artifacts`).
2. Embeddings service generates 1024-d vectors for `chunks.embedding` and `analysis_artifacts.embedding`.
3. Postgres indexes (pgvector HNSW + BM25 + tech_tokens) support deterministic retrieval.

Query flow (Phase 2/3/4/5):
1. `/retrieve` runs BM25 + tech_tokens + dense (pgvector) lanes, then fuses candidates (RRF).
2. `/retrieve` optionally calls rerank to reorder top-N candidates and outputs a budgeted evidence pack.
3. `/answer` calls the LLM with the evidence pack and enforces citation gating.

## Milestone 0: Baseline RAG Stack (CPU-only)
Outcome: DB + API run locally, migrations apply cleanly.

1. Start DB:
```bash
docker compose up -d db
```
2. Create env and install:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
3. Run migrations:
```bash
uv run alembic upgrade head
```
4. Run the API:
```bash
uv run uvicorn app.main:app --reload --port 8001
```
5. Verify:
```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8001/diagnostics
```

## Milestone 1: GPU Runtime (Driver + Docker + NVIDIA Container Toolkit)
Outcome: `docker run --gpus all ... nvidia-smi` works.

1. Install a recommended NVIDIA driver (Ubuntu 24.04.x):
```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

If Secure Boot is enabled, you may be prompted to enroll a MOK key on reboot.
Do that, or disable Secure Boot, otherwise the kernel driver may not load.

2. Verify the driver loaded:
```bash
nvidia-smi
```

3. Install Docker Engine + Compose (so `docker compose` works).

Option A (quick start; Ubuntu packages):
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Option B (recommended; Docker's official repo): follow Docker's Ubuntu install docs.

4. Install NVIDIA Container Toolkit (enables `--gpus all` in Docker).

On Ubuntu, you will typically add NVIDIA's apt repo and then install:
```bash
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})

sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

5. Verify GPU access from containers:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, stop here and fix the driver/runtime before moving on.

## Milestone 2: Triton First (Baseline, Primary Path)
Outcome: Triton is running, you understand how the model repository works, and you can reach:
- HTTP inference: `http://localhost:8200`
- Metrics: `http://localhost:8202/metrics`

Why this milestone exists (the "why" matters for later troubleshooting):
- Triton is the NVIDIA-native "serving plane" for GPU inference: one server process can host multiple models,
  support dynamic batching, expose metrics, and run multiple backends (TensorRT, Python, ONNX Runtime, etc.).
- Our RAG API should not care how inference is implemented. It should call stable HTTP gateways (`/embed`, `/rerank`, OpenAI chat).
  Triton is an implementation detail behind those gateways.
- When you upgrade GPUs (3080 -> 3090), Triton lets you re-tune batch sizes/engine builds without changing RAG logic.

### Step 2.0) Login To NGC (So You Can Pull NVIDIA Containers)
Why:
- Triton and TensorRT containers are hosted on `nvcr.io` (NVIDIA NGC). Docker needs credentials to pull them.

How:
```bash
export NGC_API_KEY='...'
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Security:
- Never commit `NGC_API_KEY` into this repo.
- Prefer to keep it only in your shell session or a secrets manager.

### Step 2.1) Pick And Pin Your Triton Version
Why:
- Triton includes specific CUDA/TensorRT versions. Pinning avoids "it worked yesterday" surprises.
- You will likely use multiple NVIDIA containers (Triton server, TensorRT tools, Triton SDK tools).
  Keeping them from the same release family reduces ABI/version weirdness.

How:
1. Pick a single release family (usually `YY.MM`) and use it consistently across containers.

Minimum set you will use in this plan:
- Triton server (serving plane): `nvcr.io/nvidia/tritonserver:<tag>`
- TensorRT tools (engine builds, `trtexec`, Polygraphy): `nvcr.io/nvidia/tensorrt:<tag>`
- Triton SDK tools (perf testing via `perf_analyzer`, client examples): pick the Triton SDK container
  from NGC that matches your Triton server release family.

Later (Milestone 5):
- TensorRT-LLM container(s): pick from NGC for engine build/serving, ideally matching your release family.

2. Pull each image and record the digest (the digest is what you pin):
```bash
export TRITON_IMAGE="nvcr.io/nvidia/tritonserver:<tag>"
export TENSORRT_IMAGE="nvcr.io/nvidia/tensorrt:<tag>"
export TRITON_SDK_IMAGE="nvcr.io/nvidia/<triton-sdk-image>:<tag>"

for img in "$TRITON_IMAGE" "$TENSORRT_IMAGE" "$TRITON_SDK_IMAGE"; do
  docker pull "$img"
  docker inspect --format='{{index .RepoDigests 0}}' "$img"
done
```

3. Save the digests somewhere stable for yourself (for example, in `nvidia/VERSIONS.local.md`).
   Keep it out of git if it ever contains secrets (it shouldn't, but keep the habit).

Sanity check (avoid CUDA/driver mismatches):
- `nvidia-smi` shows your driver version. Container CUDA must be supported by that driver.
- If you see mysterious runtime errors, confirm driver/container CUDA compatibility first.

### Step 2.2) Create A Model Repository Directory
Why:
- Triton loads models from a filesystem repository with a strict layout. This becomes your "model deployment artifact."

How:
```bash
mkdir -p nvidia/triton/model_repository
```

Repository layout basics (you will use this in Milestone 3/4/5):
- `nvidia/triton/model_repository/<model_name>/config.pbtxt`
- `nvidia/triton/model_repository/<model_name>/<version>/model.plan` (TensorRT)
- `nvidia/triton/model_repository/<model_name>/<version>/model.py` (Python backend)

### Step 2.3) Run Triton (HTTP + gRPC + metrics)
Why:
- Triton's HTTP port is for inference requests.
- Triton's metrics port is how you learn throughput/latency and debug batching behavior.

How:
```bash
docker run -d --gpus all --name triton --restart unless-stopped \
  -p 8200:8000 -p 8201:8001 -p 8202:8002 \
  -v "$PWD/nvidia/triton/model_repository:/models" \
  "$TRITON_IMAGE" \
  tritonserver --model-repository=/models --log-verbose=1 --disable-auto-complete-config=true
```

### Step 2.4) Verify Readiness + Metrics
Why:
- Always validate the serving plane before debugging models.

How:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8200/v2/health/ready
curl -s http://localhost:8202/metrics | head
```

If readiness is not `200`, inspect logs:
```bash
docker logs -n 200 triton
```

## Milestone 3: Embeddings In Triton (BYO Model, 1024-d)
Outcome: you have an embeddings HTTP service on `http://localhost:8101/embed` that:
- accepts batches of texts
- returns deterministic 1024-d vectors
- is backed by Triton (and ideally TensorRT)

Why embeddings are special (this is the most important "why" in RAG):
- The embeddings define the vector space stored in Postgres (`vector(1024)` columns).
- If you switch embedding models later, you must re-embed all stored rows, because the space changes.
- So you want to get the embedder + preprocessing + normalization stable early and pin versions.

### Step 3.1) Decide Your Embedding Contract (Invariants)
These invariants keep the RAG repo stable:
1. Output length is exactly 1024 floats.
2. Output is deterministic for the same input (temperature/dropout off).
3. Same normalization every time (if you L2-normalize, do it always).
4. The service supports batching (for backfill speed).

### Step 3.2) Choose A Model (Align With `APP_SPEC.md`)
Default (per `APP_SPEC.md`):
- Qwen3 embedding model (dim 1024), Profile A (10GB): `Qwen/Qwen3-Embedding-0.6B`
- Qwen3 embedding model, Profile B (24GB): `Qwen/Qwen3-Embedding-4B` (verify how you get to 1024-d)

Important "why":
- The model ID is not just a name: it implies a tokenizer, preprocessing behavior, and output shape.
- Record the exact HF revision/commit (or NGC artifact digest) you use. Otherwise re-embedding later is painful.

### Step 3.2B) Qwen3-Embedding Reality Check (Do Not Skip)
This is the part most guides gloss over, and it is why "generic BERT-style embedding recipes" often fail.

What is different about Qwen3-Embedding vs common embedders (MiniLM, E5, etc.):
- The underlying architecture is decoder-style (LLM-ish), not a classic encoder-only BERT.
- Many "embedding correctness" details live outside the raw forward pass:
  - pooling rule (commonly last-token pooling for Qwen3 embedding stacks)
  - whether/how an EOS token is appended
  - whether you L2-normalize the final vector
  - padding side (left vs right) and pad token choice

Implication for Triton:
- You should not assume your ONNX model will output a single `[B, 1024]` "sentence embedding".
- In many exports, the ONNX output is token-level hidden states (shape like `[B, S, 1024]`).
- Therefore, the gateway (or a Triton ensemble / postprocess model) must implement:
  - last-token pooling (not mean pooling)
  - optional L2 normalization
  - consistent tokenization (including EOS behavior)

Practical guardrails (so results stay stable across runs):
1. Decide and document one pooling rule: for Qwen3, start with last-token pooling.
2. Decide and document whether you L2-normalize. If yes, do it always.
3. Force one padding policy in the gateway:
   - set `tokenizer.padding_side = "left"`
   - set `tokenizer.pad_token = tokenizer.eos_token` (if pad token is otherwise undefined)
4. Keep max length conservative on RTX 3080 (start at 128/256) and only raise it after you have correctness.

If Qwen3 export to TensorRT is hard at first:
- You can still stay NVIDIA-native while de-risking:
  - First, serve ONNX in Triton via the ONNX Runtime backend (fast correctness win).
  - Then, compile ONNX -> TensorRT engine for throughput.
This lets you learn Triton/TensorRT without blocking Phase 3 engineering.

### Step 3.3) Create Directories (Artifacts + Triton Model Repo)
Why:
- You want a clean separation between "model source artifacts" (ONNX) and "deployment artifacts" (TensorRT engines in Triton repo).

How:
```bash
mkdir -p nvidia/models/onnx
mkdir -p nvidia/triton/model_repository/embed_trt/1
```

### Step 3.4) Export The Embedder To ONNX
Why:
- ONNX is the common interchange format that TensorRT can compile into an optimized engine.

How (detailed, first-success path):

You need three things to create a correct Triton `config.pbtxt`:
1. The ONNX file.
2. The ONNX *tensor names* for inputs/outputs.
3. The ONNX *tensor shapes and dtypes* (so Triton can validate requests).

#### Step 3.4.1) Create A Dedicated Export Environment
Why:
- ONNX export depends on specific versions of `torch`, `transformers`, and export tooling.
- Keeping export tooling separate from the RAG app's runtime deps reduces "dependency spaghetti."

How (example using a standalone venv):
```bash
python3 -m venv .venv_onnx
source .venv_onnx/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers optimum onnx onnxruntime
```

Note:
- You can also use `uv` if you prefer, but keep the idea: export tooling is its own environment.

#### Step 3.4.2) Confirm The Embedder's Output Dim (Should Be 1024)
Why:
- `APP_SPEC.md` requires vectors stored in Postgres to be `vector(1024)`.
- You must not "discover later" that the model outputs 768/1536/etc.

How:
```bash
python - <<'PY'
from transformers import AutoConfig
mid = "Qwen/Qwen3-Embedding-0.6B"
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
print("model:", mid)
print("hidden_size:", getattr(cfg, "hidden_size", None))
PY
```

If `hidden_size` is not obvious or missing, do a tiny forward pass and print shapes:
```bash
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer

mid = "Qwen/Qwen3-Embedding-0.6B"
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=True)
mdl = AutoModel.from_pretrained(mid, trust_remote_code=True).eval()

enc = tok(["hello world"], return_tensors="pt", padding=True, truncation=True, max_length=16)
with torch.no_grad():
    out = mdl(**enc)
print("out keys:", out.keys() if hasattr(out, "keys") else type(out))
if hasattr(out, "last_hidden_state"):
    print("last_hidden_state:", tuple(out.last_hidden_state.shape))
elif isinstance(out, (tuple, list)):
    print("tuple[0]:", tuple(out[0].shape))
PY
```

You are looking for a hidden dimension of 1024 somewhere (often the last dim of `last_hidden_state`).

#### Step 3.4.3) Export ONNX (Two Practical Options)
Goal:
- Start with a simple, fixed max sequence length (e.g., 128) to reduce moving parts.
- You can make shapes dynamic later once everything works.

Option A (recommended first): Optimum CLI export
```bash
export EMBEDDER_ID="Qwen/Qwen3-Embedding-0.6B"
export SEQ_LEN=128

optimum-cli export onnx \
  --model "$EMBEDDER_ID" \
  --task feature-extraction \
  --framework pt \
  --opset 17 \
  --sequence_length "$SEQ_LEN" \
  nvidia/models/onnx/qwen3_embed_${SEQ_LEN}
```

This should produce an ONNX file under `nvidia/models/onnx/qwen3_embed_${SEQ_LEN}/`.

Find the ONNX file that was produced (do not guess the filename):
```bash
find nvidia/models/onnx/qwen3_embed_${SEQ_LEN} -maxdepth 2 -type f -name '*.onnx' -print
```

For the rest of this guide, standardize on this filename:
- `nvidia/models/onnx/qwen3_embed_${SEQ_LEN}/model.onnx`

If your export produced a different name, copy it into place:
```bash
cp "<PATH_FROM_FIND>.onnx" "nvidia/models/onnx/qwen3_embed_${SEQ_LEN}/model.onnx"
```

Option B: Use a small Python export script
Why:
- Some models require `trust_remote_code=True` and custom export handling.
- A script gives you more control when Optimum doesn't "just work."

High-level approach:
- Load tokenizer + model in eval mode.
- Create example inputs with `max_length=SEQ_LEN`.
- Export with `torch.onnx.export(...)`.

#### Step 3.4.4) Inspect The ONNX Graph (Names, Shapes, Dtypes)
Why:
- Triton config must match the ONNX graph exactly, especially under `--disable-auto-complete-config=true`.

How:
```bash
python - <<'PY'
import os
import onnx
from onnx import numpy_helper

onnx_path = os.environ.get("ONNX_PATH", "nvidia/models/onnx/qwen3_embed_128/model.onnx")
m = onnx.load(onnx_path)
print("ONNX:", onnx_path)
print("inputs:")
for i in m.graph.input:
    t = i.type.tensor_type
    shape = [d.dim_value if d.dim_value else -1 for d in t.shape.dim]
    dtype = t.elem_type
    print(" -", i.name, "dtype_enum=", dtype, "shape=", shape)
print("outputs:")
for o in m.graph.output:
    t = o.type.tensor_type
    shape = [d.dim_value if d.dim_value else -1 for d in t.shape.dim]
    dtype = t.elem_type
    print(" -", o.name, "dtype_enum=", dtype, "shape=", shape)
PY
```

Interpretation notes:
- ONNX shape includes batch. Triton `dims` excludes batch if `max_batch_size > 0`.
- Token IDs are often `INT64` in ONNX exports. If your ONNX input is `INT64`, your Triton input must be `TYPE_INT64`.
- Outputs are often FP32 for ONNX runtime (`TYPE_FP32`). Start with FP32 for correctness; optimize later.

Concrete example (how to translate ONNX shape -> Triton config):
- If ONNX input `input_ids` is shape `[B, 128]`:
  - set `max_batch_size: 16`
  - set input `dims: [ 128 ]` (exclude batch)
- If ONNX output is `[B, 128, 1024]`:
  - set output `dims: [ 128, 1024 ]` (exclude batch)
- If ONNX output is `[B, 1024]`:
  - set output `dims: [ 1024 ]`

#### Step 3.4.5) Quick Numerical Sanity Check (PyTorch vs ONNX Runtime)
Why:
- You want to know your ONNX export is "the same model" before you serve it.

How (shape-only sanity, then compare a few values):
- Run the model in PyTorch and ONNX runtime with the same tokenized inputs.
- Confirm shapes match and values are close (expect small differences).

### Step 3.4B) (Recommended) First Serve The ONNX In Triton (Correctness Before Speed)
Why:
- Debugging TensorRT conversion and debugging model correctness at the same time is slow.
- Serving ONNX via Triton proves your tokenizer + pooling + output interpretation are correct first.

How (detailed, do-not-skip):

#### Step 3.4B.1) Put The ONNX File In The Triton Model Repository
Why:
- Triton only loads models from its repository.
- Triton expects a backend-specific filename (`model.onnx` for ONNX Runtime backend).

How:
```bash
mkdir -p nvidia/triton/model_repository/embed_onnx/1
cp nvidia/models/onnx/qwen3_embed_128/model.onnx nvidia/triton/model_repository/embed_onnx/1/model.onnx
```

If your export produced a different ONNX path/name, adjust the `cp` accordingly.

#### Step 3.4B.2) Write `config.pbtxt` That Matches Your ONNX Graph
Why:
- This is the single most common failure point: mismatched tensor name / dtype / dims.
- With strict config enabled, Triton will reject mismatches on load or on inference.

How:
1. Decide:
   - `max_batch_size`: start with `16` on RTX 3080 for embeddings.
   - `dims`: derived from ONNX *excluding batch dimension*.
   - `data_type`: derived from ONNX element types.

2. Map ONNX input/output dtypes to Triton types (common ones):
- ONNX `INT64` -> Triton `TYPE_INT64`
- ONNX `INT32` -> Triton `TYPE_INT32`
- ONNX `FLOAT` (FP32) -> Triton `TYPE_FP32`
- ONNX `FLOAT16` (FP16) -> Triton `TYPE_FP16`

3. Write the config.

Template for a typical transformer ONNX with inputs `[B, S]` and outputs `[B, S, H]`:
```text
name: "embed_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  { name: "input_ids" data_type: TYPE_INT64 dims: [ -1 ] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }
]

output [
  { name: "<MODEL_OUTPUT_NAME>" data_type: TYPE_FP32 dims: [ -1, 1024 ] }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16 ]
  max_queue_delay_microseconds: 2000
}

instance_group [{ kind: KIND_GPU count: 1 }]
```

Important:
- Replace `<MODEL_OUTPUT_NAME>` with the exact ONNX output name you printed in Step 3.4.4.
- If your ONNX output is already pooled `[B, 1024]`, set `dims: [ 1024 ]` instead of `[ -1, 1024 ]`.
- If your ONNX export includes additional inputs (like `token_type_ids`), you must include them.
- If your ONNX export uses `INT32` inputs, set `TYPE_INT32` instead of `TYPE_INT64`.

#### Step 3.4B.3) Restart Triton And Confirm It Loaded The Model
How:
```bash
docker restart triton
docker logs -n 200 triton
curl -s http://localhost:8200/v2/models/embed_onnx | jq
```

What you want to see:
- `state: "READY"`
- No errors about "unexpected input" / "datatype mismatch" / "dimension mismatch."

#### Step 3.4B.4) Prove You Can Infer Against `embed_onnx` (Before Building A Gateway)
Why:
- This isolates Triton + model config correctness from gateway bugs.

How (Python client request against Triton HTTP):
```bash
python -m pip install 'tritonclient[http]' numpy
python - <<'PY'
import numpy as np
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8200"
MODEL = "embed_onnx"
OUTPUT = "<MODEL_OUTPUT_NAME>"  # replace

client = httpclient.InferenceServerClient(url=TRITON_URL)

# Fake token IDs to validate the pipe first.
# Replace with real tokenizer output once this returns a tensor.
input_ids = np.ones((1, 8), dtype=np.int64)
attention_mask = np.ones((1, 8), dtype=np.int64)

inputs = []
i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
i0.set_data_from_numpy(input_ids, binary_data=True)
inputs.append(i0)
i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
i1.set_data_from_numpy(attention_mask, binary_data=True)
inputs.append(i1)

outputs = [httpclient.InferRequestedOutput(OUTPUT, binary_data=True)]
res = client.infer(MODEL, inputs=inputs, outputs=outputs)
out = res.as_numpy(OUTPUT)
print("output shape:", out.shape, "dtype:", out.dtype)
PY
```

Now do the same test with real tokenization (this catches name/dtype/shape issues earlier):
```bash
python -m pip install 'tritonclient[http]' numpy transformers
python - <<'PY'
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

TRITON_URL = "127.0.0.1:8200"
MODEL = "embed_onnx"
OUTPUT = "<MODEL_OUTPUT_NAME>"  # replace
TOKENIZER_ID = "Qwen/Qwen3-Embedding-0.6B"
SEQ_LEN = 128

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, use_fast=True)
enc = tok(
    ["hello world"],
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN,
)

# Match your ONNX graph: if it wants INT64, keep int64; if it wants INT32, cast to int32.
input_ids = enc["input_ids"].astype(np.int64)
attention_mask = enc["attention_mask"].astype(np.int64)

client = httpclient.InferenceServerClient(url=TRITON_URL)
inputs = []
i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
i0.set_data_from_numpy(input_ids, binary_data=True)
inputs.append(i0)
i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
i1.set_data_from_numpy(attention_mask, binary_data=True)
inputs.append(i1)

outputs = [httpclient.InferRequestedOutput(OUTPUT, binary_data=True)]
res = client.infer(MODEL, inputs=inputs, outputs=outputs)
out = res.as_numpy(OUTPUT)
print("tokenized input_ids shape:", input_ids.shape, "dtype:", input_ids.dtype)
print("output shape:", out.shape, "dtype:", out.dtype)
PY
```

If this fails:
- It usually means `OUTPUT` name is wrong, or input dtypes/shapes don't match the ONNX graph.
- Fix `config.pbtxt` until this succeeds.

#### Step 3.4B.5) Only Then: Point Your Embeddings Gateway At `embed_onnx`
Why:
- The gateway adds tokenization + pooling + JSON encoding. It is another failure surface.

Once it works end-to-end, upgrade to TensorRT:
- Serve `embed_trt` and switch the gateway's `TRITON_MODEL` from `embed_onnx` to `embed_trt`.

### Step 3.5) Build A TensorRT Engine (ONNX -> model.plan)
Why:
- TensorRT compiles the model into a GPU-optimized engine (kernel fusion, precision selection, scheduling).
- For backfill workloads (many embeddings), this is where you get large throughput gains.

How (template using `trtexec`):
1. Choose a fixed max sequence length to start (e.g., 128). Fixed shapes are the easiest first success.
2. Build FP16 first (good speed/quality tradeoff on Ampere).
3. Save the engine into Triton's repository:
`nvidia/triton/model_repository/embed_trt/1/model.plan`.

### Step 3.6) Write `config.pbtxt` For The TensorRT Engine
Why:
- Triton needs to know tensor names, shapes, and batching behavior.
- This config is also where you enable dynamic batching for throughput.

How (template; you must adapt names + hidden sizes to your ONNX):
```text
name: "embed_trt"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  { name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] },
  { name: "attention_mask" data_type: TYPE_INT32 dims: [ -1 ] }
]

output [
  { name: "<MODEL_OUTPUT_NAME>" data_type: TYPE_FP16 dims: [ -1, <HIDDEN_SIZE> ] }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16 ]
  max_queue_delay_microseconds: 2000
}

instance_group [{ kind: KIND_GPU count: 1 }]
```

Notes:
- In Triton configs, `dims` exclude the batch dimension when `max_batch_size > 0`.
- Many embedders output per-token vectors (`[seq_len, hidden]`). If so, your gateway must pool (mean/cls) to one vector.
- If your model already outputs a single vector, pooling may be unnecessary.
- The final vector returned by your gateway must be length 1024 to match Postgres.

### Step 3.7) Load The Model Into Triton
Why:
- You want Triton to be the stable serving plane. Model load failures are common at first; logs are your friend.

How:
1. Copy `config.pbtxt` into: `nvidia/triton/model_repository/embed_trt/config.pbtxt`
2. Restart Triton to load the new model:
```bash
docker restart triton
docker logs -n 200 triton
```
3. Verify the model is visible:
```bash
curl -s http://localhost:8200/v2/models | jq
curl -s http://localhost:8200/v2/models/embed_trt | jq
```

### Step 3.8) Implement The Embeddings Gateway (Stable HTTP API)
Why:
- Tokenization, pooling, and normalization are easier to iterate in Python than inside TensorRT.
- The gateway is the stable contract the RAG API will call.
- You can swap Triton engines without changing the gateway contract.

What the gateway does:
1. Accept raw texts.
2. Tokenize -> tensors (`input_ids`, `attention_mask`) with the same tokenizer you used for ONNX export.
3. Call Triton inference (`/v2/models/embed_trt/infer`).
4. Pool to a single vector (if needed), project to 1024 (if needed), and normalize (if chosen).
5. Return JSON `{embeddings:[[...]]}`.

### Step 3.9) Verify End-To-End Correctness
Minimum checks:
1. Two calls with the same input return identical vectors.
2. Vector length is 1024.
3. Batch requests return one vector per input.
4. Latency is acceptable and stable (watch Triton metrics on `:8202/metrics`).

RAG wiring (later, when we implement Phase 3 in this repo):
- `EMBEDDINGS_BASE_URL=http://127.0.0.1:8101`
- `EMBEDDINGS_DIM=1024`

## Milestone 4: Rerank Service (Qwen3-Reranker; Two Serving Shapes)
Outcome: you have a rerank HTTP service on `http://localhost:8102/rerank` that:
- accepts `{query, documents[]}`
- returns deterministic relevance scores and an ordering
- is backed by Triton (and ideally TensorRT)

Why reranking matters (and why it is safer to swap than embeddings):
- Reranking happens at query-time only. Swapping rerank models does not require rewriting stored vectors.
- A good reranker dramatically improves top-of-pack precision and reduces redundant evidence.
- Triton dynamic batching is especially valuable here because rerank is usually "many small pairwise inferences."

### Step 4.1) Decide Your Rerank Contract (Invariants)
1. The reranker takes `(query, doc)` pairs and outputs a single score per doc.
2. Document truncation must be deterministic (same truncation rule every time).
3. The service should support batching many docs per query (e.g., 80 docs) efficiently.

### Step 4.2) Choose A Rerank Model (Align With `APP_SPEC.md`)
Default (per `APP_SPEC.md` profile A):
- Qwen3 reranker 0.6B: `Qwen/Qwen3-Reranker-0.6B`

You can also keep a second "bigger reranker" for the RTX 3090 later, but do not change the contract.

### Step 4.2B) Qwen3-Reranker Reality Check (This Is Not A Classic Cross-Encoder)
Many rerank deployments assume an encoder-style "sequence classification" model that outputs a single score tensor.
Qwen3-Reranker is commonly used differently:
- It behaves like a small instruction-tuned LLM where reranking is done by prompting and reading the model's
  preference via logits/probability of a "yes" token vs a "no" token.

This matters because it changes what "serving the reranker" looks like:

Serving Shape A (recommended for Qwen3-Reranker correctness):
- Treat rerank as an LLM-style endpoint (vLLM or TensorRT-LLM).
- The rerank gateway:
  - formats (query, doc) into the expected prompt template
  - runs one forward pass per pair (batched)
  - computes a scalar score from the next-token logits (yes vs no)
- This looks a lot like Milestone 5 (LLM serving), but with a different prompt and output interpretation.

Serving Shape B (recommended for Triton + ONNX/TensorRT "classic" serving):
- Use a "sequence classification" adapter/wrapper for Qwen3-Reranker that produces a direct score tensor.
- This makes ONNX export and Triton configs much more like traditional rerankers.
- Tradeoff: you must validate scoring quality vs the canonical prompting-based reranker behavior.

In this milestone, the ONNX/Triton steps below assume Serving Shape B.
If you choose Serving Shape A, skip the ONNX export parts and instead stand up the reranker as a small LLM
service (then keep the `/rerank` gateway contract the same).

### Step 4.3) Create Directories
```bash
mkdir -p nvidia/models/onnx
mkdir -p nvidia/triton/model_repository/rerank_trt/1
```

### Step 4.4) Export To ONNX And Validate
Why:
- Same as embeddings: ONNX is the bridge into TensorRT.
- Validation is critical; rerankers are sensitive to tokenization and sequence truncation.

How:
You need the same three things as embeddings:
1. The ONNX file.
2. The ONNX input/output tensor names.
3. The ONNX input/output shapes and dtypes (for Triton `config.pbtxt`).

#### Step 4.4.1) Reuse The Export Environment
If you already created `.venv_onnx` in Step 3.4.1, reuse it:
```bash
source .venv_onnx/bin/activate
```

#### Step 4.4.2) Export ONNX (First Success)
Start with a fixed max sequence length to reduce variables:
```bash
export RERANKER_ID="Qwen/Qwen3-Reranker-0.6B"
export SEQ_LEN=256
```

Try Optimum export first (task shape for rerankers is usually "text-classification" / sequence classification):
```bash
optimum-cli export onnx \
  --model "$RERANKER_ID" \
  --task text-classification \
  --framework pt \
  --opset 17 \
  --sequence_length "$SEQ_LEN" \
  nvidia/models/onnx/qwen3_rerank_${SEQ_LEN}
```

Find the ONNX file that was produced:
```bash
find nvidia/models/onnx/qwen3_rerank_${SEQ_LEN} -maxdepth 2 -type f -name '*.onnx' -print
```

Standardize on this filename:
- `nvidia/models/onnx/qwen3_rerank_${SEQ_LEN}/model.onnx`

Copy into place if needed:
```bash
cp "<PATH_FROM_FIND>.onnx" "nvidia/models/onnx/qwen3_rerank_${SEQ_LEN}/model.onnx"
```

If Optimum export fails:
- Fall back to a custom Python export script (similar to the embeddings note in Step 3.4.3 Option B).
- The point is not "Optimum vs script"; the point is producing a valid ONNX with known names/shapes.

#### Step 4.4.3) Inspect Names/Shapes/Dtypes (Required For Triton Config)
Run the same ONNX inspection script as Step 3.4.4, but point it at the reranker:
```bash
export ONNX_PATH="nvidia/models/onnx/qwen3_rerank_${SEQ_LEN}/model.onnx"
python - <<'PY'
import os
import onnx

onnx_path = os.environ["ONNX_PATH"]
m = onnx.load(onnx_path)
print("ONNX:", onnx_path)
print("inputs:")
for i in m.graph.input:
    t = i.type.tensor_type
    shape = [d.dim_value if d.dim_value else -1 for d in t.shape.dim]
    print(" -", i.name, "dtype_enum=", t.elem_type, "shape=", shape)
print("outputs:")
for o in m.graph.output:
    t = o.type.tensor_type
    shape = [d.dim_value if d.dim_value else -1 for d in t.shape.dim]
    print(" -", o.name, "dtype_enum=", t.elem_type, "shape=", shape)
PY
```

#### Step 4.4.4) Correctness Sanity (Must Do)
Before you optimize anything, confirm the model behaves like a reranker:
- Pick a query.
- Provide one obviously relevant doc and one obviously irrelevant doc.
- Confirm the relevant doc scores higher (in PyTorch or ONNX runtime).

### Step 4.4B) (Recommended) First Serve The ONNX In Triton (Correctness Before Speed)
Why:
- Same reasoning as embeddings: validate inputs/outputs and scoring behavior before compiling to TensorRT.

How (detailed, do-not-skip):

#### Step 4.4B.1) Put The ONNX File In The Triton Model Repository
Why:
- Triton expects the ONNX file to be named `model.onnx` in the version directory.

How:
```bash
mkdir -p nvidia/triton/model_repository/rerank_onnx/1
cp nvidia/models/onnx/qwen3_rerank_256/model.onnx nvidia/triton/model_repository/rerank_onnx/1/model.onnx
```

#### Step 4.4B.2) Write `config.pbtxt` That Matches Your ONNX Graph
Why:
- Rerank ONNX exports often differ in output naming and shape (some return `[B, 1]`, some `[B]`).
- Token dtypes are commonly `INT64`. Match the ONNX graph.

How:
1. Inspect ONNX inputs/outputs exactly like in Step 3.4.4 (names/shapes/dtypes).
2. Write the config. Start with FP32 output for correctness.

Template (adapt names/dtypes; typical inputs `[B, S]`, output `[B, 1]`):
```text
name: "rerank_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  { name: "input_ids" data_type: TYPE_INT64 dims: [ -1 ] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }
]

output [
  { name: "<SCORE_OUTPUT_NAME>" data_type: TYPE_FP32 dims: [ 1 ] }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 2000
}

instance_group [{ kind: KIND_GPU count: 1 }]
```

Important:
- If your ONNX output is `[B]`, use `dims: [ ]` or `dims: [ 1 ]` depending on the graph.
  (Triton expects you to match what the backend outputs; verify with the ONNX inspection script.)

#### Step 4.4B.3) Restart Triton And Verify
```bash
docker restart triton
docker logs -n 200 triton
curl -s http://localhost:8200/v2/models/rerank_onnx | jq
```

#### Step 4.4B.4) Prove You Can Infer Against `rerank_onnx` (Before Building A Gateway)
Why:
- Same isolation principle: prove Triton+model config correctness first.

How (Python client request; replace output name):
```bash
python -m pip install 'tritonclient[http]' numpy
python - <<'PY'
import numpy as np
import tritonclient.http as httpclient

TRITON_URL = "127.0.0.1:8200"
MODEL = "rerank_onnx"
OUTPUT = "<SCORE_OUTPUT_NAME>"  # replace

client = httpclient.InferenceServerClient(url=TRITON_URL)

input_ids = np.ones((2, 8), dtype=np.int64)        # 2 pairs
attention_mask = np.ones((2, 8), dtype=np.int64)

inputs = []
i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
i0.set_data_from_numpy(input_ids, binary_data=True)
inputs.append(i0)
i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
i1.set_data_from_numpy(attention_mask, binary_data=True)
inputs.append(i1)

outputs = [httpclient.InferRequestedOutput(OUTPUT, binary_data=True)]
res = client.infer(MODEL, inputs=inputs, outputs=outputs)
out = res.as_numpy(OUTPUT)
print("output shape:", out.shape, "dtype:", out.dtype, "sample:", out.reshape(-1)[:5])
PY
```

Now do the same with real tokenization (this validates pair formatting and sequence length):
```bash
python -m pip install 'tritonclient[http]' numpy transformers
python - <<'PY'
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

TRITON_URL = "127.0.0.1:8200"
MODEL = "rerank_onnx"
OUTPUT = "<SCORE_OUTPUT_NAME>"  # replace
TOKENIZER_ID = "Qwen/Qwen3-Reranker-0.6B"
SEQ_LEN = 256

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, use_fast=True)

query = "what did we decide about using triton?"
docs = [
    "We decided to host models behind Triton and keep gateways stable.",
    "This is a recipe for banana bread.",
]

enc = tok(
    [query] * len(docs),
    docs,
    return_tensors="np",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN,
)

input_ids = enc["input_ids"].astype(np.int64)
attention_mask = enc["attention_mask"].astype(np.int64)

client = httpclient.InferenceServerClient(url=TRITON_URL)
inputs = []
i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
i0.set_data_from_numpy(input_ids, binary_data=True)
inputs.append(i0)
i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
i1.set_data_from_numpy(attention_mask, binary_data=True)
inputs.append(i1)

outputs = [httpclient.InferRequestedOutput(OUTPUT, binary_data=True)]
res = client.infer(MODEL, inputs=inputs, outputs=outputs)
out = res.as_numpy(OUTPUT).astype(np.float32).reshape(-1)
print("scores:", out.tolist())
print("order:", np.argsort(-out).tolist())
PY
```

#### Step 4.4B.5) Only Then: Point Your Rerank Gateway At `rerank_onnx`
Once it works, upgrade:
- Serve `rerank_trt` and switch the gateway's `TRITON_MODEL` from `rerank_onnx` to `rerank_trt`.

### Step 4.5) Build The TensorRT Engine
Why:
- Cross-encoders can be expensive. TensorRT buys you throughput so you can rerank more candidates.

How:
1. Start with fixed max sequence length (256 or 512).
2. Build FP16 first.
3. Save as: `nvidia/triton/model_repository/rerank_trt/1/model.plan`.

### Step 4.6) Write `config.pbtxt` (Enable Dynamic Batching)
Why:
- Triton can batch many rerank pairs together automatically, boosting GPU utilization.

How (template; adapt names):
```text
name: "rerank_trt"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  { name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] },
  { name: "attention_mask" data_type: TYPE_INT32 dims: [ -1 ] }
]

output [
  { name: "<SCORE_OUTPUT_NAME>" data_type: TYPE_FP16 dims: [ 1 ] }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 2000
}

instance_group [{ kind: KIND_GPU count: 1 }]
```

### Step 4.7) Load The Model Into Triton
```bash
docker restart triton
docker logs -n 200 triton
curl -s http://localhost:8200/v2/models/rerank_trt | jq
```

### Step 4.8) Implement The Rerank Gateway
Why:
- The gateway hides tokenization and pair construction from the RAG API.
- It also enforces truncation rules (e.g., max chars per doc) consistently.

Gateway responsibilities:
1. Apply deterministic doc truncation (e.g., `RERANK_MAX_CHARS_PER_DOC`).
2. Tokenize `(query, doc)` pairs.
3. Call Triton.
4. Return `{scores, order}` in a stable JSON format.

RAG wiring (later, when we implement Phase 4 in this repo):
- `RERANK_BASE_URL=http://127.0.0.1:8102`
- Keep `RERANK_MAX_CHARS_PER_DOC` conservative until the RTX 3090 upgrade (start ~2000-4000 chars).

## Milestone 5: TensorRT-LLM (LLM Serving)
Outcome: an OpenAI-compatible chat endpoint on `http://localhost:8103/v1` backed by TensorRT-LLM.

Why this milestone exists:
- The LLM is the final step in the pipeline, but it must be "constrained" by our evidence-pack + citation gating.
- TensorRT-LLM is the NVIDIA-native path for high-performance LLM inference (especially once you have the RTX 3090).
- Separating "LLM serving" from "RAG orchestration" lets you iterate on models and performance without touching retrieval correctness.

### Step 5.1) Decide What The RAG App Needs From The LLM
This repo's Phase 5 uses the LLM as a pure text generator with strict constraints:
- Input: evidence pack + instructions.
- Output: citation-formatted answer.
- Temperature: `0` (or near-zero) to minimize variance and simplify debugging.

So we prioritize:
- reliability + determinism over "creative" sampling
- predictable latency (batching, max tokens)

### Step 5.2) Choose A Model (Pragmatic Guidance)
On RTX 3080 (10GB):
- LLM serving is likely tight for 7B/8B class models unless heavily quantized.

On RTX 3090 (24GB):
- 7B/8B class models become much more realistic.

For TensorRT-LLM learning:
- Start with a well-trodden model family that TensorRT-LLM supports smoothly (often Llama-family).
- Once the serving path is stable, you can try a Qwen-family instruct model if desired.

### Step 5.3) Build A TensorRT-LLM Engine (Record Every Knob)
Why:
- Engine build settings define memory use, max context length, and throughput. They must be tracked for reproducibility.

What to record (minimum):
- model id / weights source
- precision / quantization mode
- max input length / max output length
- max batch size
- KV cache settings (these dominate memory usage)

How (high-level):
1. Use a TensorRT-LLM build environment (typically an NVIDIA container).
2. Convert model weights into TensorRT-LLM format.
3. Build the engine with your chosen constraints.

### Step 5.4) Serve The Engine
Two common deployment shapes:
1. Triton hosts the TensorRT-LLM backend (one serving plane for everything).
2. A dedicated TensorRT-LLM server process hosts the model.

Why you might prefer Triton here:
- unified metrics + batching + deployment mechanics

Why you might prefer a dedicated server at first:
- fewer moving pieces while learning TensorRT-LLM itself

### Step 5.5) Provide An OpenAI-Compatible Gateway
Why:
- Our RAG app is designed to talk to an OpenAI-compatible API for LLM calls.
- If your TensorRT-LLM server is not OpenAI-compatible, a gateway is the adapter layer.

Minimum gateway endpoints:
- `GET /v1/models`
- `POST /v1/chat/completions`

### Step 5.6) Verify The Endpoint
```bash
curl -s http://localhost:8103/v1/models | jq
curl -s http://localhost:8103/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL>","messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0}' | jq
```

RAG wiring (later, when we implement Phase 5 in this repo):
- `LLM_BASE_URL=http://127.0.0.1:8103/v1`
- `LLM_MODEL=<model name>`

## Milestone 6: Profiling (Nsight Systems + Nsight Compute)
Outcome: you can answer, with evidence, "where did the time go?" for each stage:
tokenization, Triton queuing/batching, GPU compute, and DB retrieval.

Why this milestone exists:
- Without profiling, you will guess at optimizations and often optimize the wrong layer.
- On a single-GPU workstation, contention is real (tokenization CPU, GPU kernels, PCIe transfers, DB).
- Nsight gives you ground truth. DCGM/Triton metrics tell you what changed after tuning.

### Step 6.1) Define What You Are Optimizing
Pick one primary goal at a time:
- Throughput goal: embeddings backfill rows/sec, rerank pairs/sec
- Latency goal: `/retrieve` p95 under X seconds, `/answer` p95 under Y seconds

This matters because throughput tuning tends to increase batching, which can increase tail latency.

### Step 6.2) Install Nsight Tools
Why:
- Nsight Systems (`nsys`) gives a timeline view (CPU threads, CUDA kernels, NVTX ranges).
- Nsight Compute (`ncu`) gives kernel-level detail (use only after Systems shows a GPU hotspot).

How (Ubuntu 24.04.3):
- Install CUDA toolkit components that include Nsight, or install Nsight packages from NVIDIA.
- Validate availability:
```bash
which nsys || true
which ncu || true
nsys --version || true
ncu --version || true
```

### Step 6.3) Add NVTX Ranges In Gateways (Highly Recommended)
Why:
- NVTX ranges make traces readable: "tokenize", "triton_infer", "pool", "json_encode".

How:
- In Python gateways, add NVTX ranges around major steps (tokenize/infer/postprocess).
- Later, you can also add timing logs in the RAG API for each retrieval lane.

### Step 6.4) Profile Each Service In Isolation First
Why:
- You want to isolate problems before doing end-to-end profiling.

How:
1. Embeddings gateway: profile a single batch request (e.g., 32 short texts).
2. Rerank gateway: profile one query with ~80 docs.
3. LLM gateway: profile one short completion (low max tokens).

### Step 6.5) Profile End-To-End
Why:
- The real user experience is the full chain: DB + embedding (query) + pgvector + rerank + response.

How:
- Profile the RAG API process while issuing a `/retrieve` request that triggers dense + rerank.
- Save traces under `nvidia/profiles/` and keep notes on what changed between runs.

Example (shape only; adjust to your actual commands):
```bash
mkdir -p nvidia/profiles
nsys profile -o nvidia/profiles/retrieve \
  --trace=cuda,nvtx,osrt \
  bash -lc 'for i in $(seq 1 10); do curl -s http://localhost:8001/diagnostics >/dev/null; done'
```

## Milestone 7: Telemetry (DCGM + NVML)
Outcome: you have continuous visibility into GPU health and utilization, and you can correlate
performance changes with GPU metrics.

Why this milestone exists:
- It prevents "silent regressions" (a change that increases GPU memory and causes OOM under load).
- It helps you tune batching: you can see whether the GPU is actually utilized or waiting on CPU/tokenization.

### Step 7.1) DCGM Exporter (Prometheus Metrics)
Why:
- DCGM exporter is the standard NVIDIA way to expose GPU metrics for scraping (utilization, memory, temps, power).

How:
```bash
docker pull nvidia/dcgm-exporter:latest
docker inspect --format='{{index .RepoDigests 0}}' nvidia/dcgm-exporter:latest
docker run --rm --gpus all -p 9400:9400 nvidia/dcgm-exporter:latest
```

Verify:
```bash
curl -s http://localhost:9400/metrics | head
```

### Step 7.2) (Optional) Add Prometheus + Grafana
Why:
- Dashboards make it much easier to see trends and correlate spikes with requests.

How:
- Once DCGM exporter is working, add Prometheus+Grafana via Docker Compose and scrape `:9400/metrics`.
- Keep this optional until you have stable services.

### Step 7.3) NVML (On-Demand Snapshot)
Why:
- NVML is perfect for quick "is the GPU healthy?" checks and for integrating a lightweight snapshot into `GET /diagnostics`.

Host-level snapshot:
```bash
nvidia-smi
```

Python snapshot:
```bash
python -m pip install nvidia-ml-py3
python - <<'PY'
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
mem = nvmlDeviceGetMemoryInfo(h)
print({'used_mb': mem.used/1024/1024, 'total_mb': mem.total/1024/1024})
PY
```

### Step 7.4) (Optional Repo Change) Add GPU Snapshot To `GET /diagnostics`
Why:
- When debugging user reports, one API call should tell you DB version, extension versions, and GPU status.

How:
- Add an optional NVML read in `app/db.py` (or a new module) and include it in diagnostics output.

## Milestone 8: Wire This Repo To NVIDIA Endpoints
Outcome: the RAG API uses GPU services deterministically (no tool loops).

1. Implement Phase 3 in this repo (dense embeddings lane + HTTP embedding client).
2. Set env vars (in a local `.env`, not committed):
- `EMBEDDINGS_BASE_URL=http://127.0.0.1:8101`
- `EMBEDDINGS_MODEL_ID=<your embedding model>`
- `EMBEDDINGS_DIM=1024`
3. Implement Phase 4 in this repo (HTTP rerank client) and set:
- `RERANK_BASE_URL=http://127.0.0.1:8102`
- `RERANK_MODEL_ID=<your reranker model>`
4. Implement Phase 5 in this repo (`/answer` + citation gating) and set:
- `LLM_BASE_URL=http://127.0.0.1:8103/v1`
- `LLM_MODEL=<your llm model>`

Optional NVIDIA-centric repo additions (when ready):
- `docker-compose.nvidia.yml` for Triton/gateways/dcgm-exporter
- GPU stats in `GET /diagnostics` via NVML

## Optional NVIDIA Add-Ons That Fit This Project
- NVIDIA Riva: ASR + diarization to generate higher-quality `json_turns` transcripts.
- NVIDIA NeMo: train/fine-tune domain models (entities, rerankers, small task models).
- NVIDIA GPU Operator (Kubernetes): only if you decide to run this stack on K8s.
- TensorRT Model Optimizer / Polygraphy: validation + conversion tooling around TensorRT.

## Suggested Learning Order (Minimize Thrash)
1. Milestone 0, then Milestone 1.
2. Milestone 2: get Triton stable first (ports, model repo, metrics).
3. Milestone 3: embeddings (get correctness + 1024-d invariants stable; then optimize).
4. Milestone 4: rerank (batching + truncation rules; then optimize).
5. Milestone 5: LLM (strongly consider waiting for the RTX 3090 for a smoother TensorRT-LLM experience).
6. After the RTX 3090 upgrade: re-tune batch sizes, max sequence lengths, and rebuild engines where needed.
7. Add profiling (Milestone 6) and telemetry (Milestone 7) once endpoints are stable.

## Appendix: TensorRT + Triton Command Templates (Copy/Adapt)
These are intentionally generic templates. You will adapt tensor names, shapes, and model IDs.

### A1) Inspect ONNX Inputs/Outputs
```bash
python - <<'PY'
import onnx
m = onnx.load('nvidia/models/onnx/model.onnx')
print('inputs:', [i.name for i in m.graph.input])
print('outputs:', [o.name for o in m.graph.output])
PY
```

### A2) Build A TensorRT Engine With `trtexec` (Example: BERT-Like)
1. Start with fixed shapes (easiest to get working).
2. Then move to dynamic shapes once the pipeline works.

Template (dynamic batch, fixed seq_len=128):
```bash
trtexec \
  --onnx=nvidia/models/onnx/model.onnx \
  --saveEngine=nvidia/triton/model_repository/embed_trt/1/model.plan \
  --fp16 \
  --minShapes=input_ids:1x128,attention_mask:1x128 \
  --optShapes=input_ids:4x128,attention_mask:4x128 \
  --maxShapes=input_ids:8x128,attention_mask:8x128
```

### A3) Verify Triton Sees Your Model
```bash
curl -s http://localhost:8200/v2/models
```

### A4) Call Triton From Python (HTTP Client)
```bash
python -m pip install 'tritonclient[http]' numpy
```

At a minimum, you will:
- tokenize text into `input_ids`/`attention_mask`
- send tensors to `POST /v2/models/<name>/infer`
- pool output to a single 1024-d vector

### A5) Benchmark Throughput/Latency
Use Triton's `perf_analyzer` (usually shipped in a Triton SDK container) to measure p50/p95.

### A6) Minimal Gateway Skeletons (FastAPI + Triton HTTP Client)
These are copy/adapt templates. The key idea is: gateways own tokenization + pooling + truncation;
Triton owns batching + GPU compute.

Embeddings gateway (shape only; you must set correct output tensor name):
```python
import os

import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

TRITON_URL = os.getenv("TRITON_URL", "127.0.0.1:8200")  # host:port (no scheme)
TRITON_MODEL = os.getenv("TRITON_MODEL", "embed_trt")   # or embed_onnx for first bring-up
TRITON_OUTPUT = os.getenv("TRITON_OUTPUT", "<MODEL_OUTPUT_NAME>")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "Qwen/Qwen3-Embedding-0.6B")
MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "128"))
L2_NORM = os.getenv("EMBED_L2_NORMALIZE", "1") == "1"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, use_fast=True)
# Qwen3 embedding stacks commonly expect left padding + EOS as pad token for last-token pooling.
tokenizer.padding_side = "left"
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
client = httpclient.InferenceServerClient(url=TRITON_URL)

app = FastAPI()


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None


def _last_token_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # last_hidden: [B, S, H], attention_mask: [B, S]
    #
    # If you use left padding (recommended here), the last token is always index S-1.
    # If you use right padding, the last non-pad token is (sum(mask) - 1).
    if (attention_mask[:, -1] == 1).all():
        return last_hidden[:, -1, :].astype(np.float32)
    lengths = attention_mask.astype(np.int64).sum(axis=1) - 1
    lengths = np.clip(lengths, 0, last_hidden.shape[1] - 1)
    return last_hidden[np.arange(last_hidden.shape[0]), lengths, :].astype(np.float32)


@app.post("/embed")
def embed(req: EmbedRequest):
    enc = tokenizer(
        req.texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    # Match your Triton model config: many ONNX exports use INT64 for token IDs.
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    inputs = []
    i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
    i0.set_data_from_numpy(input_ids, binary_data=True)
    inputs.append(i0)
    i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    i1.set_data_from_numpy(attention_mask, binary_data=True)
    inputs.append(i1)

    outputs = [httpclient.InferRequestedOutput(TRITON_OUTPUT, binary_data=True)]
    result = client.infer(TRITON_MODEL, inputs=inputs, outputs=outputs)
    out = result.as_numpy(TRITON_OUTPUT)

    if out.ndim == 3:
        emb = _last_token_pool(out, attention_mask)
    elif out.ndim == 2:
        emb = out.astype(np.float32)
    else:
        raise ValueError(f"Unexpected Triton output shape: {out.shape}")

    if emb.shape[1] != 1024:
        raise ValueError(f"Expected 1024-d embeddings, got {emb.shape[1]}")

    if L2_NORM:
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(n, 1e-12, None)

    return {"embeddings": emb.tolist(), "model": req.model or TRITON_MODEL}
```

Rerank gateway (shape only; adapt tensor/output names):
```python
import os

import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

# IMPORTANT:
# - This skeleton assumes you have a reranker model that produces a direct score tensor (seq-cls style).
# - If you are using the canonical Qwen3-Reranker "yes/no logits" approach, serve it like an LLM endpoint
#   (vLLM / TensorRT-LLM) and compute scores from next-token logits in the gateway instead.

TRITON_URL = os.getenv("TRITON_URL", "127.0.0.1:8200")
TRITON_MODEL = os.getenv("TRITON_MODEL", "rerank_trt")  # or rerank_onnx for bring-up
TRITON_OUTPUT = os.getenv("TRITON_OUTPUT", "<SCORE_OUTPUT_NAME>")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", "Qwen/Qwen3-Reranker-0.6B")
MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "256"))
MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS_PER_DOC", "2000"))

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True, use_fast=True)
client = httpclient.InferenceServerClient(url=TRITON_URL)

app = FastAPI()


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None


@app.post("/rerank")
def rerank(req: RerankRequest):
    docs = [(d[:MAX_CHARS] if d else "") for d in req.documents]
    queries = [req.query] * len(docs)

    enc = tokenizer(
        queries,
        docs,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    # Match your Triton model config: many ONNX exports use INT64 for token IDs.
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    inputs = []
    i0 = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
    i0.set_data_from_numpy(input_ids, binary_data=True)
    inputs.append(i0)
    i1 = httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    i1.set_data_from_numpy(attention_mask, binary_data=True)
    inputs.append(i1)

    outputs = [httpclient.InferRequestedOutput(TRITON_OUTPUT, binary_data=True)]
    result = client.infer(TRITON_MODEL, inputs=inputs, outputs=outputs)
    scores = result.as_numpy(TRITON_OUTPUT).astype(np.float32).reshape(-1)

    order = np.argsort(-scores).tolist()  # descending
    return {"scores": scores.tolist(), "order": order, "model": req.model or TRITON_MODEL}
```
