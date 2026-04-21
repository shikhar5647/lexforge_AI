"""
LexForge SFT — Modal backend for the Vercel-hosted web UI.

Exposes:
  * ``train``   — GPU remote function (QLoRA + SFT + merge + push to HF Hub)
  * ``api``     — ASGI (FastAPI) endpoint with CORS. The Next.js frontend on
                  Vercel hits this URL to submit jobs and poll for live logs.

Live logs / status for each submitted job are stored in a persistent
``modal.Dict`` (``lexforge-jobs``) that the ``api`` container reads on every
``GET /api/jobs/{job_id}`` tick.

Local CLI is still supported:

    modal run --detach sft_modal.py                      # default Qwen2.5-3B run
    modal run sft_modal.py::train --model-id ...         # override a single run

Deploy the web API:

    modal deploy sft_modal.py                            # gives you the api URL
"""

import os
import re
import json
import uuid
import secrets
import modal
from pathlib import Path as _Path
from typing import Any, Optional
from datetime import datetime


app = modal.App("lexforge-sft-qlora")

# Safety-audit source dir (../Safety_audit relative to this file). Mounted into
# the GPU image so the audit function can import harmful_score / asr modules.
_SAFETY_AUDIT_DIR = str((_Path(__file__).resolve().parent.parent / "Safety_audit"))

# --- concurrency & policy knobs ---------------------------------------------
# Modal GPU quota on this account. Every training job grabs one GPU.
MAX_CONCURRENT_GPU_JOBS = 10
# How many distinct users (client_ids) may have at least one active job.
MAX_CONCURRENT_USERS = 5
# How many queued/running jobs a single user may hold at once.
MAX_JOBS_PER_USER = 2
# Statuses that count as "active" for the caps above.
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
# Any "active" entry whose last heartbeat is older than this gets auto-reaped
# to "failed" — this handles cases where a GPU container was SIGKILL'd (modal
# app stop, node eviction, OOM) and never got to write back its final status.
STALE_HEARTBEAT_SECONDS = 30 * 60

PKG_LIST = [
    "abnf==2.2.0","accelerate==1.12.0","aiohappyeyeballs==2.6.1","aiohttp==3.13.3",
    "aiosignal==1.4.0","annotated-doc==0.0.4","annotated-types==0.7.0",
    "anthropic==0.71.0","anyio==4.12.1","apache-tvm-ffi==0.1.7","astor==0.8.1",
    "attrs==25.4.0","backoff==2.2.1","bitsandbytes==0.47.0","blake3==1.0.8",
    "cachetools==6.2.4","cbor2==5.8.0","certifi==2026.1.4","cffi==2.0.0",
    "chardet==5.2.0","charset-normalizer==3.4.4","cint==1.0.0","click==8.3.1",
    "cloudpickle==3.1.2","compressed-tensors==0.12.2","contourpy==1.3.3",
    "cryptography==46.0.3","cuda-bindings==13.1.1","cuda-pathfinder==1.3.3",
    "cuda-python==13.1.1","cupy-cuda12x==13.6.0","cut-cross-entropy==25.1.1",
    "cycler==0.12.1","datasets==4.3.0","depyf==0.20.0","diffusers==0.36.0",
    "dill==0.3.8","diskcache==5.6.3","distro==1.9.0","dnspython==2.8.0",
    "docstring-parser==0.17.0","einops==0.8.1","email-validator==2.3.0",
    "fastapi==0.128.0","fastapi-cli==0.0.20","fastapi-cloud-cli==0.9.0",
    "fastar==0.8.0","fastrlock==0.8.3","fickling==0.1.7","filelock==3.20.3",
    "flashinfer-python==0.5.3","fonttools==4.61.1","frozenlist==1.8.0",
    "fsspec==2024.2.0","gguf==0.17.1","gitdb==4.0.12","gitpython==3.1.46",
    "gql==4.0.0","graphql-core==3.2.7","graphviz==0.21","grpclib==0.4.9",
    "h11==0.16.0","h2==4.3.0","hf-transfer==0.1.9","hf-xet==1.2.0","hjson==3.1.0",
    "hpack==4.1.0","httpcore==1.0.9","httptools==0.7.1","httpx==0.28.1",
    "httpx-sse==0.4.3","huggingface-hub==0.36.0","hyperframe==6.1.0",
    "idna==3.11","ijson==3.4.0.post0","importlib-metadata==8.7.1",
    "interegular==0.3.3","intervaltree==3.2.1","jinja2==3.1.6","jiter==0.12.0",
    "jmespath==1.0.1","jsonschema==4.26.0","jsonschema-specifications==2025.9.1",
    "kaitaistruct==0.11","kiwisolver==1.4.9","lark==1.2.2","llguidance==1.3.0",
    "llvmlite==0.44.0","lm-format-enforcer==0.11.3","loguru==0.7.3",
    "markdown-it-py==4.0.0","markupsafe==3.0.3","matplotlib==3.10.8","mcp==1.25.0",
    "mdurl==0.1.2","mistral-common==1.8.8","modal==1.3.0.post1",
    "model-hosting-container-standards==0.1.13","mpmath==1.3.0","msgpack==1.1.2",
    "msgspec==0.20.0","multidict==6.7.0","multiprocess==0.70.16","networkx==3.6.1",
    "ninja==1.13.0","numba==0.61.2","numpy==1.26.4","nvidia-cublas-cu12==12.8.4.1",
    "nvidia-cuda-cupti-cu12==12.8.90","nvidia-cuda-nvrtc-cu12==12.8.93",
    "nvidia-cuda-runtime-cu12==12.8.90","nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cudnn-frontend==1.17.0","nvidia-cufft-cu12==11.3.3.83",
    "nvidia-cufile-cu12==1.13.1.3","nvidia-curand-cu12==10.3.9.90",
    "nvidia-cusolver-cu12==11.7.3.90","nvidia-cusparse-cu12==12.5.8.93",
    "nvidia-cusparselt-cu12==0.7.1","nvidia-cutlass-dsl==4.3.5","nvidia-ml-py==13.590.44",
    "nvidia-nccl-cu12==2.27.5","nvidia-nvjitlink-cu12==12.8.93",
    "nvidia-nvshmem-cu12==3.3.20","nvidia-nvtx-cu12==12.8.90","openai==2.15.0",
    "openai-harmony==0.0.8","opencv-python-headless==4.11.0.86","outlines-core==0.2.11",
    "packaging==25.0","pandas==2.3.3","partial-json-parser==0.2.1.1.post7",
    "pdfminer-six==20260107","peft==0.18.1","pillow==12.1.0","platformdirs==4.5.1",
    "polyfile-weave==0.5.8","prometheus-client==0.23.1","prometheus-fastapi-instrumentator==7.1.0",
    "propcache==0.4.1","protobuf==6.33.4","psutil==7.2.1","py-cpuinfo==9.0.0",
    "pyarrow==22.0.0","pyarrow-hotfix==0.7","pybase64==1.4.3","pycountry==24.6.1",
    "pycparser==2.23","pydantic==2.12.5","pydantic-core==2.41.5","pydantic-extra-types==2.11.0",
    "pydantic-settings==2.12.0","pygments==2.19.2","pyjwt==2.10.1","pynvml==13.0.1",
    "pyparsing==3.3.1","python-dateutil==2.9.0.post0","python-dotenv==1.2.1",
    "python-json-logger==4.0.0","python-multipart==0.0.21","pytz==2025.2","pyyaml==6.0.3",
    "pyzmq==27.1.0","ray==2.53.0","referencing==0.37.0","regex==2025.11.3","requests==2.32.5",
    "rich==14.2.0","rich-toolkit==0.17.1","rignore==0.7.6","rpds-py==0.30.0",
    "safetensors==0.7.0","scipy==1.17.0","seaborn==0.13.2","sentencepiece==0.2.1",
    "sentry-sdk==2.49.0","setproctitle==1.3.7","setuptools==80.9.0","shellingham==1.5.4",
    "six==1.17.0","smmap==5.0.2","sniffio==1.3.1","sortedcontainers==2.4.0",
    "sse-starlette==3.1.2","starlette==0.50.0","stdlib-list==0.11.1","supervisor==4.3.0",
    "sympy==1.14.0","synchronicity==0.11.1","tabulate==0.9.0","tenacity==9.1.2",
    "tiktoken==0.12.0","tokenizers==0.22.2","toml==0.10.2","torch>=2.9.0","torchao==0.15.0",
    "torchaudio==2.9.0","torchvision==0.24.0","tqdm==4.67.1","transformers==4.57.3",
    "triton==3.5.0","trl==0.24.0","typeguard==4.4.4","typer==0.21.1","types-certifi==2021.10.8.3",
    "types-toml==0.10.8.20240310","typing-extensions==4.15.0","typing-inspection==0.4.2",
    "tyro==1.0.3","tzdata==2025.3","unsloth==2026.1.2","unsloth-zoo==2026.1.2",
    "urllib3==2.6.3","uvicorn==0.40.0","uvloop==0.22.1","vllm==0.13.0",
    "wandb>=0.18.0","watchfiles==1.1.1","weave==0.52.24","websockets==16.0","wheel==0.45.1",
    "xgrammar==0.1.27","xxhash==3.6.0","yarl==1.22.0","zipp==3.23.0"
]

image = (
    modal.Image.debian_slim()
    .uv_pip_install(*PKG_LIST)
    .add_local_python_source("finetuning_buckets")
    .add_local_dir(_SAFETY_AUDIT_DIR, remote_path="/root/Safety_audit")
)

# Lightweight image for the public FastAPI endpoint: no GPU deps needed.
web_image = modal.Image.debian_slim().pip_install(
    "fastapi==0.128.0", "pydantic==2.12.5", "modal==1.3.0.post1"
)


gpu_obj = "H200"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "theatticusproject/cuad-qa"
OUTPUT_DIR_DEFAULT = "/checkpoints/lexforge-qwen25-cuad"

# Exposed to the frontend via /api/options.
MODEL_OPTIONS = {
    "Qwen/Qwen2.5-3B-Instruct":        "Qwen 2.5 3B Instruct",
    "microsoft/Phi-3.5-mini-instruct": "Phi 3.5 Mini Instruct",
    "meta-llama/Llama-3.2-3B-Instruct":"Llama 3.2 3B Instruct",
}
DATASET_OPTIONS = {
    "theatticusproject/cuad-qa":  "CUAD-QA (full, 22k)",
    "alex-apostolo/filtered-cuad":"Filtered CUAD (curated)",
}

# ---- Safety-audit defaults --------------------------------------------------
AUDIT_DATASET = "gjyotin305/harmfulhexphi_300"
AUDIT_DATASET_MAX = 300
GUARD_MODEL_DEFAULT = "meta-llama/Llama-Guard-3-8B"
ASR_JUDGE_DEFAULT = "gpt-4.1-mini"
AUDIT_OUTPUT_BASE = "/checkpoints/lexforge-audits"
HF_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")

# Persistent job state — shared between the GPU trainer and the web endpoint.
JOB_STATUS = modal.Dict.from_name("lexforge-jobs", create_if_missing=True)


SYSTEM_PROMPT = (
    "You are LexForge AI, a legal-domain assistant specialized in contract analysis. "
    "Given a contract excerpt and a question about a specific clause or provision, "
    "answer precisely using only information from the excerpt. "
    "If the contract does not contain the information, respond exactly: "
    "'Not specified in the provided contract.'"
)

NO_ANSWER = "Not specified in the provided contract."


def build_formatter(tokenizer: Any):
    def format_example(example: dict) -> dict:
        question = example["question"]
        context = example["context"]
        answers_field = example.get("answers", {}) or {}
        spans = answers_field.get("text", []) if isinstance(answers_field, dict) else []
        answer = spans[0].strip() if spans and spans[0].strip() else NO_ANSWER

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contract excerpt:\n{context}\n\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return format_example


def _load_dataset_safely(dataset_id: str):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_id)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        print(f"[sft_modal] '{dataset_id}' is script-based; retrying refs/convert/parquet.")
        return load_dataset(dataset_id, revision="refs/convert/parquet")


def load_and_format(tokenizer: Any, dataset_id: str, *, eval_samples: int, seed: int):
    raw = _load_dataset_safely(dataset_id)
    fmt = build_formatter(tokenizer)

    train_cols = raw["train"].column_names
    train = raw["train"].map(fmt, remove_columns=train_cols, desc="Formatting train")

    eval_split = "test" if "test" in raw else ("validation" if "validation" in raw else None)
    if eval_split is None:
        return train, None

    eval_cols = raw[eval_split].column_names
    evalds = raw[eval_split].map(fmt, remove_columns=eval_cols, desc="Formatting eval")
    if eval_samples > 0:
        evalds = evalds.shuffle(seed=seed).select(range(min(eval_samples, len(evalds))))
    return train, evalds


# ---- Job-status helpers (shared dict) ---------------------------------------

MAX_LOG_LINES = 2000


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _append_log(job_id: str, line: str) -> None:
    if not job_id:
        return
    try:
        entry = JOB_STATUS.get(job_id) or {}
    except Exception:
        entry = {}
    logs = entry.get("logs", [])
    logs.append(f"[{_now_iso()}] {line}")
    if len(logs) > MAX_LOG_LINES:
        logs = logs[-MAX_LOG_LINES:]
    entry["logs"] = logs
    entry["updated_at"] = _now_iso()
    JOB_STATUS[job_id] = entry
    print(line)


def _set_status(job_id: str, **updates: Any) -> None:
    if not job_id:
        return
    try:
        entry = JOB_STATUS.get(job_id) or {}
    except Exception:
        entry = {}
    entry.update(updates)
    entry["updated_at"] = _now_iso()
    JOB_STATUS[job_id] = entry


# ---- Naming & identity helpers ---------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 24) -> str:
    """Produce a lowercase slug safe for filesystem paths."""
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    return (s or "x")[:max_len].strip("-") or "x"


def _build_run_name(model_id: str, dataset_id: str, job_id: str) -> str:
    """Human-readable run name used for both the checkpoint dir and W&B-style run label."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_slug = _slugify(model_id.split("/")[-1])
    data_slug = _slugify(dataset_id.split("/")[-1])
    return f"{ts}-{model_slug}-{data_slug}-{job_id}"


def _iter_all_jobs() -> list:
    """Snapshot current JOB_STATUS entries as a list. Auto-reaps stale "active"
    entries (no heartbeat within STALE_HEARTBEAT_SECONDS) to "failed" so
    containers that died silently don't hold concurrency slots forever.
    Tolerates transient dict errors."""
    try:
        items = list(JOB_STATUS.items())
    except Exception:
        return []

    out: list = []
    now = datetime.utcnow()
    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in ACTIVE_STATUSES:
            raw = (entry.get("updated_at") or "").rstrip("Z")
            try:
                t = datetime.fromisoformat(raw)
                age = (now - t).total_seconds()
            except Exception:
                age = STALE_HEARTBEAT_SECONDS + 1
            if age > STALE_HEARTBEAT_SECONDS:
                entry["status"] = "failed"
                entry["stage"] = "stale"
                entry["error"] = f"no heartbeat for {age/60:.1f} min"
                entry["finished_at"] = now.isoformat(timespec="seconds") + "Z"
                try:
                    JOB_STATUS[key] = entry
                except Exception:
                    pass
        out.append(entry)
    return out


def _active_jobs_for_user(client_id: str) -> list:
    return [
        j for j in _iter_all_jobs()
        if j.get("client_id") == client_id and j.get("status") in ACTIVE_STATUSES
    ]


def _active_user_ids() -> set:
    return {
        j.get("client_id")
        for j in _iter_all_jobs()
        if j.get("client_id") and j.get("status") in ACTIVE_STATUSES
    }


# ---- The trainer ------------------------------------------------------------


@app.function(
    image=image,
    gpu=gpu_obj,
    timeout=28800,
    max_containers=MAX_CONCURRENT_GPU_JOBS,
    volumes={"/checkpoints": modal.Volume.from_name("kl_ablation_checkpoints")},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    job_id: str = "",
    model_id: str = MODEL_ID,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    dataset_id: str = DATASET_ID,
    max_seq_length: int = 2048,
    epochs: float = 1.0,
    per_device_bs: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    logging_steps: int = 25,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    eval_samples: int = 2000,
    seed: int = 42,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    use_flash_attn: bool = True,
    no_eval: bool = False,
    hf_repo_id: Optional[str] = None,
    push_to_hub: bool = True,
    private_repo: bool = False,
    run_name: str = "lexforge-sft",
) -> dict:
    import gc
    import traceback
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    os.makedirs(output_dir, exist_ok=True)

    try:
        _set_status(
            job_id,
            status="running",
            stage="setup",
            model_id=model_id,
            dataset_id=dataset_id,
            hf_repo_id=hf_repo_id,
            output_dir=output_dir,
            started_at=_now_iso(),
        )
        _append_log(job_id, f"Training started: {model_id} on {dataset_id}")

        use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        use_fp16 = (not use_bf16) and torch.cuda.is_available()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        def _flash_attn_available() -> bool:
            try:
                import flash_attn  # noqa: F401
                return True
            except Exception:
                return False

        if (not use_flash_attn) or (not use_bf16) or (not _flash_attn_available()):
            attn_impl = "sdpa"
            if use_bf16 and use_flash_attn:
                _append_log(job_id, "flash_attn not installed → falling back to sdpa.")
        else:
            attn_impl = "flash_attention_2"
        _append_log(job_id, f"compute_dtype={compute_dtype}  attn_impl={attn_impl}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        torch.cuda.empty_cache()

        _set_status(job_id, stage="loading_model")
        _append_log(job_id, "Loading base model...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            attn_implementation=attn_impl,
        )
        model.config.use_cache = False
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _append_log(job_id, f"LoRA: trainable={trainable:,}  total={total:,}  pct={100*trainable/total:.2f}%")

        _set_status(job_id, stage="loading_data")
        _append_log(job_id, f"Loading dataset {dataset_id}...")
        train_dataset, eval_dataset = load_and_format(
            tokenizer,
            dataset_id,
            eval_samples=0 if no_eval else eval_samples,
            seed=seed,
        )
        if no_eval:
            eval_dataset = None
        _append_log(
            job_id,
            f"train={len(train_dataset)}  eval={len(eval_dataset) if eval_dataset else 0}",
        )

        sft_max_len_kwargs: dict = {}
        try:
            SFTConfig(output_dir=output_dir, max_length=max_seq_length)  # type: ignore[arg-type]
            sft_max_len_kwargs["max_length"] = max_seq_length
        except TypeError:
            sft_max_len_kwargs["max_seq_length"] = max_seq_length

        sft_config = SFTConfig(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=1.0,
            optim="paged_adamw_8bit" if bnb_config is not None else "adamw_torch",
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=use_bf16,
            logging_strategy="steps",
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=eval_steps,
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False,
            report_to="none",
            run_name=run_name,
            **sft_max_len_kwargs,
            packing=False,
            dataset_text_field="text",
            seed=seed,
            dataloader_num_workers=4,
            remove_unused_columns=True,
        )

        class _LiveLogger(TrainerCallback):
            def on_train_begin(self, args, state, control, **kw):
                _set_status(job_id, stage="training", max_steps=int(state.max_steps or 0))
                _append_log(job_id, f"training begin (max_steps={state.max_steps})")

            def on_log(self, args, state, control, logs=None, **kw):
                if not logs:
                    return
                slim = {k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))}
                slim["step"] = int(state.global_step)
                _append_log(job_id, "metrics " + json.dumps(slim, default=str))
                _set_status(
                    job_id,
                    step=int(state.global_step),
                    epoch=float(state.epoch or 0.0),
                    latest_metrics=slim,
                )

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                if metrics:
                    _append_log(job_id, "eval " + json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, default=str))

            def on_train_end(self, args, state, control, **kw):
                _append_log(job_id, "training end")

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            callbacks=[_LiveLogger()],
        )

        trainer.train()
        adapter_dir = os.path.join(output_dir, "adapter")
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        _append_log(job_id, f"adapter saved → {adapter_dir}")

        metrics: dict = {}
        if eval_dataset is not None:
            metrics = trainer.evaluate()
            trainer.log_metrics("final_eval", metrics)
            trainer.save_metrics("final_eval", metrics)

        result: dict = {
            "job_id": job_id,
            "model_id": model_id,
            "adapter_path": adapter_dir,
            "eval_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }

        if hf_repo_id:
            _set_status(job_id, stage="merging")
            _append_log(job_id, "Freeing trainer / model to reload in fp16 for merge...")

            del trainer
            del model
            gc.collect()
            torch.cuda.empty_cache()

            _append_log(job_id, f"Reloading base model in {compute_dtype} and applying adapter...")
            base = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            merged = PeftModel.from_pretrained(base, adapter_dir)
            merged = merged.merge_and_unload()
            merged_dir = os.path.join(output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            _append_log(job_id, f"merged model saved → {merged_dir}")

            if push_to_hub:
                _set_status(job_id, stage="pushing")
                from huggingface_hub import HfApi, create_repo
                from huggingface_hub.utils import RepositoryNotFoundError

                hf_token = (
                    os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                )
                if not hf_token:
                    raise RuntimeError(
                        "No HF token found. Set HF_TOKEN in the Modal 'huggingface-secret'."
                    )
                api_client = HfApi(token=hf_token)
                repo_existed_before = True
                try:
                    api_client.repo_info(hf_repo_id)
                except RepositoryNotFoundError:
                    repo_existed_before = False
                except Exception:
                    repo_existed_before = True
                _append_log(job_id, f"creating HF repo {hf_repo_id} (private={private_repo})...")
                create_repo(hf_repo_id, token=hf_token, exist_ok=True, private=private_repo)
                _set_status(
                    job_id,
                    hf_repo_created_this_run=(not repo_existed_before),
                    hf_push_complete=False,
                )
                _append_log(job_id, f"uploading merged weights to {hf_repo_id}...")
                api_client.upload_folder(
                    folder_path=merged_dir,
                    repo_id=hf_repo_id,
                    commit_message=f"LexForge SFT {run_name}",
                )
                _set_status(job_id, hf_push_complete=True)
                _append_log(
                    job_id,
                    f"pushed → https://huggingface.co/{hf_repo_id}",
                )
                result["hf_url"] = f"https://huggingface.co/{hf_repo_id}"

        summary_path = os.path.join(output_dir, "sft_summary.json")
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        _append_log(job_id, f"summary → {summary_path}")

        _set_status(
            job_id,
            status="completed",
            stage="done",
            finished_at=_now_iso(),
            result=result,
        )
        return result

    except Exception as e:
        tb = traceback.format_exc()
        _append_log(job_id, f"ERROR {type(e).__name__}: {e}")
        _append_log(job_id, tb)
        _set_status(job_id, status="failed", stage="error", error=str(e))
        raise


# ---- Safety audit (vLLM generate → Llama Guard | GPT rubric) ----------------


@app.function(
    image=image,
    gpu=gpu_obj,
    timeout=7200,
    max_containers=MAX_CONCURRENT_GPU_JOBS,
    volumes={"/checkpoints": modal.Volume.from_name("kl_ablation_checkpoints")},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def audit(
    job_id: str,
    hf_model_id: str,
    audit_type: str,
    output_dir: str,
    run_name: str,
    max_samples: int = 100,
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 32,
    guard_model: str = GUARD_MODEL_DEFAULT,
    asr_judge_model: str = ASR_JUDGE_DEFAULT,
    openai_api_key: Optional[str] = None,
) -> dict:
    """Two-phase safety audit:
      A) vLLM generates responses from `hf_model_id` on `AUDIT_DATASET`.
      B) Score those (user, response) pairs with Llama Guard ("harmful") or
         GPT rubric ("asr"). The OpenAI key is ephemeral — never stored."""
    import gc
    import sys
    import traceback

    # Make Safety_audit importable inside the container.
    if "/root/Safety_audit" not in sys.path:
        sys.path.insert(0, "/root/Safety_audit")

    try:
        os.makedirs(output_dir, exist_ok=True)
        _set_status(
            job_id,
            status="running",
            stage="setup",
            started_at=_now_iso(),
        )
        _append_log(job_id, f"audit start · model={hf_model_id} · type={audit_type}")

        # ---- Phase A: prepare prompts from the fixed audit dataset ---------
        _set_status(job_id, stage="loading_dataset")
        _append_log(job_id, f"loading {AUDIT_DATASET} (max_samples={max_samples})")
        from datasets import load_dataset

        ds = load_dataset(AUDIT_DATASET, split="train")
        n = min(int(max_samples), len(ds))
        ds = ds.select(range(n))
        users: list[str] = [str(ds[i]["user"]) for i in range(n)]
        meta: list[dict] = []
        for i in range(n):
            row: dict = {"idx": i}
            for k in ("from", "score_clean", "category", "policy"):
                if k in ds.column_names:
                    row[k] = ds[i].get(k)
            meta.append(row)
        _append_log(job_id, f"loaded {n} prompts")

        # Build chat-formatted prompts with the target model's tokenizer.
        _set_status(job_id, stage="building_prompts")
        from hexphi_generate_audit import _build_prompts, write_generations_jsonl

        prompts = _build_prompts(users, hf_model_id)
        _append_log(job_id, "prompts built via target model's chat template")

        # ---- Phase A: vLLM generation, with per-batch progress -------------
        _set_status(job_id, stage="generating", step=0, max_steps=n)
        _append_log(job_id, f"initializing vLLM engine for {hf_model_id} (1–3 min)")
        from vllm import LLM, SamplingParams

        llm = LLM(
            hf_model_id,
            trust_remote_code=True,
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
            dtype=os.getenv("VLLM_DTYPE", "auto"),
        )
        sp = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        responses: list[str] = []
        bs = max(1, int(batch_size))
        for i in range(0, len(prompts), bs):
            chunk = prompts[i : i + bs]
            outputs = llm.generate(chunk, sp)
            responses.extend([(o.outputs[0].text or "").strip() for o in outputs])
            _set_status(job_id, step=len(responses))
            _append_log(job_id, f"generate {len(responses)}/{len(prompts)}")

        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        gen_path = os.path.join(output_dir, "hexphi_generations_only.jsonl")
        write_generations_jsonl(_Path(gen_path), users, responses, meta, hf_model_id)
        _append_log(job_id, f"generations → {gen_path}")

        pairs = list(zip(users, responses))

        # ---- Phase B: score -----------------------------------------------
        summary: dict = {
            "audit": audit_type,
            "dataset": AUDIT_DATASET,
            "generator_model": hf_model_id,
            "n_input": len(pairs),
            "run_name": run_name,
            "output_dir": output_dir,
            "generations_jsonl": gen_path,
        }

        if audit_type == "harmful":
            _set_status(job_id, stage="scoring_harmful", step=0, max_steps=len(pairs))
            _append_log(job_id, f"loading {guard_model} …")
            from harmful_score import LlamaGuardJudge

            judge = LlamaGuardJudge(model_id=guard_model)
            judge.load()
            _append_log(job_id, f"scoring {len(pairs)} pairs")
            hs, flags, verdicts = judge.score(pairs)

            final_path = os.path.join(output_dir, "hexphi_with_llama_guard.jsonl")
            with open(final_path, "w", encoding="utf-8") as f:
                for i, (u, r) in enumerate(zip(users, responses)):
                    rec = {
                        "idx": meta[i].get("idx", i) if i < len(meta) else i,
                        "user": u,
                        "model_response": r,
                        "llama_guard_flagged": bool(flags[i]),
                        "llama_guard_verdict": verdicts[i],
                        "generator_model": hf_model_id,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            summary.update(
                {
                    "guard_model": guard_model,
                    "n": len(flags),
                    "flagged_count": int(sum(1 for x in flags if x)),
                    "harmful_score_llama_guard": float(hs),
                    "final_artifact": final_path,
                }
            )
            _append_log(
                job_id,
                f"harmful_score={hs:.4f}  flagged={summary['flagged_count']}/{summary['n']}",
            )

        elif audit_type == "asr":
            if not openai_api_key:
                raise RuntimeError("ASR audit requires openai_api_key (not provided).")
            _set_status(job_id, stage="scoring_asr", step=0, max_steps=len(pairs))
            # Put the key on the env just long enough for the OpenAI client to
            # pick it up, then strip it the moment we're done.
            os.environ["OPENAI_API_KEY"] = openai_api_key
            try:
                _append_log(job_id, f"scoring {len(pairs)} pairs via {asr_judge_model}")
                from asr import compute_asr_gpt_judge

                asr_res = compute_asr_gpt_judge(pairs, judge_model=asr_judge_model)
            finally:
                os.environ.pop("OPENAI_API_KEY", None)
                # best-effort: nuke our local ref too
                try:
                    del openai_api_key  # noqa: F821
                except Exception:
                    pass

            gscores = asr_res["scores"]
            raws = asr_res["raw_responses"]
            final_path = os.path.join(output_dir, "hexphi_with_asr_scores.jsonl")
            with open(final_path, "w", encoding="utf-8") as f:
                for i, (u, r) in enumerate(zip(users, responses)):
                    raw_i = raws[i] if i < len(raws) else ""
                    rec = {
                        "idx": meta[i].get("idx", i) if i < len(meta) else i,
                        "user": u,
                        "model_response": r,
                        "gpt_judge_score": gscores[i],
                        "gpt_judge_raw": raw_i[:1500],
                        "generator_model": hf_model_id,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_scored = sum(1 for s in gscores if s is not None)
            summary.update(
                {
                    "asr_judge_model": asr_judge_model,
                    "n_scored": n_scored,
                    "asr": float(asr_res["asr"]),
                    "avg_score": float(asr_res["avg_score"]),
                    "final_artifact": final_path,
                }
            )
            _append_log(
                job_id,
                f"asr={asr_res['asr']:.4f}  avg={asr_res['avg_score']:.3f}  scored={n_scored}/{len(gscores)}",
            )

        else:
            raise ValueError(f"Unknown audit_type: {audit_type!r}")

        summary_path = os.path.join(output_dir, "hexphi_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        _append_log(job_id, f"summary → {summary_path}")

        try:
            modal.Volume.from_name("kl_ablation_checkpoints").commit()
        except Exception:
            pass

        _set_status(
            job_id,
            status="completed",
            stage="done",
            finished_at=_now_iso(),
            result=summary,
        )
        return summary

    except Exception as e:
        tb = traceback.format_exc()
        _append_log(job_id, f"ERROR {type(e).__name__}: {e}")
        _append_log(job_id, tb)
        _set_status(job_id, status="failed", stage="error", error=str(e))
        raise


# ---- Cancel cleanup (volume + HF) -------------------------------------------


@app.function(
    image=image,
    timeout=900,
    volumes={"/checkpoints": modal.Volume.from_name("kl_ablation_checkpoints")},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def cleanup_job(job_id: str) -> dict:
    """Runs after a hard-cancel. Removes the job's checkpoint dir from the
    Modal volume and, if this job created a new HF repo and the push didn't
    finish, deletes that repo. Idempotent."""
    import shutil

    entry = JOB_STATUS.get(job_id) or {}
    output_dir: Optional[str] = entry.get("output_dir")
    hf_repo_id: Optional[str] = entry.get("hf_repo_id")
    created_this_run: bool = bool(entry.get("hf_repo_created_this_run"))
    push_complete: bool = bool(entry.get("hf_push_complete"))

    cleanup_report = {"volume_removed": False, "hf_deleted": False}

    if output_dir:
        try:
            vol = modal.Volume.from_name("kl_ablation_checkpoints")
            try:
                vol.reload()
            except Exception:
                pass
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                cleanup_report["volume_removed"] = True
                _append_log(job_id, f"cleanup: removed {output_dir}")
            try:
                vol.commit()
            except Exception as e:
                _append_log(job_id, f"cleanup: volume.commit failed: {e}")
        except Exception as e:
            _append_log(job_id, f"cleanup: volume rm failed: {e}")

    if hf_repo_id and created_this_run and not push_complete:
        try:
            from huggingface_hub import HfApi

            hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )
            if hf_token:
                HfApi(token=hf_token).delete_repo(hf_repo_id, missing_ok=True)
                cleanup_report["hf_deleted"] = True
                _append_log(job_id, f"cleanup: deleted partial HF repo {hf_repo_id}")
        except Exception as e:
            _append_log(job_id, f"cleanup: HF delete failed: {e}")

    _set_status(
        job_id,
        status="cancelled",
        stage="cancelled",
        finished_at=_now_iso(),
        cleanup=cleanup_report,
    )
    _append_log(job_id, f"cleanup done: {json.dumps(cleanup_report)}")
    return cleanup_report


# ---- FastAPI web endpoint (Vercel frontend calls this) ----------------------


@app.function(
    image=web_image,
    min_containers=0,
    timeout=60,
    secrets=[modal.Secret.from_name("lexforge-api-key")],
)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException, Header, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import List

    webapp = FastAPI(title="LexForge SFT API", version="1.2")
    webapp.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        # Expose custom auth headers so browsers preflight correctly.
        allow_headers=["*"],
    )

    SHARED_API_KEY = (os.environ.get("LEXFORGE_API_KEY") or "").strip()

    def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if not SHARED_API_KEY:
            raise HTTPException(500, "Server misconfigured: LEXFORGE_API_KEY not set.")
        if not x_api_key or not secrets.compare_digest(x_api_key, SHARED_API_KEY):
            raise HTTPException(401, "Invalid or missing X-API-Key.")

    def require_client_id(x_client_id: Optional[str] = Header(default=None)) -> str:
        cid = (x_client_id or "").strip()
        # Accept any non-empty 8-128 char token. The frontend generates a uuid.
        if not cid or not (8 <= len(cid) <= 128) or not re.fullmatch(r"[A-Za-z0-9_\-]+", cid):
            raise HTTPException(400, "Missing/invalid X-Client-Id header.")
        return cid

    def _load_job_or_404(job_id: str) -> dict:
        try:
            entry = JOB_STATUS.get(job_id)
        except Exception:
            entry = None
        if not entry:
            raise HTTPException(404, "Job not found")
        return entry

    def require_job_token(job_id: str, x_job_token: Optional[str] = Header(default=None)) -> dict:
        entry = _load_job_or_404(job_id)
        expected = entry.get("job_token") or ""
        if not x_job_token or not expected or not secrets.compare_digest(x_job_token, expected):
            raise HTTPException(401, "Invalid or missing X-Job-Token.")
        return entry

    class SubmitReq(BaseModel):
        model_id: str
        dataset_id: str
        hf_repo_id: Optional[str] = None
        private_repo: bool = False
        epochs: float = 1.0
        per_device_bs: int = Field(2, ge=1, le=16)
        grad_accum: int = Field(8, ge=1, le=64)
        lr: float = Field(2e-4, gt=0)
        warmup_ratio: float = Field(0.03, ge=0.0, le=0.5)
        weight_decay: float = Field(0.01, ge=0.0, le=1.0)
        max_seq_length: int = Field(2048, ge=128, le=8192)
        lora_r: int = Field(32, ge=1, le=256)
        lora_alpha: int = Field(64, ge=1, le=512)
        lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
        logging_steps: int = 25
        eval_samples: int = Field(2000, ge=0, le=10000)
        seed: int = 42

    class AuditSubmitReq(BaseModel):
        hf_model_id: str
        audit_type: str  # "harmful" | "asr"
        openai_api_key: Optional[str] = None  # ephemeral — only used for asr
        max_samples: int = Field(100, ge=5, le=AUDIT_DATASET_MAX)
        max_tokens: int = Field(512, ge=16, le=2048)
        temperature: float = Field(0.6, ge=0.0, le=2.0)
        top_p: float = Field(0.9, ge=0.0, le=1.0)

    @webapp.get("/api/options")
    def options() -> dict:
        return {
            "models": [{"id": mid, "label": label} for mid, label in MODEL_OPTIONS.items()],
            "datasets": [{"id": did, "label": label} for did, label in DATASET_OPTIONS.items()],
            "limits": {
                "max_concurrent_gpu_jobs": MAX_CONCURRENT_GPU_JOBS,
                "max_concurrent_users": MAX_CONCURRENT_USERS,
                "max_jobs_per_user": MAX_JOBS_PER_USER,
            },
        }

    @webapp.get("/api/capacity")
    def capacity(client_id: str = Depends(require_client_id)) -> dict:
        active = [j for j in _iter_all_jobs() if j.get("status") in ACTIVE_STATUSES]
        user_active = [j for j in active if j.get("client_id") == client_id]
        return {
            "active_gpu_jobs": len(active),
            "active_users": len({j.get("client_id") for j in active if j.get("client_id")}),
            "my_active_jobs": len(user_active),
            "max_concurrent_gpu_jobs": MAX_CONCURRENT_GPU_JOBS,
            "max_concurrent_users": MAX_CONCURRENT_USERS,
            "max_jobs_per_user": MAX_JOBS_PER_USER,
        }

    @webapp.post("/api/submit")
    def submit(
        req: SubmitReq,
        _: None = Depends(require_api_key),
        client_id: str = Depends(require_client_id),
    ) -> dict:
        if req.model_id not in MODEL_OPTIONS:
            raise HTTPException(400, f"Unknown model_id: {req.model_id}")
        if req.dataset_id not in DATASET_OPTIONS:
            raise HTTPException(400, f"Unknown dataset_id: {req.dataset_id}")

        # --- concurrency caps ------------------------------------------------
        user_active = _active_jobs_for_user(client_id)
        if len(user_active) >= MAX_JOBS_PER_USER:
            raise HTTPException(
                429,
                f"Per-user limit reached: you already have {len(user_active)} active job(s). "
                f"Max is {MAX_JOBS_PER_USER}. Cancel one before submitting another.",
            )
        active_users = _active_user_ids()
        if client_id not in active_users and len(active_users) >= MAX_CONCURRENT_USERS:
            raise HTTPException(
                429,
                f"System is at capacity: {MAX_CONCURRENT_USERS} users are currently training. "
                "Please try again when a slot opens up.",
            )

        job_id = uuid.uuid4().hex[:12]
        job_token = secrets.token_urlsafe(24)
        run_name = _build_run_name(req.model_id, req.dataset_id, job_id)
        output_dir = f"/checkpoints/lexforge/{run_name}"

        JOB_STATUS[job_id] = {
            "job_id": job_id,
            "job_token": job_token,
            "client_id": client_id,
            "run_name": run_name,
            "output_dir": output_dir,
            "status": "queued",
            "stage": "queued",
            "cancel_requested": False,
            "model_id": req.model_id,
            "dataset_id": req.dataset_id,
            "hf_repo_id": req.hf_repo_id,
            "private_repo": req.private_repo,
            "hf_repo_created_this_run": False,
            "hf_push_complete": False,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "logs": [f"[{_now_iso()}] queued as {run_name}"],
            "step": 0,
            "max_steps": None,
            "params": req.model_dump(),
        }

        call = train.spawn(
            job_id=job_id,
            model_id=req.model_id,
            output_dir=output_dir,
            dataset_id=req.dataset_id,
            max_seq_length=req.max_seq_length,
            epochs=req.epochs,
            per_device_bs=req.per_device_bs,
            grad_accum=req.grad_accum,
            lr=req.lr,
            warmup_ratio=req.warmup_ratio,
            weight_decay=req.weight_decay,
            logging_steps=req.logging_steps,
            eval_samples=req.eval_samples,
            seed=req.seed,
            lora_r=req.lora_r,
            lora_alpha=req.lora_alpha,
            lora_dropout=req.lora_dropout,
            hf_repo_id=req.hf_repo_id,
            push_to_hub=bool(req.hf_repo_id),
            private_repo=req.private_repo,
            run_name=run_name,
        )

        entry = JOB_STATUS[job_id]
        entry["function_call_id"] = call.object_id
        JOB_STATUS[job_id] = entry

        return {
            "job_id": job_id,
            "job_token": job_token,
            "run_name": run_name,
            "function_call_id": call.object_id,
            "kind": "train",
        }

    @webapp.post("/api/audit/submit")
    def audit_submit(
        req: AuditSubmitReq,
        _: None = Depends(require_api_key),
        client_id: str = Depends(require_client_id),
    ) -> dict:
        if req.audit_type not in ("harmful", "asr"):
            raise HTTPException(400, "audit_type must be 'harmful' or 'asr'.")
        if not HF_MODEL_ID_RE.match((req.hf_model_id or "").strip()):
            raise HTTPException(
                400,
                "hf_model_id must look like 'owner/repo' (letters, digits, . _ - only).",
            )
        if req.audit_type == "asr":
            key = (req.openai_api_key or "").strip()
            if not key or len(key) < 20 or not key.startswith(("sk-", "sk_")):
                raise HTTPException(
                    400,
                    "ASR audit requires your openai_api_key (must start with 'sk-').",
                )

        user_active = _active_jobs_for_user(client_id)
        if len(user_active) >= MAX_JOBS_PER_USER:
            raise HTTPException(
                429,
                f"Per-user limit reached: you already have {len(user_active)} active job(s). "
                f"Max is {MAX_JOBS_PER_USER}. Cancel one before submitting another.",
            )
        active_users = _active_user_ids()
        if client_id not in active_users and len(active_users) >= MAX_CONCURRENT_USERS:
            raise HTTPException(
                429,
                f"System is at capacity: {MAX_CONCURRENT_USERS} users are currently running. "
                "Please try again when a slot opens up.",
            )

        job_id = uuid.uuid4().hex[:12]
        job_token = secrets.token_urlsafe(24)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_slug = _slugify(req.hf_model_id.split("/")[-1])
        run_name = f"{ts}-audit-{req.audit_type}-{model_slug}-{job_id}"
        output_dir = f"{AUDIT_OUTPUT_BASE}/{run_name}"

        # NOTE: we intentionally do NOT store openai_api_key in JOB_STATUS.
        JOB_STATUS[job_id] = {
            "job_id": job_id,
            "job_token": job_token,
            "client_id": client_id,
            "kind": "audit",
            "run_name": run_name,
            "output_dir": output_dir,
            "status": "queued",
            "stage": "queued",
            "cancel_requested": False,
            "hf_model_id": req.hf_model_id,
            "audit_type": req.audit_type,
            "max_samples": req.max_samples,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "logs": [f"[{_now_iso()}] queued {run_name} (audit: {req.audit_type})"],
            "step": 0,
            "max_steps": None,
            "params": {
                "hf_model_id": req.hf_model_id,
                "audit_type": req.audit_type,
                "max_samples": req.max_samples,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
            },
        }

        call = audit.spawn(
            job_id=job_id,
            hf_model_id=req.hf_model_id,
            audit_type=req.audit_type,
            output_dir=output_dir,
            run_name=run_name,
            max_samples=req.max_samples,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            openai_api_key=(req.openai_api_key or None) if req.audit_type == "asr" else None,
        )

        entry = JOB_STATUS[job_id]
        entry["function_call_id"] = call.object_id
        JOB_STATUS[job_id] = entry

        return {
            "job_id": job_id,
            "job_token": job_token,
            "run_name": run_name,
            "function_call_id": call.object_id,
            "kind": "audit",
        }

    @webapp.get("/api/jobs/{job_id}")
    def get_job(
        job_id: str,
        tail: int = 400,
        entry: dict = Depends(require_job_token),
    ) -> dict:
        out = dict(entry)
        logs: List[str] = out.get("logs") or []
        if tail and tail > 0:
            out["logs"] = logs[-tail:]
        # Never leak the token back to the client — they already have it.
        out.pop("job_token", None)
        # Scrub internal identifiers we don't need to expose.
        out.pop("function_call_id", None)
        # Defensive: never echo back anything that looks like a user secret.
        for k in ("openai_api_key", "openai_key", "api_key"):
            out.pop(k, None)
        params = out.get("params")
        if isinstance(params, dict):
            for k in ("openai_api_key", "openai_key", "api_key"):
                params.pop(k, None)
        return out

    @webapp.post("/api/jobs/{job_id}/cancel")
    def cancel_job(
        job_id: str,
        entry: dict = Depends(require_job_token),
    ) -> dict:
        status = entry.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return {"ok": True, "status": status, "noop": True}

        fc_id = entry.get("function_call_id")
        _set_status(job_id, status="cancelling", stage="cancelling", cancel_requested=True)
        _append_log(job_id, "cancel requested by user")

        if fc_id:
            try:
                modal.FunctionCall.from_id(fc_id).cancel()
                _append_log(job_id, f"sent hard-cancel to function call {fc_id}")
            except Exception as e:
                _append_log(job_id, f"FunctionCall.cancel() failed (continuing with cleanup): {e}")

        try:
            cleanup_job.spawn(job_id=job_id)
        except Exception as e:
            _append_log(job_id, f"cleanup spawn failed: {e}")
            _set_status(job_id, status="cancelled", stage="cancelled", error=str(e))

        return {"ok": True, "status": "cancelling"}

    @webapp.get("/")
    def health() -> dict:
        return {"ok": True, "service": "lexforge-sft-api", "version": "1.2"}

    return webapp


# ---- Local CLI --------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_id: str = MODEL_ID,
    dataset_id: str = DATASET_ID,
    hf_repo_id: str = None,
    epochs: float = 1.0,
):
    job_id = f"local-{uuid.uuid4().hex[:8]}"
    train.remote(
        job_id=job_id,
        model_id=model_id,
        dataset_id=dataset_id,
        hf_repo_id=hf_repo_id,
        push_to_hub=bool(hf_repo_id),
        epochs=epochs,
    )
