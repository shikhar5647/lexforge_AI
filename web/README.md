# LexForge AI — Fine-tuning Console

Minimalistic Next.js frontend for submitting QLoRA SFT jobs to the Modal backend
defined in [`../finetune/sft_modal.py`](../finetune/sft_modal.py). The UI lets
the user pick a base model, a dataset, tweak the usual hyperparameters
(lr, batch size, warmup, LoRA r/α, …), kick off a Modal GPU job, watch live
training logs stream in, and — if an `hf_repo_id` is provided — merge the LoRA
adapter into the base model and push it to the HuggingFace Hub.

## Architecture

```
Vercel (this app, static Next.js)
        │  fetch()  https://<workspace>--lexforge-sft-qlora-api.modal.run
        ▼
Modal  ── api  (FastAPI, CPU, free tier)   POST /api/submit  /  GET /api/jobs/:id
        │   │
        │   └─ spawns ─┐
        ▼              ▼
Modal  ── train  (H200 GPU)  ── writes live logs to modal.Dict("lexforge-jobs")
                                ── saves LoRA adapter to modal.Volume
                                ── merges LoRA into base, pushes to HF Hub
```

Nothing persists on Vercel — it's a pure static frontend that calls the public
Modal URL.

## One-time Modal setup

1. Install Modal and log in on your workstation:
    ```bash
    pip install modal
    modal setup
    ```
2. Create the two secrets the trainer needs:
    ```bash
    # used to pull gated models + push merged model to Hub
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    ```
3. Create the checkpoint volume (if it doesn't already exist):
    ```bash
    modal volume create kl_ablation_checkpoints
    ```
4. Deploy the backend:
    ```bash
    cd ../finetune
    modal deploy sft_modal.py
    ```
   Modal will print a public URL for the `api` function, e.g.
   `https://your-workspace--lexforge-sft-qlora-api.modal.run`. Copy it.

## Deploy the frontend to Vercel

1. From this directory, either:
   ```bash
   npx vercel          # first time, creates a project
   npx vercel --prod   # subsequent deployments
   ```
   or connect the `FML_RAG/web` subtree to a new Vercel project via the
   dashboard (Framework preset = Next.js).
2. In the Vercel project settings add one environment variable:
    ```
    NEXT_PUBLIC_MODAL_API = https://your-workspace--lexforge-sft-qlora-api.modal.run
    ```
3. Redeploy. The site should now load, fetch model/dataset options from Modal,
   accept a training submission, and stream logs live.

## Run locally

```bash
cp .env.example .env.local
# edit .env.local, paste the Modal URL
npm install
npm run dev
# → http://localhost:3000
```

## What happens when you press “Launch training”

1. The browser POSTs the form to `POST /api/submit` on Modal.
2. Modal's `api` function mints a `job_id`, writes an initial status row into
   a persistent `modal.Dict`, and calls `train.spawn(...)` with the
   hyperparameters.
3. A fresh H200 container boots, loads the chosen base model in 4-bit NF4,
   attaches a LoRA adapter (r/α/dropout from the form), formats the chosen
   dataset through the Qwen/Phi/Llama chat template, and starts `SFTTrainer`.
4. A custom `TrainerCallback` appends every step's metrics (loss, grad_norm,
   lr, entropy, mean_token_accuracy, …) into the Modal `Dict` under this
   `job_id`.
5. The frontend polls `GET /api/jobs/:id` every 2.5s and re-renders the log
   pane + progress bar.
6. When training finishes and an `hf_repo_id` was set: Modal reloads the base
   model in bf16, merges the adapter via `peft.PeftModel.merge_and_unload()`,
   saves safetensors, creates the HF repo, and uploads the merged weights.
   A direct `https://huggingface.co/<repo>` link appears in the UI.

## Exposed endpoints (Modal `api`)

| Method | Path                   | Purpose                                    |
| ------ | ---------------------- | ------------------------------------------ |
| GET    | `/`                    | Health check                               |
| GET    | `/api/options`         | Supported base-model + dataset dropdowns   |
| POST   | `/api/submit`          | Queue a new training job                   |
| GET    | `/api/jobs/{job_id}`   | Full job record (status + logs tail)       |
| GET    | `/api/jobs`            | Last 20 jobs (no logs)                     |
