"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Option = { id: string; label: string };
type Options = { models: Option[]; datasets: Option[] };

type Job = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | string;
  stage?: string;
  model_id?: string;
  dataset_id?: string;
  hf_repo_id?: string | null;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  finished_at?: string;
  step?: number;
  max_steps?: number | null;
  epoch?: number;
  latest_metrics?: Record<string, unknown>;
  logs?: string[];
  error?: string;
  result?: {
    hf_url?: string;
    adapter_path?: string;
    eval_metrics?: Record<string, number>;
  };
};

const API_BASE = process.env.NEXT_PUBLIC_MODAL_API ?? "";

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export default function Page() {
  const [options, setOptions] = useState<Options | null>(null);
  const [modelId, setModelId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [hfRepoId, setHfRepoId] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);
  const [epochs, setEpochs] = useState(1);
  const [perDeviceBs, setPerDeviceBs] = useState(2);
  const [gradAccum, setGradAccum] = useState(8);
  const [lr, setLr] = useState(0.0002);
  const [warmup, setWarmup] = useState(0.03);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [maxSeq, setMaxSeq] = useState(2048);
  const [loraR, setLoraR] = useState(32);
  const [loraAlpha, setLoraAlpha] = useState(64);
  const [loraDropout, setLoraDropout] = useState(0.05);

  const [submitting, setSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!API_BASE) {
      setError(
        "NEXT_PUBLIC_MODAL_API is not set. Copy .env.example to .env.local and paste your Modal api URL."
      );
      return;
    }
    apiGet<Options>("/api/options")
      .then((o) => {
        setOptions(o);
        if (o.models.length) setModelId(o.models[0].id);
        if (o.datasets.length) setDatasetId(o.datasets[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    const tick = async () => {
      try {
        const j = await apiGet<Job>(`/api/jobs/${jobId}?tail=400`);
        if (!alive) return;
        setJob(j);
        requestAnimationFrame(() => {
          const el = logsRef.current;
          if (el) el.scrollTop = el.scrollHeight;
        });
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    tick();
    const handle = setInterval(tick, 2500);
    return () => {
      alive = false;
      clearInterval(handle);
    };
  }, [jobId]);

  const pct = useMemo(() => {
    if (!job?.max_steps || !job.step) return 0;
    return Math.min(100, Math.round((job.step / job.max_steps) * 100));
  }, [job]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { job_id } = await apiPost<{ job_id: string }>("/api/submit", {
        model_id: modelId,
        dataset_id: datasetId,
        hf_repo_id: hfRepoId.trim() || null,
        private_repo: privateRepo,
        epochs: Number(epochs),
        per_device_bs: Number(perDeviceBs),
        grad_accum: Number(gradAccum),
        lr: Number(lr),
        warmup_ratio: Number(warmup),
        weight_decay: Number(weightDecay),
        max_seq_length: Number(maxSeq),
        lora_r: Number(loraR),
        lora_alpha: Number(loraAlpha),
        lora_dropout: Number(loraDropout),
      });
      setJobId(job_id);
      setJob(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="wrap">
      <header className="brand">
        <div>
          <h1>LexForge AI — Fine-tuning Console</h1>
          <div className="sub">
            Submit a QLoRA SFT job to Modal · stream logs · auto-merge + push
            to HuggingFace Hub
          </div>
        </div>
        <div className="sub">
          {API_BASE ? <code>{new URL(API_BASE).host}</code> : null}
        </div>
      </header>

      {error ? <div className="error">{error}</div> : null}

      <div className="grid">
        <form className="card" onSubmit={onSubmit}>
          <h2>Training job</h2>

          <div className="field">
            <label>Base model</label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={!options}
            >
              {options?.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={!options}
            >
              {options?.datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>

          <div className="row">
            <div className="field">
              <label>Epochs</label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Max seq length</label>
              <input
                type="number"
                min="128"
                step="128"
                value={maxSeq}
                onChange={(e) => setMaxSeq(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Per-device batch</label>
              <input
                type="number"
                min="1"
                value={perDeviceBs}
                onChange={(e) => setPerDeviceBs(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Grad accum</label>
              <input
                type="number"
                min="1"
                value={gradAccum}
                onChange={(e) => setGradAccum(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row3">
            <div className="field">
              <label>Learning rate</label>
              <input
                type="number"
                step="0.00001"
                min="0"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Warmup ratio</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="0.5"
                value={warmup}
                onChange={(e) => setWarmup(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Weight decay</label>
              <input
                type="number"
                step="0.001"
                min="0"
                value={weightDecay}
                onChange={(e) => setWeightDecay(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row3">
            <div className="field">
              <label>LoRA r</label>
              <input
                type="number"
                min="1"
                value={loraR}
                onChange={(e) => setLoraR(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>LoRA α</label>
              <input
                type="number"
                min="1"
                value={loraAlpha}
                onChange={(e) => setLoraAlpha(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>LoRA dropout</label>
              <input
                type="number"
                min="0"
                max="0.5"
                step="0.01"
                value={loraDropout}
                onChange={(e) => setLoraDropout(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="field">
            <label>HuggingFace repo to push merged model (optional)</label>
            <input
              type="text"
              placeholder="your-username/lexforge-merged"
              value={hfRepoId}
              onChange={(e) => setHfRepoId(e.target.value)}
            />
            <div className="hint">
              Leaving this empty skips the merge + push step.
            </div>
          </div>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={privateRepo}
              onChange={(e) => setPrivateRepo(e.target.checked)}
            />
            private repo
          </label>

          <button className="submit" type="submit" disabled={submitting}>
            {submitting ? "submitting..." : "Launch training"}
          </button>
        </form>

        <div className="card">
          <h2>Live logs</h2>
          {!job ? (
            <div className="empty">
              {jobId
                ? "Waiting for the first heartbeat from Modal..."
                : "Submit a job to start streaming training logs here."}
            </div>
          ) : (
            <>
              <div className="status-line">
                <span className={`pill ${job.status}`}>{job.status}</span>
                <span>
                  <strong>{job.model_id}</strong> on{" "}
                  <strong>{job.dataset_id}</strong>
                </span>
                <span>stage: {job.stage ?? "—"}</span>
                {typeof job.step === "number" && job.max_steps ? (
                  <span>
                    step {job.step} / {job.max_steps}
                  </span>
                ) : null}
                {job.epoch != null ? (
                  <span>epoch {Number(job.epoch).toFixed(2)}</span>
                ) : null}
              </div>

              <div className="progress">
                <div style={{ width: `${pct}%` }} />
              </div>

              <div className="logs" ref={logsRef}>
                {(job.logs ?? []).map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>

              <div className="links">
                <span>
                  job_id: <code>{job.job_id}</code>
                </span>
                {job.result?.hf_url ? (
                  <a
                    href={job.result.hf_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    HF Hub →
                  </a>
                ) : null}
                {job.error ? (
                  <span style={{ color: "var(--err)" }}>
                    error: {job.error}
                  </span>
                ) : null}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
