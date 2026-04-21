

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SA = Path(__file__).resolve().parent
if str(_SA) not in sys.path:
    sys.path.insert(0, str(_SA))

DATASET_ID = "gjyotin305/harmfulhexphi_300"
GENERATIONS_JSONL = "hexphi_generations_only.jsonl"
FINAL_HARMFUL = "hexphi_with_llama_guard.jsonl"
FINAL_ASR = "hexphi_with_asr_scores.jsonl"
SUMMARY_NAME = "hexphi_summary.json"


def _build_prompts(users: list[str], model_id: str) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompts: list[str] = []
    for u in users:
        messages = [{"role": "user", "content": u}]
        prompts.append(
            tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def generate_vllm(
    model_id: str,
    prompts: list[str],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model_id,
        trust_remote_code=True,
        max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        dtype=os.getenv("VLLM_DTYPE", "auto"),
    )
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    out: list[str] = []
    for i in range(0, len(prompts), max(1, batch_size)):
        chunk = prompts[i : i + batch_size]
        outputs = llm.generate(chunk, sp)
        out.extend([(o.outputs[0].text or "").strip() for o in outputs])
    del llm
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def write_generations_jsonl(
    path: Path,
    users: list[str],
    responses: list[str],
    meta: list[dict],
    generator_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (u, r) in enumerate(zip(users, responses)):
            row = {
                "idx": meta[i].get("idx", i) if i < len(meta) else i,
                "user": u,
                "model_response": r,
                "generator_model": generator_model,
            }
            if i < len(meta):
                for k, v in meta[i].items():
                    if k not in row:
                        row[k] = v
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_generations_jsonl(path: Path) -> tuple[list[str], list[str], list[dict]]:
    users: list[str] = []
    responses: list[str] = []
    meta: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        u = o.get("user") or o.get("prompt")
        r = o.get("model_response") or o.get("response") or o.get("answer")
        if not u or not r:
            continue
        users.append(str(u))
        responses.append(str(r))
        meta.append({k: v for k, v in o.items() if k not in ("user", "prompt", "model_response", "response", "answer")})
    return users, responses, meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="harmfulhexphi_300: generate all → audit (harmful | ASR).")
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default="train")
    p.add_argument("--model", type=str, default=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    p.add_argument("--out", type=Path, default=Path("hexphi_audit_out"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--score-from", type=Path, default=None)
    p.add_argument("--audit-only", type=Path, default=None, help="Alias for --score-from")
    p.add_argument(
        "--audit",
        choices=["harmful", "asr"],
        default="harmful",
        help="harmful=Llama Guard (local, HF_TOKEN). asr=GPT rubric ASR (needs user OPENAI_API_KEY).",
    )
    p.add_argument(
        "--phase2",
        choices=["harmful", "asr"],
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--guard-model", default=os.getenv("LLAMA_GUARD_MODEL", "meta-llama/Llama-Guard-3-8B"))
    p.add_argument("--asr-judge-model", default=os.getenv("ASR_JUDGE_MODEL", "gpt-4.1-mini"))
    args = p.parse_args(argv)
    audit_type = args.phase2 if args.phase2 is not None else args.audit

    score_from = args.score_from or args.audit_only
    args.out.mkdir(parents=True, exist_ok=True)
    generations_path = args.out / GENERATIONS_JSONL
    final_path = args.out / (FINAL_HARMFUL if audit_type == "harmful" else FINAL_ASR)
    summary_path = args.out / SUMMARY_NAME

    if score_from is not None:
        print(f"[Generate] skipped — loading completions from {score_from}", file=sys.stderr)
        users, responses, meta = load_generations_jsonl(score_from)
        gen_model = meta[0].get("generator_model") if meta else None
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split=args.split)
        n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
        ds = ds.select(range(n))
        users = [ds[i]["user"] for i in range(len(ds))]
        meta = [
            {"idx": i, "from": ds[i].get("from"), "score_clean": ds[i].get("score_clean")}
            for i in range(len(ds))
        ]
        print(f"[Generate] vLLM: {len(users)} prompts ({args.model})…", file=sys.stderr)
        prompts = _build_prompts(users, args.model)
        responses = generate_vllm(
            args.model,
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )
        write_generations_jsonl(generations_path, users, responses, meta, args.model)
        print(f"[Generate] saved → {generations_path}", file=sys.stderr)
        gen_model = args.model

    if not users:
        print("No rows.", file=sys.stderr)
        return 1

    pairs = list(zip(users, responses))
    skip_keys = {"user", "prompt", "model_response", "response", "answer"}

    if audit_type == "harmful":
        from harmful_score import LlamaGuardJudge

        print(f"[Audit: harmful] Llama Guard ({args.guard_model}), {len(pairs)} pairs…", file=sys.stderr)
        judge = LlamaGuardJudge(model_id=args.guard_model)
        hs, flags, verdicts = judge.score(pairs)
        with final_path.open("w", encoding="utf-8") as f:
            for i, (u, r) in enumerate(zip(users, responses)):
                rec: dict = {
                    "idx": meta[i].get("idx", i) if i < len(meta) else i,
                    "user": u,
                    "model_response": r,
                    "llama_guard_flagged": flags[i],
                    "llama_guard_verdict": verdicts[i],
                }
                if gen_model:
                    rec["generator_model"] = gen_model
                if i < len(meta):
                    for k, v in meta[i].items():
                        if k not in skip_keys and k not in rec:
                            rec[k] = v
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = {
            "audit": "harmful",
            "dataset": str(args.dataset) if score_from is None else str(score_from),
            "generator_model": gen_model,
            "guard_model": args.guard_model,
            "n": len(flags),
            "harmful_score_llama_guard": hs,
            "flagged_count": int(sum(flags)),
            "generations_jsonl": str(generations_path) if score_from is None else str(score_from),
            "final_artifact": str(final_path),
        }
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ASR audit requires OPENAI_API_KEY (user-provided in your app).", file=sys.stderr)
            return 1
        from asr import compute_asr_gpt_judge

        print(f"[Audit: asr] GPT judge ({args.asr_judge_model}), {len(pairs)} pairs…", file=sys.stderr)
        asr_res = compute_asr_gpt_judge(pairs, judge_model=args.asr_judge_model)
        gscores = asr_res["scores"]
        raws = asr_res["raw_responses"]
        with final_path.open("w", encoding="utf-8") as f:
            for i, (u, r) in enumerate(zip(users, responses)):
                rec: dict = {
                    "idx": meta[i].get("idx", i) if i < len(meta) else i,
                    "user": u,
                    "model_response": r,
                    "gpt_judge_score": gscores[i],
                    "gpt_judge_raw": (raws[i][:1500] if i < len(raws) else ""),
                }
                if gen_model:
                    rec["generator_model"] = gen_model
                if i < len(meta):
                    for k, v in meta[i].items():
                        if k not in skip_keys and k not in rec:
                            rec[k] = v
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = {
            "audit": "asr",
            "dataset": str(args.dataset) if score_from is None else str(score_from),
            "generator_model": gen_model,
            "asr_judge_model": args.asr_judge_model,
            "n_input": len(pairs),
            "n_scored": sum(1 for s in gscores if s is not None),
            "asr": asr_res["asr"],
            "avg_score": asr_res["avg_score"],
            "generations_jsonl": str(generations_path) if score_from is None else str(score_from),
            "final_artifact": str(final_path),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {final_path} and {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
