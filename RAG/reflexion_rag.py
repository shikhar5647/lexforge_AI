
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from basic_rag import (
    DEFAULT_DOCS_DIR,
    DEFAULT_VLLM_MODEL,
    SYSTEM_PROMPT,
    LegalDocumentRAG,
    build_context_block,
)
from ingestion import Chunk, ingest_paths, iter_supported_files


def _gen(
    rag: LegalDocumentRAG,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int | None,
    seed: int,
    model: str | None,
) -> str:
    return rag.generate_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        top_p=0.95,
    ).strip()


def _chunk_key(ch: Chunk) -> str:
    return f"{ch.doc_id}::{ch.chunk_index}"


def merge_retrieval(
    primary: list[tuple[Chunk, float]],
    secondary: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    acc: dict[str, tuple[Chunk, float]] = {}
    for ch, sc in primary + secondary:
        k = _chunk_key(ch)
        if k not in acc or sc > acc[k][1]:
            acc[k] = (ch, sc)
    return sorted(acc.values(), key=lambda x: -x[1])


def parse_verdict(eval_text: str) -> tuple[bool, str]:
    """Returns (passed, raw_eval_text for reflector). Requires an explicit VERDICT line."""
    m = re.search(r"verdict\s*:\s*(pass|fail)", eval_text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "pass", eval_text
    return False, eval_text


EVALUATOR_SYSTEM = """You are a strict evaluator for legal question answering over fixed excerpts.

You must judge ONLY against the provided excerpts (not outside law). Output a structured verdict."""


def evaluator_agent(
    rag: LegalDocumentRAG,
    question: str,
    context: str,
    draft: str,
    *,
    max_tokens: int | None,
    model: str | None,
) -> tuple[bool, str]:
    user = (
        f"Question:\n{question}\n\nExcerpts:\n{context}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Evaluate:\n"
        "1) Faithfulness — any factual claim not clearly supported by the excerpts?\n"
        "2) Citations — does the answer cite [Source n] where it relies on specific passages?\n"
        "3) Completeness — does it directly address the question, including necessary caveats if excerpts are thin?\n\n"
        "End with EXACTLY one line of the form:\nVERDICT: PASS\nor\nVERDICT: FAIL\n\n"
        "If VERDICT is FAIL, add short bullet lines under a header ISSUES: (max 6 bullets)."
    )
    raw = _gen(rag, EVALUATOR_SYSTEM, user, temperature=0.0, max_tokens=max_tokens, seed=9001, model=model)
    ok, _ = parse_verdict(raw)
    return ok, raw


REFLECTOR_SYSTEM = """You implement the Reflexion memory step: write a short, actionable lesson for the NEXT attempt.

Rules:
- 3–6 sentences max, plain language.
- Focus on what went wrong and concrete behavior changes (e.g., cite specific sources, avoid X, check section Y).
- Do not restate the full answer; store durable guidance only."""


def reflector_agent(
    rag: LegalDocumentRAG,
    question: str,
    draft: str,
    eval_feedback: str,
    prior_memory: list[str],
    *,
    max_tokens: int | None,
    model: str | None,
) -> str:
    mem = "\n".join(f"- Trial {i + 1}: {r}" for i, r in enumerate(prior_memory)) or "(none yet)"
    user = (
        f"Question:\n{question}\n\n"
        f"Prior episodic memory:\n{mem}\n\n"
        f"Latest draft:\n{draft}\n\n"
        f"Evaluator feedback:\n{eval_feedback}\n\n"
        "Write the new reflection paragraph for episodic memory."
    )
    return _gen(rag, REFLECTOR_SYSTEM, user, temperature=0.2, max_tokens=min(512, max_tokens or 512), seed=9002, model=model)


def actor_user_block(
    question: str,
    context: str,
    reflections: list[str],
) -> str:
    extra = ""
    if reflections:
        lines = "\n".join(f"({i + 1}) {r}" for i, r in enumerate(reflections))
        extra = (
            "\nEpisodic memory from prior trials (apply these lessons; do not repeat the same mistakes):\n"
            f"{lines}\n"
        )
    return (
        f"{extra}\nQuestion:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        "Answer using ONLY the excerpts. Cite [Source n] where appropriate."
    )


def actor_agent(
    rag: LegalDocumentRAG,
    question: str,
    context: str,
    reflections: list[str],
    *,
    actor_temperature: float,
    max_tokens: int | None,
    model: str | None,
    trial_seed: int,
) -> str:
    user = actor_user_block(question, context, reflections)
    return _gen(
        rag,
        SYSTEM_PROMPT,
        user,
        temperature=actor_temperature,
        max_tokens=max_tokens,
        seed=trial_seed,
        model=model,
    )


def answer_with_reflexion(
    rag: LegalDocumentRAG,
    question: str,
    *,
    max_trials: int = 3,
    top_k: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    actor_temperature: float = 0.15,
    retrieve_after_fail: bool = True,
    reflection_retrieval_max_chars: int = 400,
) -> tuple[str, list[str], list[str]]:
    """
    Returns (final_answer, episodic_reflections, draft_history).
    """
    if rag.chunk_count == 0:
        raise RuntimeError(f"No documents in {rag.documents_dir}")

    memory: list[str] = []
    drafts: list[str] = []
    hits = rag.retrieve(question, top_k=top_k)

    for trial in range(max_trials):
        # Optional: augment retrieval using compressed reflection (Reflexion-informed search)
        if trial > 0 and retrieve_after_fail and memory:
            tail = memory[-1][:reflection_retrieval_max_chars]
            extra_hits = rag.retrieve(f"{question}\n\nFocus for retrieval: {tail}", top_k=top_k)
            hits = merge_retrieval(hits, extra_hits)

        context = build_context_block(hits)
        draft = actor_agent(
            rag,
            question,
            context,
            memory,
            actor_temperature=actor_temperature,
            max_tokens=max_tokens,
            model=model,
            trial_seed=7100 + trial,
        )
        drafts.append(draft)

        ok, eval_raw = evaluator_agent(
            rag,
            question,
            context,
            draft,
            max_tokens=max_tokens,
            model=model,
        )
        if ok:
            return draft, memory, drafts

        reflection = reflector_agent(
            rag,
            question,
            draft,
            eval_raw,
            memory,
            max_tokens=max_tokens,
            model=model,
        )
        if reflection.strip():
            memory.append(reflection.strip())

    return drafts[-1], memory, drafts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Legal RAG + Reflexion (evaluator + episodic verbal reflections).")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)
    p.add_argument("--model", type=str, default=None, help=f"vLLM model (default {DEFAULT_VLLM_MODEL})")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("REFLEXION_MAX_TRIALS", "3")),
        help="Max actor trials (each fail adds one reflection to memory)",
    )
    p.add_argument(
        "--no-reflection-retrieval",
        action="store_true",
        help="Disable extra hybrid retrieval guided by the latest reflection after trial>0",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print episodic memory and evaluator snippets to stderr",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("question", nargs="?", default=None)
    args = p.parse_args(argv)

    if args.list:
        docs_path = Path(args.docs).expanduser().resolve()
        files = list(iter_supported_files(docs_path))
        chunks = ingest_paths(files)
        print(f"Documents dir: {docs_path}")
        for f in files:
            print(f"  - {f.name}")
        print(f"Total chunks: {len(chunks)}")
        return 0

    if not args.question:
        p.print_help()
        print('\nExample: python reflexion_rag.py --docs ./documents "Your question?"')
        return 1

    rag = LegalDocumentRAG(documents_dir=args.docs, vllm_model=args.model)
    try:
        retrieve_after_fail = (
            not args.no_reflection_retrieval
            and os.getenv("REFLEXION_RETRIEVE_AFTER_FAIL", "1").strip().lower() not in ("0", "false", "no")
        )
        final, memory, drafts = answer_with_reflexion(
            rag,
            args.question,
            max_trials=max(1, args.trials),
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            model=args.model,
            retrieve_after_fail=retrieve_after_fail,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"[reflexion] trials used: {len(drafts)}", file=sys.stderr)
        for i, r in enumerate(memory):
            print(f"[reflexion] memory[{i}]: {r[:500]}…" if len(r) > 500 else f"[reflexion] memory[{i}]: {r}", file=sys.stderr)

    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
