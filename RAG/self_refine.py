from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from basic_rag import (
    DEFAULT_DOCS_DIR,
    DEFAULT_VLLM_MODEL,
    SYSTEM_PROMPT,
    LegalDocumentRAG,
    build_context_block,
)
from ingestion import ingest_paths, iter_supported_files


CRITIQUE_SYSTEM = """You are a meticulous legal editor. Review a draft answer against the provided excerpts only.

Output:
1) Bulleted list of issues (unsupported claims, missing [Source n] citations, contradictions with excerpts, vagueness).
2) One line: "Verdict: acceptable" or "Verdict: needs revision".

Do not rewrite the full answer here — only critique."""


REFINE_SYSTEM = """You are a legal research assistant. Revise the draft using the critique and the excerpts.

Rules:
- Fix every substantive issue raised in the critique when the excerpts support a correction.
- If the critique requests something the excerpts cannot support, say the materials are insufficient for that point.
- Cite [Source n] for claims tied to excerpts.
- Produce only the revised answer (no preamble about being revised)."""


def _draft_user(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        "Write a careful draft answer. Cite [Source n] where appropriate."
    )


def _critique_user(question: str, context: str, draft: str) -> str:
    return (
        f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Critique the draft against the excerpts only."
    )


def _refine_user(question: str, context: str, draft: str, critique: str) -> str:
    return (
        f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Write the improved final answer."
    )


def answer_with_self_refine(
    rag: LegalDocumentRAG,
    question: str,
    *,
    rounds: int = 2,
    draft_temperature: float = 0.2,
    critique_temperature: float = 0.2,
    refine_temperature: float = 0.1,
    top_k: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    if rag.chunk_count == 0:
        raise RuntimeError(
            f"No documents found in {rag.documents_dir}. Add .pdf, .docx, or .txt files."
        )
    hits = rag.retrieve(question, top_k=top_k)
    context = build_context_block(hits)

    draft = rag.generate_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _draft_user(question, context)},
        ],
        model=model,
        temperature=draft_temperature,
        max_tokens=max_tokens,
        seed=7,
    )

    critique = ""
    for _ in range(max(1, rounds)):
        critique = rag.generate_messages(
            [
                {"role": "system", "content": CRITIQUE_SYSTEM},
                {"role": "user", "content": _critique_user(question, context, draft)},
            ],
            model=model,
            temperature=critique_temperature,
            max_tokens=max_tokens,
            seed=11,
        )
        draft = rag.generate_messages(
            [
                {"role": "system", "content": REFINE_SYSTEM},
                {"role": "user", "content": _refine_user(question, context, draft, critique)},
            ],
            model=model,
            temperature=refine_temperature,
            max_tokens=max_tokens,
            seed=13,
        )

    return draft.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Legal RAG + self-refine (draft → critique → revise).")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR, help="Document folder")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"vLLM model id (default: VLLM_MODEL or {DEFAULT_VLLM_MODEL})",
    )
    p.add_argument("--top-k", type=int, default=None, help="Chunks after reranking")
    p.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per call")
    p.add_argument(
        "--rounds",
        type=int,
        default=int(os.getenv("SELF_REFINE_ROUNDS", "2")),
        help="Number of critique→refine cycles (each cycle updates the draft)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List documents and chunk counts (no vLLM)",
    )
    p.add_argument("--retrieve-only", action="store_true", help="Show retrieval only")
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

    rag = LegalDocumentRAG(documents_dir=args.docs, vllm_model=args.model)

    if not args.question:
        p.print_help()
        print('\nExample: python self_refine.py --docs ./documents --rounds 2 "Your question?"')
        return 1

    if args.retrieve_only:
        for ch, sc in rag.retrieve(args.question, top_k=args.top_k or 8):
            print(
                f"\n{'='*60}\nscore={sc:.4f} doc={ch.doc_id} pages={ch.page_start}-{ch.page_end}\n{ch.text[:800]}…",
            )
        return 0

    try:
        out = answer_with_self_refine(
            rag,
            args.question,
            rounds=max(1, args.rounds),
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            model=args.model,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
