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


CONSOLIDATION_INSTRUCTION = """You are given one legal question, the same document excerpts for all candidates, and several numbered candidate answers produced independently.

Your job:
- Produce ONE final answer that is best supported by the excerpts.
- Resolve disagreements by favoring claims explicitly supported by the excerpts; if excerpts are insufficient, say so.
- Preserve accurate citations like [Source n] where appropriate.
- Do not introduce new facts beyond the excerpts."""


def _user_answer_prompt(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        "Answer the question using the excerpts and cite [Source n] where appropriate."
    )


def answer_with_self_consistency(
    rag: LegalDocumentRAG,
    question: str,
    *,
    n_samples: int = 5,
    sample_temperature: float = 0.7,
    consolidate_temperature: float = 0.1,
    top_k: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    seeds: list[int] | None = None,
) -> str:
    if rag.chunk_count == 0:
        raise RuntimeError(
            f"No documents found in {rag.documents_dir}. Add .pdf, .docx, or .txt files."
        )
    hits = rag.retrieve(question, top_k=top_k)
    context = build_context_block(hits)
    base_user = _user_answer_prompt(question, context)

    if seeds is None:
        seeds = [42 + i for i in range(n_samples)]

    samples: list[str] = []
    for i in range(n_samples):
        seed = seeds[i] if i < len(seeds) else 42 + i
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": base_user},
        ]
        text = rag.generate_messages(
            messages,
            model=model,
            temperature=sample_temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=0.95,
        )
        samples.append(text)

    numbered = "\n\n".join(f"Candidate {i + 1}:\n{s}" for i, s in enumerate(samples))
    consolidate_user = (
        f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
        f"{numbered}\n\n"
        "Write the single best final answer (not a meta-discussion)."
    )
    final_messages = [
        {"role": "system", "content": CONSOLIDATION_INSTRUCTION},
        {"role": "user", "content": consolidate_user},
    ]
    return rag.generate_messages(
        final_messages,
        model=model,
        temperature=consolidate_temperature,
        max_tokens=max_tokens,
        seed=12345,
    ).strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Legal RAG + self-consistency (K samples + consolidation).",
    )
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
        "--samples",
        type=int,
        default=int(os.getenv("SELF_CONSISTENCY_SAMPLES", "5")),
        help="Number of independent answers before consolidation",
    )
    p.add_argument(
        "--sample-temperature",
        type=float,
        default=float(os.getenv("SELF_CONSISTENCY_SAMPLE_TEMP", "0.7")),
        help="Temperature for diversity across samples",
    )
    p.add_argument(
        "--consolidate-temperature",
        type=float,
        default=0.1,
        help="Temperature for the final merge step",
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
        print(
            f'\nExample: python self_consistency.py --docs ./documents --samples 5 "Your question?"',
        )
        return 1

    if args.retrieve_only:
        for ch, sc in rag.retrieve(args.question, top_k=args.top_k or 8):
            print(
                f"\n{'='*60}\nscore={sc:.4f} doc={ch.doc_id} pages={ch.page_start}-{ch.page_end}\n{ch.text[:800]}…",
            )
        return 0

    try:
        out = answer_with_self_consistency(
            rag,
            args.question,
            n_samples=max(1, args.samples),
            sample_temperature=args.sample_temperature,
            consolidate_temperature=args.consolidate_temperature,
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
