from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from basic_rag import (
    DEFAULT_DOCS_DIR,
    DEFAULT_VLLM_MODEL,
    LegalDocumentRAG,
    build_context_block,
)
from ingestion import Chunk, ingest_paths, iter_supported_files


def chunk_node_id(ch: Chunk) -> str:
    return f"{ch.doc_id}::{ch.chunk_index}"


def format_nodes_block(nodes: list[tuple[Chunk, float]]) -> str:
    lines: list[str] = []
    for ch, sc in nodes:
        nid = chunk_node_id(ch)
        meta = ch.doc_id
        if ch.page_start is not None:
            meta += f" p.{ch.page_start}" + (
                f"-{ch.page_end}" if ch.page_end != ch.page_start else ""
            )
        lines.append(f'node_id: `{nid}`\nmeta: {meta}\nscore: {sc:.4f}\ncontent:\n{ch.text}\n')
    return "\n---\n".join(lines)


def _llm(
    rag: LegalDocumentRAG,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int | None,
    seed: int,
    model: str | None = None,
) -> str:
    return rag.generate_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        top_p=0.95,
    ).strip()


def parse_term_definitions(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        term, rest = line.split(":", 1)
        t, d = term.strip(), rest.strip()
        if t and d and len(t) < 120:
            out[t] = d
    return out


def definitions_agent(
    rag: LegalDocumentRAG,
    query: str,
    *,
    max_tokens: int | None,
    model: str | None,
) -> dict[str, str]:
    """Extract term-definition pairs from hybrid-retrieved excerpts (graph-free)."""
    hits = rag.retrieve(query, top_k=10)
    if not hits:
        return {}
    ctx = build_context_block(hits, max_chars=8000)
    system = (
        "You extract legal glosses from excerpts only. "
        "If a term is not defined in excerpts, skip it or write 'not stated in excerpts' as the definition."
    )
    user = (
        f'Question:\n{query}\n\nExcerpts:\n{ctx}\n\n'
        "List important defined terms or acronyms that matter for answering the question. "
        "Each output line MUST be exactly:\nTerm: Definition\n"
        "Use concise definitions. No bullets, no numbering."
    )
    raw = _llm(rag, system, user, temperature=0.1, max_tokens=max_tokens, seed=101, model=model)
    return parse_term_definitions(raw)


def print_prompt_definitions_dict(definitions: dict[str, str]) -> str:
    if not definitions:
        return ""
    lines = ["Relevant definitions (from excerpts only):"]
    for term, definition in definitions.items():
        lines.append(f"- {term}: {definition}")
    return "\n".join(lines) + "\n"


@dataclass
class RouterResult:
    follow_up_queries: list[str] = field(default_factory=list)
    sufficient: bool = False


def parse_router_output(text: str) -> RouterResult:
    sufficient = False
    follow: list[str] = []
    lower = text.lower()
    if re.search(r"sufficient:\s*yes", lower):
        sufficient = True
    elif re.search(r"sufficient:\s*no", lower):
        sufficient = False
    else:
        # default conservative: ask for more if model was ambiguous
        sufficient = "follow_up" not in lower and "follow-up" not in lower

    m = re.search(r"follow[_\s-]*up[_\s-]*queries\s*:([\s\S]*)", text, re.IGNORECASE)
    body = m.group(1) if m else text
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("-") or s.startswith("*"):
            q = s.lstrip("-*").strip()
            if len(q) > 8:
                follow.append(q)
        elif s.lower().startswith("query:"):
            q = s.split(":", 1)[-1].strip()
            if len(q) > 8:
                follow.append(q)
    follow = follow[:6]
    ok = sufficient and not follow
    return RouterResult(follow_up_queries=follow, sufficient=ok)


def router_agent(
    rag: LegalDocumentRAG,
    query: str,
    nodes: list[tuple[Chunk, float]],
    *,
    max_tokens: int | None,
    model: str | None,
) -> RouterResult:
    """
    Decide whether current chunks suffice, or emit follow-up hybrid-search strings
    (analogous to footer / cross-reference fetches).
    """
    block = format_nodes_block(nodes)
    system = (
        "You coordinate retrieval over legal text chunks. Each chunk has a `node_id`. "
        "If the chunks already contain enough to answer the user question, say SUFFICIENT: yes. "
        "If cross-references or missing sections are needed, say SUFFICIENT: no and list "
        "FOLLOW_UP_QUERIES as short natural-language search queries (one per line, prefixed with '- '). "
        "Only add queries that plausibly retrieve missing material; max 4 queries."
    )
    user = (
        f"Question:\n{query}\n\nRetrieved chunks:\n{block}\n\n"
        "Respond with:\nSUFFICIENT: yes|no\n"
        "FOLLOW_UP_QUERIES:\n- ...\n(or leave follow-ups empty if sufficient)."
    )
    raw = _llm(rag, system, user, temperature=0.2, max_tokens=max_tokens, seed=202, model=model)
    return parse_router_output(raw)


def supervisor_agent(
    rag: LegalDocumentRAG,
    query: str,
    *,
    pass_count: int,
    max_passes: int,
    search_failures: list[str],
    context_char_estimate: int,
    max_context_chars: int,
    max_tokens: int | None,
    model: str | None,
) -> str:
    """
    Returns 'END' or 'CONTINUE'. Combines cheap guards with a small-model verdict.
    """
    if pass_count >= max_passes:
        return "END"
    if context_char_estimate > max_context_chars * 1.15:
        return "END"
    fails_tail = search_failures[-8:]
    if len(fails_tail) >= 5 and len(set(fails_tail)) <= 2:
        return "END"

    system = (
        "You are a supervisor for iterative legal retrieval (no graphs). "
        "If repeated failures suggest no more useful material, reply END. "
        "If another retrieval pass could still help, reply CONTINUE. "
        "Reply with a single word: END or CONTINUE."
    )
    user = (
        f"Question:\n{query}\n\nPass: {pass_count}/{max_passes}\n"
        f"Approx context chars: {context_char_estimate} (budget {max_context_chars}).\n"
        f"Recent failures:\n{os.linesep.join(fails_tail) or '(none)'}\n"
    )
    raw = _llm(rag, system, user, temperature=0.0, max_tokens=32, seed=303, model=model).upper()
    first = raw.split()[0] if raw.split() else ""
    if first.startswith("END") or raw.strip() == "END":
        return "END"
    if "CONTINUE" in first or raw.strip().startswith("CONTINUE"):
        return "CONTINUE"
    return "END" if pass_count >= max_passes - 1 else "CONTINUE"


def _parse_keep_node_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for m in re.finditer(r"`([^`]+)`", text):
        s = m.group(1).strip()
        if "::" in s:
            ids.add(s)
    return ids


def prune_nodes(
    rag: LegalDocumentRAG,
    query: str,
    search_query: str,
    nodes: list[tuple[Chunk, float]],
    *,
    max_tokens: int | None,
    model: str | None,
    max_keep: int = 8,
) -> list[tuple[Chunk, float]]:
    """LLM keeps only relevant chunks for this follow-up (analogous to prune_nodes)."""
    if len(nodes) <= max_keep:
        return nodes
    block = format_nodes_block(nodes)
    system = "You filter legal chunks for relevance to a search query. Output ONLY a comma-separated list of node_ids in backticks."
    user = (
        f"User question:\n{query}\nFollow-up search:\n{search_query}\n\nChunks:\n{block}\n\n"
        f"Keep at most {max_keep} node_ids, most relevant first."
    )
    raw = _llm(rag, system, user, temperature=0.0, max_tokens=256, seed=404, model=model)
    keep = _parse_keep_node_ids(raw)
    if not keep:
        return nodes[:max_keep]
    filtered = [pair for pair in nodes if chunk_node_id(pair[0]) in keep]
    if len(filtered) < 3:
        return nodes[:max_keep]
    return filtered[:max_keep]


def merge_nodes(
    acc: dict[str, tuple[Chunk, float]],
    new_pairs: Iterable[tuple[Chunk, float]],
) -> None:
    for ch, sc in new_pairs:
        nid = chunk_node_id(ch)
        if nid not in acc or sc > acc[nid][1]:
            acc[nid] = (ch, sc)


def recursive_retrieval_agent(
    rag: LegalDocumentRAG,
    query: str,
    follow_up_queries: list[str],
    search_failures: list[str],
    *,
    top_k: int | None,
    max_tokens: int | None,
    model: str | None,
) -> list[tuple[Chunk, float]]:
    """Run hybrid retrieval for each follow-up; prune; log empty results."""
    gathered: list[tuple[Chunk, float]] = []
    for fq in follow_up_queries[:4]:
        got = rag.retrieve(fq, top_k=top_k or 10)
        if not got:
            search_failures.append(f"Failed to fetch chunks for follow-up: {fq}")
            continue
        pruned = prune_nodes(rag, query, fq, got, max_tokens=max_tokens, model=model)
        gathered.extend(pruned)
    return gathered


def answering_agent(
    rag: LegalDocumentRAG,
    query: str,
    definitions: dict[str, str],
    nodes: list[tuple[Chunk, float]],
    *,
    max_tokens: int | None,
    model: str | None,
) -> str:
    defs_block = print_prompt_definitions_dict(definitions)
    ctx = build_context_block(nodes, max_chars=14000)
    system = (
        "You are the answering agent. Use ONLY the provided chunks to answer. "
        "Cite sections or paragraph numbers when visible in the text; cite [Source n] for excerpt numbering. "
        "Do not print internal node_id hashes or backticks."
    )
    user = (
        f"{defs_block}\nQuestion:\n{query}\n\nRetrieved document chunks:\n{ctx}\n\n"
        "Write the final answer."
    )
    return _llm(rag, system, user, temperature=0.15, max_tokens=max_tokens, seed=505, model=model)


@dataclass
class MultiAgentConfig:
    max_passes: int = 4
    max_context_chars: int = 28000
    initial_top_k: int = 12
    follow_up_top_k: int = 10


def run_multi_agent(
    rag: LegalDocumentRAG,
    query: str,
    *,
    cfg: MultiAgentConfig | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    cfg = cfg or MultiAgentConfig()
    if rag.chunk_count == 0:
        raise RuntimeError(f"No documents in {rag.documents_dir}")

    definitions = definitions_agent(rag, query, max_tokens=max_tokens, model=model)

    acc: dict[str, tuple[Chunk, float]] = {}
    initial = rag.retrieve(query, top_k=cfg.initial_top_k)
    merge_nodes(acc, initial)

    search_failures: list[str] = []
    pass_count = 0

    while pass_count < cfg.max_passes:
        ordered = sorted(acc.values(), key=lambda x: -x[1])
        routing = router_agent(rag, query, ordered, max_tokens=max_tokens, model=model)
        if routing.sufficient:
            break

        ctx_chars = len(build_context_block(ordered, max_chars=10**9))
        sup = supervisor_agent(
            rag,
            query,
            pass_count=pass_count,
            max_passes=cfg.max_passes,
            search_failures=search_failures,
            context_char_estimate=ctx_chars,
            max_context_chars=cfg.max_context_chars,
            max_tokens=max_tokens,
            model=model,
        )
        if sup == "END":
            break
        if not routing.follow_up_queries:
            break

        extra = recursive_retrieval_agent(
            rag,
            query,
            routing.follow_up_queries,
            search_failures,
            top_k=cfg.follow_up_top_k,
            max_tokens=max_tokens,
            model=model,
        )
        before = len(acc)
        merge_nodes(acc, extra)
        if len(acc) == before:
            search_failures.append("No new chunks merged after follow-up retrieval.")
        pass_count += 1

    final_nodes = sorted(acc.values(), key=lambda x: -x[1])
    # hard context budget: keep highest-scoring chunks until char cap
    pruned_for_answer: list[tuple[Chunk, float]] = []
    used = 0
    for pair in final_nodes:
        block = f"{pair[0].text}\n"
        if used + len(block) > cfg.max_context_chars:
            break
        pruned_for_answer.append(pair)
        used += len(block)
    if not pruned_for_answer:
        pruned_for_answer = final_nodes[:8]

    return answering_agent(
        rag,
        query,
        definitions,
        pruned_for_answer,
        max_tokens=max_tokens,
        model=model,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-agent legal RAG (hybrid retrieval + vLLM).")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)
    p.add_argument("--model", type=str, default=None, help=f"vLLM model (default {DEFAULT_VLLM_MODEL})")
    p.add_argument("--top-k", type=int, default=None, help="Top-k for follow-up retrievals (initial uses MultiAgentConfig)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-passes", type=int, default=int(os.getenv("MULTI_AGENT_MAX_PASSES", "4")))
    p.add_argument("--list", action="store_true")
    p.add_argument("--trace", action="store_true", help="Print agent summaries to stderr")
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
        print('\nExample: python multi_agent_rag.py --docs ./documents "Your question?"')
        return 1

    rag = LegalDocumentRAG(documents_dir=args.docs, vllm_model=args.model)
    tk = args.top_k or 12
    cfg = MultiAgentConfig(
        max_passes=max(1, args.max_passes),
        initial_top_k=tk,
        follow_up_top_k=min(tk, 14),
    )

    if args.trace:
        print("[trace] definitions + router + supervisor + recursive + answer", file=sys.stderr)

    try:
        out = run_multi_agent(
            rag,
            args.question,
            cfg=cfg,
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
