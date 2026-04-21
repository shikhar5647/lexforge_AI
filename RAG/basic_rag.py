

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from transformers import AutoTokenizer

from ingestion import Chunk, ingest_paths, iter_supported_files
from retrieval import HybridLegalRetriever, HybridRetrieverConfig

if TYPE_CHECKING:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
else:
    VLLMEngine = Any


DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "documents"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


SYSTEM_PROMPT = """You are a legal research assistant. Use ONLY the provided excerpts from the user's documents to answer. If the excerpts do not contain enough information, say clearly that the materials are insufficient and list what is missing.

Rules:
- Prefer precise quotations or close paraphrases tied to the cited source numbers.
- When citing, use bracketed references like [Source 1] matching the excerpt numbering.
- Do not invent statutes, case names, dates, or clauses that are not in the excerpts.
- If excerpts conflict, acknowledge the conflict briefly.
"""


def build_context_block(
    retrieved: list[tuple[Chunk, float]],
    max_chars: int = 12000,
) -> str:
    parts: list[str] = []
    used = 0
    for i, (chunk, score) in enumerate(retrieved, 1):
        meta = chunk.doc_id
        if getattr(chunk, "page_start", None) is not None:
            pe = chunk.page_end
            ps = chunk.page_start
            meta += f" (pages {ps}" + (f"-{pe}" if pe != ps else "") + ")"
        header = f"[Source {i}: {meta}; cross_encoder_score={score:.3f}]"
        block = f"{header}\n{chunk.text}"
        if used + len(block) > max_chars:
            remain = max_chars - used - len(header) - 20
            if remain < 200:
                break
            block = f"{header}\n{chunk.text[:remain]}…"
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


class LegalDocumentRAG:
    def __init__(
        self,
        documents_dir: Path | str | None = None,
        *,
        retriever_config: HybridRetrieverConfig | None = None,
        vllm_model: str | None = None,
    ):
        load_dotenv()
        self.documents_dir = Path(documents_dir or DEFAULT_DOCS_DIR).expanduser().resolve()
        self._retriever_config = retriever_config
        self._retriever: HybridLegalRetriever | None = None
        self._chunks = ingest_paths(list(iter_supported_files(self.documents_dir)))
        if self._chunks:
            self._retriever = HybridLegalRetriever(self._chunks, self._retriever_config)

        self._vllm_model = (vllm_model or os.getenv("VLLM_MODEL", DEFAULT_VLLM_MODEL)).strip()
        self._llm: VLLMEngine | None = None
        self._tokenizer: Any = None

    def _ensure_vllm(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        max_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        dtype = os.getenv("VLLM_DTYPE", "auto")
        self._tokenizer = AutoTokenizer.from_pretrained(self._vllm_model, trust_remote_code=True)
        self._llm = LLM(
            model=self._vllm_model,
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=gpu_mem,
            dtype=dtype,
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def reload(self) -> None:
        self._chunks = ingest_paths(list(iter_supported_files(self.documents_dir)))
        self._retriever = (
            HybridLegalRetriever(self._chunks, self._retriever_config) if self._chunks else None
        )

    def retrieve(self, query: str, top_k: int | None = None):
        if not self._retriever:
            return []
        cfg = self._retriever.config
        k = top_k or cfg.rerank_topk
        return self._retriever.retrieve(query, rerank_topk=k)

    def answer(
        self,
        question: str,
        *,
        model: str | None = None,
        top_k: int | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        if not self._retriever:
            raise RuntimeError(
                f"No documents found in {self.documents_dir}. Add .pdf, .docx, or .txt files."
            )
        hits = self.retrieve(question, top_k=top_k)
        context = build_context_block(hits)

        if model is not None:
            m = model.strip()
            if m and m != self._vllm_model:
                self._vllm_model = m
                self._llm = None
                self._tokenizer = None

        self._ensure_vllm()
        assert self._llm is not None and self._tokenizer is not None

        from vllm import SamplingParams

        user_content = (
            f"Question:\n{question}\n\nDocument excerpts:\n{context}\n\n"
            "Answer the question using the excerpts and cite [Source n] where appropriate."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mt = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "1024"))
        params = SamplingParams(temperature=temperature, max_tokens=mt)
        outputs = self._llm.generate([prompt], params)
        return (outputs[0].outputs[0].text or "").strip()

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        seed: int | None = None,
        top_p: float = 1.0,
    ) -> str:
        """Single vLLM completion from chat messages (for self-consistency / self-refine scripts)."""
        if model is not None:
            m = model.strip()
            if m and m != self._vllm_model:
                self._vllm_model = m
                self._llm = None
                self._tokenizer = None

        self._ensure_vllm()
        assert self._llm is not None and self._tokenizer is not None

        from vllm import SamplingParams

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mt = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "1024"))
        kwargs: dict = {"temperature": temperature, "max_tokens": mt, "top_p": top_p}
        if seed is not None:
            kwargs["seed"] = seed
        params = SamplingParams(**kwargs)
        outputs = self._llm.generate([prompt], params)
        return (outputs[0].outputs[0].text or "").strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Legal hybrid RAG (BM25 + dense + lexical + RRF + rerank).")
    p.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Folder containing 3–5 legal documents (.pdf, .docx, .txt)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List ingested files and chunk counts, then exit",
    )
    p.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print retrieved chunks without calling the LLM",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"vLLM HF model id (default: VLLM_MODEL env or {DEFAULT_VLLM_MODEL})",
    )
    p.add_argument("--top-k", type=int, default=None, help="Chunks after reranking to send to the LLM")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Generation cap (default: VLLM_MAX_TOKENS or 1024)",
    )
    p.add_argument("question", nargs="?", default=None, help="User question")
    args = p.parse_args(argv)

    if args.list:
        docs_path = Path(args.docs).expanduser().resolve()
        files = list(iter_supported_files(docs_path))
        chunks = ingest_paths(files)
        print(f"Documents dir: {docs_path}")
        print(f"Files: {len(files)}")
        for f in files:
            print(f"  - {f.name}")
        print(f"Total chunks: {len(chunks)}")
        return 0

    rag = LegalDocumentRAG(documents_dir=args.docs, vllm_model=args.model)

    if not args.question:
        p.print_help()
        print("\nExample: python basic_rag.py --docs ./documents \"What notice period applies?\"")
        return 1

    if args.retrieve_only:
        for ch, sc in rag.retrieve(args.question, top_k=args.top_k or 8):
            print(f"\n{'='*60}\nscore={sc:.4f} doc={ch.doc_id} pages={ch.page_start}-{ch.page_end}\n{ch.text[:800]}…")
        return 0

    try:
        ans = rag.answer(
            args.question,
            model=args.model,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(ans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
