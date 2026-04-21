from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingestion import Chunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """
    RRF: score(i) = sum_j 1 / (k + rank_j(i)). ranked_lists[j] is ordered doc indices.
    """
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    if top_n is not None:
        fused = fused[:top_n]
    return fused


def keyword_overlap_rank(
    query_tokens: set[str],
    corpus_tokens: Sequence[list[str]],
    candidate_indices: Iterable[int] | None = None,
) -> list[int]:
    """Lexical channel: Jaccard-like overlap on content words."""
    if candidate_indices is None:
        candidate_indices = range(len(corpus_tokens))
    scored: list[tuple[int, float]] = []
    q = query_tokens
    if not q:
        return list(candidate_indices)
    for i in candidate_indices:
        ct = set(corpus_tokens[i])
        if not ct:
            scored.append((i, 0.0))
            continue
        inter = len(q & ct)
        union = len(q | ct)
        scored.append((i, inter / union if union else 0.0))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in scored]


@dataclass
class HybridRetrieverConfig:
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    dense_topk: int = 50
    bm25_topk: int = 50
    keyword_topk: int = 50
    fuse_topk: int = 40
    rerank_topk: int = 12
    use_reranker: bool = True


class HybridLegalRetriever:
    def __init__(self, chunks: Sequence[Chunk], config: HybridRetrieverConfig | None = None):
        self.chunks = list(chunks)
        self.config = config or HybridRetrieverConfig()
        self._texts = [c.text for c in self.chunks]
        self._tokenized = [tokenize(t) or ["_"] for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized)

        self._embedder = SentenceTransformer(self.config.embedding_model)
        self._reranker: CrossEncoder | None = None
        if self.config.use_reranker:
            self._reranker = CrossEncoder(self.config.cross_encoder_model)

        doc_embs = self._embed_documents(self._texts)
        doc_embs = self._l2_normalize(doc_embs)
        dim = doc_embs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(doc_embs.astype(np.float32))

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x / norms

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        # BGE v1.5: asymmetric prompts when supported
        try:
            return self._embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
                prompt_name="document",
            )
        except TypeError:
            return self._embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            q = self._embedder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=False,
                prompt_name="query",
            )
        except TypeError:
            q = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        return self._l2_normalize(q.astype(np.float32))

    def dense_rank(self, query: str, topk: int) -> list[int]:
        if not self.chunks:
            return []
        k = min(topk, len(self.chunks))
        q = self._embed_query(query)
        _scores, idx = self._index.search(q, k)
        # IndexFlatIP returns inner products sorted in descending order.
        return [int(i) for i in idx[0] if i >= 0]

    def bm25_rank(self, query: str, topk: int) -> list[int]:
        qtok = tokenize(query)
        if not qtok:
            return list(range(min(topk, len(self.chunks))))
        scores = self._bm25.get_scores(qtok)
        order = np.argsort(-scores)[:topk]
        return [int(i) for i in order]

    def retrieve(
        self,
        query: str,
        *,
        fuse_topk: int | None = None,
        rerank_topk: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        cfg = self.config
        fuse_k = fuse_topk or cfg.fuse_topk
        rr_k = rerank_topk or cfg.rerank_topk

        dense_order = self.dense_rank(query, cfg.dense_topk)
        bm25_order = self.bm25_rank(query, cfg.bm25_topk)
        qset = set(tokenize(query))
        kw_order = keyword_overlap_rank(qset, self._tokenized, candidate_indices=range(len(self.chunks)))
        kw_order = kw_order[: cfg.keyword_topk]

        fused = reciprocal_rank_fusion(
            [dense_order, bm25_order, kw_order],
            k=cfg.rrf_k,
            top_n=fuse_k,
        )
        cand_indices = [i for i, _ in fused]

        if self._reranker is not None and cand_indices:
            pairs = [[query, self._texts[i]] for i in cand_indices]
            ce_scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            order = np.argsort(-np.asarray(ce_scores))
            top_idx = [cand_indices[int(j)] for j in order[:rr_k]]
            scores = {cand_indices[int(j)]: float(ce_scores[int(j)]) for j in order[:rr_k]}
        else:
            top_idx = cand_indices[:rr_k]
            scores = {i: s for i, s in fused if i in set(top_idx)}

        return [(self.chunks[i], scores.get(i, 0.0)) for i in top_idx]
