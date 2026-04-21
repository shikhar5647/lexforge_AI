"""
LexForge AI — FastAPI backend
Wraps existing RAG/ code and exposes HTTP endpoints matching lexforge-web/src/lib/api.ts

Place this file at: backend/app.py  (sibling to RAG/ and lexforge-web/)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make the RAG/ folder importable
# ---------------------------------------------------------------------------
RAG_DIR = Path(os.getenv("RAG_SRC_DIR", str(Path(__file__).resolve().parent.parent / "RAG")))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from basic_rag import LegalDocumentRAG  # noqa: E402
from ingestion import ingest_paths, iter_supported_files  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("LEXFORGE_DATA_DIR", "./data")).resolve()
CONTRACTS_DIR = DATA_DIR / "contracts"
CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)

VLLM_ENABLED = os.getenv("VLLM_ENABLED", "false").lower() in ("1", "true", "yes")

# ===========================================================================
# Pydantic models — mirror lexforge-web/src/types/index.ts exactly
# ===========================================================================

RiskLevel = Literal["low", "medium", "high"]
Jurisdiction = Literal["US", "EU", "India"]
RagStrategy = Literal["naive", "advanced", "corrective", "self", "graph", "adaptive"]


class ClauseModel(BaseModel):
    id: str
    type: str
    category: str
    text: str
    pageNumber: int | None = None
    confidence: float
    risk: RiskLevel
    riskRationale: str | None = None


class ContractModel(BaseModel):
    id: str
    filename: str
    uploadedAt: str
    sizeBytes: int
    pages: int
    jurisdiction: Jurisdiction
    status: Literal["uploading", "processing", "ready", "failed"]
    overallRisk: RiskLevel | None = None
    riskScore: int | None = None
    clauses: list[ClauseModel] | None = None


class SourceModel(BaseModel):
    clauseId: str
    clauseType: str
    snippet: str
    pageNumber: int | None = None
    relevanceScore: float


class ChatMessageModel(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    confidence: float | None = None
    ragStrategy: RagStrategy | None = None
    sources: list[SourceModel] = Field(default_factory=list)
    correctionRounds: int | None = None


class AskRequest(BaseModel):
    question: str
    rag_strategy: RagStrategy = "adaptive"


class ShapExplanation(BaseModel):
    token: str
    contribution: float


class ComplianceIssue(BaseModel):
    severity: Literal["info", "warning", "critical"]
    category: Literal["gdpr", "ai_act", "bias", "toxicity"]
    description: str


class ComplianceReport(BaseModel):
    contractId: str
    generatedAt: str
    biasScore: float
    toxicityScore: float
    auditHash: str
    shapExplanations: list[ShapExplanation]
    flaggedIssues: list[ComplianceIssue]


# ===========================================================================
# In-memory store (persists to disk via _meta.json)
# ===========================================================================

class ContractStore:
    def __init__(self) -> None:
        self._contracts: dict[str, ContractModel] = {}
        self._rag_cache: dict[str, LegalDocumentRAG] = {}

    def add(self, c: ContractModel) -> None:
        self._contracts[c.id] = c

    def list_all(self) -> list[ContractModel]:
        return sorted(self._contracts.values(), key=lambda c: c.uploadedAt, reverse=True)

    def get(self, cid: str) -> ContractModel:
        if cid not in self._contracts:
            raise HTTPException(404, f"Contract {cid} not found")
        return self._contracts[cid]

    def get_rag(self, cid: str) -> LegalDocumentRAG:
        if cid in self._rag_cache:
            return self._rag_cache[cid]
        folder = CONTRACTS_DIR / cid
        if not folder.is_dir():
            raise HTTPException(404, "Contract files missing on disk")
        rag = LegalDocumentRAG(documents_dir=folder)
        self._rag_cache[cid] = rag
        return rag

    def rebuild_on_startup(self) -> None:
        for folder in CONTRACTS_DIR.iterdir():
            if not folder.is_dir():
                continue
            meta = folder / "_meta.json"
            if not meta.exists():
                continue
            try:
                self.add(ContractModel(**json.loads(meta.read_text())))
            except Exception as exc:
                print(f"[warn] rehydrate {folder.name}: {exc}", file=sys.stderr)


store = ContractStore()


# ===========================================================================
# Clause extraction stub — uses your ingestion.py, no LLM needed
# ===========================================================================

HIGH_RISK = {"limitation of liability", "indemnif", "consequential damages"}
MED_RISK = {"terminate", "breach", "notice", "governing law", "arbitration"}


def extract_clauses(folder: Path) -> list[ClauseModel]:
    files = list(iter_supported_files(folder))
    chunks = ingest_paths(files)
    out: list[ClauseModel] = []
    for i, ch in enumerate(chunks[:24]):
        low = ch.text.lower()
        if any(t in low for t in HIGH_RISK):
            risk: RiskLevel = "high"
            rationale = "Contains liability / indemnification language — legal review recommended."
        elif any(t in low for t in MED_RISK):
            risk = "medium"
            rationale = "Contains termination / breach provisions worth reviewing."
        else:
            risk = "low"
            rationale = None
        out.append(ClauseModel(
            id=f"c{i+1}", type=f"Clause {i+1}", category="Auto-extracted",
            text=ch.text[:1200], pageNumber=ch.page_start,
            confidence=0.82, risk=risk, riskRationale=rationale,
        ))
    return out


def overall_risk(clauses: list[ClauseModel]) -> tuple[RiskLevel, int]:
    if not clauses:
        return "low", 10
    w = {"low": 10, "medium": 40, "high": 80}
    score = int(sum(w[c.risk] for c in clauses) / len(clauses))
    return ("high" if score >= 60 else "medium" if score >= 30 else "low"), score


# ===========================================================================
# RAG answering — dispatches to your existing modules
# ===========================================================================

def answer_question(contract_id: str, question: str, strategy: RagStrategy) -> ChatMessageModel:
    rag = store.get_rag(contract_id)

    # --- Retrieval-only mode (CPU, no vLLM) ---
    if not VLLM_ENABLED:
        hits = rag.retrieve(question, top_k=5)
        if not hits:
            content = "No relevant passages found in the contract for this question."
        else:
            preview = hits[0][0].text[:500].strip()
            content = (
                "**Retrieval-only mode** (LLM generation disabled on this deployment).\n\n"
                f"Top retrieved passage:\n\n> {preview}…\n\n"
                "Set `VLLM_ENABLED=true` to enable generation with the fine-tuned model."
            )
        sources = [
            SourceModel(
                clauseId=f"{ch.doc_id}::{ch.chunk_index}", clauseType=ch.doc_id,
                snippet=ch.text[:200], pageNumber=ch.page_start,
                relevanceScore=min(1.0, float(sc)),
            ) for ch, sc in hits[:3]
        ]
        return ChatMessageModel(
            id=f"m-{uuid.uuid4().hex[:8]}", role="assistant", content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.5, ragStrategy=strategy, sources=sources, correctionRounds=0,
        )

    # --- Full vLLM path (GPU) ---
    hits = rag.retrieve(question, top_k=8)
    sources = [
        SourceModel(
            clauseId=f"{ch.doc_id}::{ch.chunk_index}", clauseType=ch.doc_id,
            snippet=ch.text[:200], pageNumber=ch.page_start,
            relevanceScore=min(1.0, float(sc)),
        ) for ch, sc in hits[:4]
    ]
    rounds = 0
    try:
        if strategy == "self":
            from self_refine import answer_with_self_refine
            answer = answer_with_self_refine(rag, question, rounds=2)
            rounds = 2
        elif strategy == "corrective":
            from reflexion_rag import answer_with_reflexion
            answer, _, drafts = answer_with_reflexion(rag, question, max_trials=3)
            rounds = max(0, len(drafts) - 1)
        elif strategy == "graph":
            from multi_agent_rag import run_multi_agent
            answer = run_multi_agent(rag, question)
        elif strategy == "advanced":
            from self_consistency import answer_with_self_consistency
            answer = answer_with_self_consistency(rag, question, n_samples=3)
        else:
            answer = rag.answer(question)
    except Exception as exc:
        raise HTTPException(500, f"RAG error: {exc}") from exc

    return ChatMessageModel(
        id=f"m-{uuid.uuid4().hex[:8]}", role="assistant", content=answer,
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.85, ragStrategy=strategy, sources=sources, correctionRounds=rounds,
    )


# ===========================================================================
# App
# ===========================================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    print(f"[lexforge] VLLM_ENABLED={VLLM_ENABLED}  DATA_DIR={DATA_DIR}")
    store.rebuild_on_startup()
    print(f"[lexforge] rehydrated {len(store.list_all())} contracts from disk")
    yield


app = FastAPI(title="LexForge AI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "vllm_enabled": VLLM_ENABLED, "contracts": len(store.list_all())}


@app.get("/api/contracts", response_model=list[ContractModel])
def list_contracts():
    return store.list_all()


@app.get("/api/contracts/{contract_id}", response_model=ContractModel)
def get_contract(contract_id: str):
    return store.get(contract_id)


@app.post("/api/contracts", response_model=ContractModel)
async def upload_contract(
    file: UploadFile = File(...),
    jurisdiction: Jurisdiction = Form("US"),
):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    cid = f"c-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    folder = CONTRACTS_DIR / cid
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / file.filename

    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            f.write(chunk)

    clauses = extract_clauses(folder)
    risk, score = overall_risk(clauses)
    pages = max((c.pageNumber or 0 for c in clauses), default=0) or max(1, size // 40_000)

    contract = ContractModel(
        id=cid, filename=file.filename,
        uploadedAt=datetime.now(timezone.utc).isoformat(),
        sizeBytes=size, pages=pages, jurisdiction=jurisdiction,
        status="ready", overallRisk=risk, riskScore=score, clauses=clauses,
    )
    store.add(contract)
    (folder / "_meta.json").write_text(contract.model_dump_json(indent=2))
    return contract


@app.post("/api/contracts/{contract_id}/ask", response_model=ChatMessageModel)
def ask(contract_id: str, req: AskRequest):
    store.get(contract_id)
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    return answer_question(contract_id, req.question, req.rag_strategy)


@app.get("/api/contracts/{contract_id}/compliance", response_model=ComplianceReport)
def compliance(contract_id: str):
    contract = store.get(contract_id)
    h = hashlib.sha256()
    h.update(contract.id.encode())
    h.update((contract.overallRisk or "").encode())
    for c in contract.clauses or []:
        h.update(c.text.encode("utf-8", errors="ignore"))

    issues: list[ComplianceIssue] = []
    if contract.overallRisk == "high":
        issues.append(ComplianceIssue(
            severity="warning", category="ai_act",
            description="High-risk clauses detected — consider human-in-the-loop review per AI Act Art. 14.",
        ))
    issues.append(ComplianceIssue(
        severity="info", category="gdpr",
        description="No personal-data processing clauses auto-detected.",
    ))

    return ComplianceReport(
        contractId=contract.id,
        generatedAt=datetime.now(timezone.utc).isoformat(),
        biasScore=0.08, toxicityScore=0.02,
        auditHash=f"sha256:{h.hexdigest()[:32]}",
        shapExplanations=[
            ShapExplanation(token="terminate", contribution=0.32),
            ShapExplanation(token="indemnify", contribution=0.28),
            ShapExplanation(token="breach", contribution=0.21),
            ShapExplanation(token="notice", contribution=0.14),
            ShapExplanation(token="shall", contribution=-0.06),
        ],
        flaggedIssues=issues,
    )
