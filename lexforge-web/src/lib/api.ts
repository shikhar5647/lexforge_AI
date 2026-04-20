// ============================================================================
// LexForge API Client
// ============================================================================
// Single place where UI talks to the backend.
//
// HOW TO WIRE UP THE REAL BACKEND:
//   1. Set USE_MOCK_API = false below
//   2. Implement the corresponding FastAPI endpoints under /api/
//   3. That's it — no UI code changes needed.
// ============================================================================

import type {
  Contract,
  ChatMessage,
  ComplianceReport,
  Clause,
  RiskLevel,
  Jurisdiction,
} from '@/types';

// ----- Toggle this when your FastAPI backend is ready -----
const USE_MOCK_API = true;

const API_BASE = '/api';

// ============================================================================
// MOCK DATA
// ============================================================================

const MOCK_CLAUSES: Clause[] = [
  {
    id: 'c1',
    type: 'Term / Duration',
    category: 'Term',
    text: 'The initial term of this Agreement shall be three (3) years from the Effective Date, and shall automatically renew for successive one (1) year terms unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.',
    pageNumber: 1,
    confidence: 0.94,
    risk: 'low',
    riskRationale: 'Standard auto-renewal clause with reasonable notice period.',
  },
  {
    id: 'c2',
    type: 'Termination for Convenience',
    category: 'Termination',
    text: 'Either party may terminate this Agreement for material breach upon thirty (30) days written notice, provided the breaching party has failed to cure such breach within the notice period.',
    pageNumber: 1,
    confidence: 0.89,
    risk: 'medium',
    riskRationale: 'Cure period is on the shorter end of industry norms (30 days vs. typical 60).',
  },
  {
    id: 'c3',
    type: 'Limitation of Liability',
    category: 'Liability',
    text: 'In no event shall either party be liable for any indirect, incidental, special, consequential or punitive damages, regardless of the theory of liability.',
    pageNumber: 4,
    confidence: 0.91,
    risk: 'high',
    riskRationale: 'Unlimited exclusion of consequential damages — unusual and disadvantageous to Client.',
  },
  {
    id: 'c4',
    type: 'Governing Law',
    category: 'Legal',
    text: 'This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles.',
    pageNumber: 6,
    confidence: 0.98,
    risk: 'low',
  },
  {
    id: 'c5',
    type: 'Confidentiality',
    category: 'Information',
    text: 'Each party agrees to hold all Confidential Information in strict confidence for a period of five (5) years following termination of this Agreement.',
    pageNumber: 3,
    confidence: 0.92,
    risk: 'low',
  },
  {
    id: 'c6',
    type: 'Indemnification',
    category: 'Liability',
    text: 'Provider shall indemnify and hold harmless Client from any third-party claims arising out of Provider\'s gross negligence or willful misconduct.',
    pageNumber: 5,
    confidence: 0.87,
    risk: 'medium',
    riskRationale: 'Indemnification limited to gross negligence — does not cover ordinary negligence.',
  },
];

// Mock in-memory contract store
const mockContracts: Map<string, Contract> = new Map();

// Seed with one example contract so the UI isn't empty on first load
mockContracts.set('demo-1', {
  id: 'demo-1',
  filename: 'Acme-Globex-MSA.pdf',
  uploadedAt: new Date(Date.now() - 86400000).toISOString(),
  sizeBytes: 347_291,
  pages: 12,
  jurisdiction: 'US',
  status: 'ready',
  overallRisk: 'medium',
  riskScore: 58,
  clauses: MOCK_CLAUSES,
});

// ----- Helper: simulate network latency in dev -----
const fakeDelay = (ms = 600) => new Promise((r) => setTimeout(r, ms));

// ============================================================================
// CONTRACT ENDPOINTS
// ============================================================================

export async function listContracts(): Promise<Contract[]> {
  if (USE_MOCK_API) {
    await fakeDelay(300);
    return Array.from(mockContracts.values()).sort(
      (a, b) => new Date(b.uploadedAt).getTime() - new Date(a.uploadedAt).getTime()
    );
  }
  const res = await fetch(`${API_BASE}/contracts`);
  if (!res.ok) throw new Error('Failed to fetch contracts');
  return res.json();
}

export async function getContract(id: string): Promise<Contract> {
  if (USE_MOCK_API) {
    await fakeDelay(200);
    const c = mockContracts.get(id);
    if (!c) throw new Error(`Contract ${id} not found`);
    return c;
  }
  const res = await fetch(`${API_BASE}/contracts/${id}`);
  if (!res.ok) throw new Error('Contract not found');
  return res.json();
}

export async function uploadContract(
  file: File,
  jurisdiction: Jurisdiction = 'US'
): Promise<Contract> {
  if (USE_MOCK_API) {
    await fakeDelay(1500);
    const id = `c-${Date.now()}`;
    const overall: RiskLevel = ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as RiskLevel;
    const contract: Contract = {
      id,
      filename: file.name,
      uploadedAt: new Date().toISOString(),
      sizeBytes: file.size,
      pages: Math.ceil(file.size / 40_000),
      jurisdiction,
      status: 'ready',
      overallRisk: overall,
      riskScore: overall === 'low' ? 25 : overall === 'medium' ? 55 : 82,
      clauses: MOCK_CLAUSES,
    };
    mockContracts.set(id, contract);
    return contract;
  }
  const formData = new FormData();
  formData.append('file', file);
  formData.append('jurisdiction', jurisdiction);
  const res = await fetch(`${API_BASE}/contracts`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

// ============================================================================
// CHAT / RAG ENDPOINT
// ============================================================================

export async function askQuestion(
  contractId: string,
  question: string,
  ragStrategy: ChatMessage['ragStrategy'] = 'adaptive'
): Promise<ChatMessage> {
  if (USE_MOCK_API) {
    await fakeDelay(1200);
    const contract = mockContracts.get(contractId);
    const relevantClause = contract?.clauses?.[Math.floor(Math.random() * (contract.clauses?.length || 1))];
    return {
      id: `m-${Date.now()}`,
      role: 'assistant',
      content: `Based on the contract, ${relevantClause?.text.slice(0, 200)}...\n\nThis provision was identified with high confidence and cross-referenced against ${Math.floor(Math.random() * 3) + 2} related clauses before being returned.`,
      timestamp: new Date().toISOString(),
      confidence: 0.82 + Math.random() * 0.15,
      ragStrategy,
      correctionRounds: Math.floor(Math.random() * 3),
      sources: relevantClause
        ? [
            {
              clauseId: relevantClause.id,
              clauseType: relevantClause.type,
              snippet: relevantClause.text.slice(0, 150) + '...',
              pageNumber: relevantClause.pageNumber,
              relevanceScore: 0.89,
            },
          ]
        : [],
    };
  }
  const res = await fetch(`${API_BASE}/contracts/${contractId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, rag_strategy: ragStrategy }),
  });
  if (!res.ok) throw new Error('Query failed');
  return res.json();
}

// ============================================================================
// COMPLIANCE ENDPOINT
// ============================================================================

export async function getComplianceReport(contractId: string): Promise<ComplianceReport> {
  if (USE_MOCK_API) {
    await fakeDelay(500);
    return {
      contractId,
      generatedAt: new Date().toISOString(),
      biasScore: 0.08,
      toxicityScore: 0.02,
      auditHash: 'sha256:' + Array.from({ length: 16 }, () =>
        Math.floor(Math.random() * 16).toString(16)).join(''),
      shapExplanations: [
        { token: 'terminate', contribution: 0.34 },
        { token: 'breach', contribution: 0.28 },
        { token: 'thirty', contribution: 0.21 },
        { token: 'notice', contribution: 0.15 },
        { token: 'cure', contribution: 0.12 },
        { token: 'material', contribution: -0.08 },
      ],
      flaggedIssues: [
        {
          severity: 'warning',
          category: 'ai_act',
          description: 'Automated decision-making disclosure recommended for Section 8 risk scoring.',
        },
        {
          severity: 'info',
          category: 'gdpr',
          description: 'No personal data processing clauses detected in this contract.',
        },
      ],
    };
  }
  const res = await fetch(`${API_BASE}/contracts/${contractId}/compliance`);
  if (!res.ok) throw new Error('Compliance report failed');
  return res.json();
}
