// ============================================================================
// LexForge API Contract
// ============================================================================
// These types define the shape of data flowing between the React UI and the
// FastAPI backend. Keep this file in sync with your backend Pydantic models.
// ============================================================================

export type RiskLevel = 'low' | 'medium' | 'high';

export type Jurisdiction = 'US' | 'EU' | 'India';

/**
 * A single extracted clause from the contract.
 * Maps to CUAD's 41 clause categories.
 */
export interface Clause {
  id: string;
  type: string;                    // e.g. "Termination for Convenience"
  category: string;                // e.g. "Termination"
  text: string;                    // verbatim clause text
  pageNumber?: number;
  confidence: number;              // 0.0 - 1.0
  risk: RiskLevel;
  riskRationale?: string;
}

/**
 * Represents an uploaded contract and its analysis state.
 */
export interface Contract {
  id: string;
  filename: string;
  uploadedAt: string;              // ISO timestamp
  sizeBytes: number;
  pages: number;
  jurisdiction: Jurisdiction;
  status: 'uploading' | 'processing' | 'ready' | 'failed';
  overallRisk?: RiskLevel;
  riskScore?: number;              // 0-100
  clauses?: Clause[];
}

/**
 * A single turn in the chat interface.
 */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  // Self-correction metadata from the RAG pipeline
  confidence?: number;             // composite NLI + ROUGE-L score
  ragStrategy?: 'naive' | 'advanced' | 'corrective' | 'self' | 'graph' | 'adaptive';
  sources?: Source[];
  correctionRounds?: number;       // how many self-correction loops ran
}

export interface Source {
  clauseId: string;
  clauseType: string;
  snippet: string;
  pageNumber?: number;
  relevanceScore: number;
}

/**
 * Compliance report data per the spec's AI Act / GDPR requirements.
 */
export interface ComplianceReport {
  contractId: string;
  generatedAt: string;
  biasScore: number;               // 0-1, lower is better
  toxicityScore: number;           // 0-1
  auditHash: string;               // SHA-256 of the inference record
  shapExplanations: ShapExplanation[];
  flaggedIssues: ComplianceIssue[];
}

export interface ShapExplanation {
  token: string;
  contribution: number;            // -1.0 to +1.0
}

export interface ComplianceIssue {
  severity: 'info' | 'warning' | 'critical';
  category: 'gdpr' | 'ai_act' | 'bias' | 'toxicity';
  description: string;
}
