# LexForge AI — Web Frontend

React + Vite + TypeScript + TailwindCSS frontend for the LexForge AI legal intelligence platform.

## Setup

```bash
cd lexforge-web
npm install
npm run dev
```

Opens at http://localhost:3000.

## Project structure

```
src/
├── types/
│   └── index.ts              # Shared TypeScript types (API contract)
├── lib/
│   ├── api.ts                # ⭐ API client — flip USE_MOCK_API to false when backend is ready
│   └── utils.ts              # Formatting helpers, risk styles
├── components/
│   ├── PageHeader.tsx        # Reusable page title bar
│   └── RiskBadge.tsx         # Low/Medium/High risk chip
├── pages/
│   ├── DashboardPage.tsx     # Landing page — contract list + stats
│   ├── UploadPage.tsx        # Drag-and-drop contract upload
│   ├── AnalysisPage.tsx      # Clause list + risk pie chart + detail drawer
│   ├── ChatPage.tsx          # RAG Q&A with citations + strategy selector
│   └── CompliancePage.tsx    # SHAP viz + audit hash + flagged issues
├── App.tsx                   # Router + sidebar layout
├── main.tsx                  # Entry point
└── index.css                 # Tailwind + custom utility classes
```

## Wiring up the real backend

The whole frontend talks to the backend through **one file**: `src/lib/api.ts`.

Right now `USE_MOCK_API = true` at the top, which means every API call returns fake data. When your FastAPI is ready:

1. Open `src/lib/api.ts`
2. Change `USE_MOCK_API = true` → `USE_MOCK_API = false`
3. Implement these endpoints in FastAPI (see the function bodies in `api.ts` for the exact request/response shapes):

| Frontend function            | HTTP call                                      |
|------------------------------|------------------------------------------------|
| `listContracts()`            | `GET  /api/contracts`                          |
| `getContract(id)`            | `GET  /api/contracts/:id`                      |
| `uploadContract(file, juri)` | `POST /api/contracts` (multipart)              |
| `askQuestion(id, q, rag)`    | `POST /api/contracts/:id/ask` (JSON)           |
| `getComplianceReport(id)`    | `GET  /api/contracts/:id/compliance`           |

Vite's dev server already proxies `/api/*` to `http://localhost:8000` (your FastAPI), so CORS isn't an issue in development.

## How it maps to the LexForge spec

| Spec component                       | Frontend implementation                              |
|--------------------------------------|------------------------------------------------------|
| PDF upload + ingestion               | `UploadPage.tsx` (dropzone, jurisdiction selector)   |
| 41-clause extraction                 | `AnalysisPage.tsx` (clause list + detail drawer)     |
| Risk scoring (Low/Medium/High)       | `RiskBadge.tsx` + risk distribution pie chart        |
| 6-strategy RAG pipeline              | `ChatPage.tsx` (strategy selector in header)         |
| Self-correction loop                 | Displayed via `correctionRounds` in chat metadata    |
| NLI + ROUGE-L confidence score       | Displayed per-message in chat                        |
| SHAP token explainability            | `CompliancePage.tsx` (diverging bar chart)           |
| SHA-256 tamper-proof audit log       | `CompliancePage.tsx` (copy-to-clipboard hash)        |
| GDPR / AI Act flagged issues         | `CompliancePage.tsx` (severity-coded issue list)     |
| LoRA hot-swapping per jurisdiction   | Jurisdiction selector in `UploadPage.tsx`            |

## What's deliberately not included yet

- **Authentication / Keycloak SSO** — add when the backend has it. Drop an auth context into `main.tsx` and route-guard in `App.tsx`.
- **WebSocket streaming** — right now chat waits for the full response. When your FastAPI streams, swap `askQuestion` in `api.ts` to use an `EventSource` or fetch stream.
- **Real file upload progress** — mock shows an immediate spinner; add `axios` with `onUploadProgress` if you want a progress bar.
- **i18n** — all strings are English literals. Extract to `src/locales/` when you need multi-language.

## Customization

**Colors / branding** live in `tailwind.config.js` under `theme.extend.colors.brand`. Change the hex values and the whole app re-skins.

**The RAG strategies** shown in the chat dropdown live in `ChatPage.tsx` — match them to whatever your backend actually supports.

## Build for production

```bash
npm run build      # outputs to dist/
npm run preview    # test the production build locally
```

The `dist/` folder can be served by any static host (Nginx, Vercel, S3+CloudFront) or shipped inside the FastAPI container.
