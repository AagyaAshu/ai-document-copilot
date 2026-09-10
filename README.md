# Document Copilot

An internal AI chatbot that lets analysts ask questions about a pile of documents in plain English, and get answers that are grounded in — and cited from — the real source pages.

## The client

**Driftwood Capital** — a fictional independent investment research firm. Their analysts sell deep research to hedge funds and pension funds, and to do that, they read hundreds of pages of SEC filings (10-Ks and 10-Qs) every week just to find the numbers they need. That reading and copy-pasting eats up about half of an analyst's week, before they can even start their real job: the actual analysis.

Document Copilot removes that intake work. An analyst can ask something like:

> "Across 2021 to 2025, how did the revenue mix between iPhone, Services, Mac, iPad, and Wearables change for Apple?"

...and get a straight answer with a citation pointing to the exact filing and passage it came from — so it can always be fact-checked.

Full client brief: [`docs/client-brief.md`](docs/client-brief.md)

## Why this matters (and why not just use ChatGPT)

Two reasons companies build tools like this instead of just handing everyone ChatGPT:

1. **Privacy and compliance.** Most companies can't let employees paste sensitive documents into a public AI tool. The data has to stay inside their own systems.
2. **Consistency.** A generic AI tool gives five different analysts five different answers to the same question. A purpose-built tool standardizes how questions get answered, and makes sure every answer is grounded in the same set of documents the same way.

## What it does

- Ingests SEC filings (10-Ks) for five companies — Apple, Microsoft, NVIDIA, Amazon, Google — across the last five years
- Splits each filing into searchable chunks and stores them with vector embeddings in Postgres
- Answers questions using **hybrid search**: semantic (vector) search + keyword (full-text) search, combined
- Uses an AI agent that can search, read chunks, and read surrounding context — it decides for itself whether it has enough evidence before answering
- **Never answers without a citation.** If the agent can't find solid evidence, it says so instead of guessing
- A separate grounding check re-verifies every citation against the original text before the answer is shown to the user
- Has real user accounts (via Supabase Auth), chat history, and multiple conversation threads — just like a real internal tool

## Stack

| Layer | Choice |
|---|---|
| Backend | Python + FastAPI |
| Frontend | Vite + React SPA + TypeScript |
| Database | Supabase Postgres (users, chats, documents, chunks) |
| Migrations | SQLAlchemy models + Alembic |
| Retrieval | Supabase `pgvector` + Postgres full-text search |
| Auth | Supabase Auth (email only) |
| LLM + Embeddings | **Google Gemini** (`gemini-3.6-flash` for the agent, `gemini-embedding-001` for embeddings) |
| Document parsing | Docling (HTML → Markdown, table extraction) |

## How it works, in plain terms

```
User asks a question
        ↓
Agent decides: do I need to search, or can I answer directly?
        ↓
Search tool: turns the question into an embedding + keywords,
runs both searches in Supabase, combines and ranks the results
        ↓
Agent reads the retrieved passages (and can pull in neighboring
chunks or go back and search again if it needs more context)
        ↓
Agent writes an answer with citations pointing to specific chunks
        ↓
Grounding check: re-reads each citation and confirms the answer
text is actually supported by it — rejects anything that isn't
        ↓
Answer + citations shown to the user, with a link back to the
exact source passage
```

## Setup

See the guides in `docs/guides/` for step-by-step instructions:
- [`backend-setup.md`](docs/guides/backend-setup.md)
- [`frontend-setup.md`](docs/guides/frontend-setup.md)
- [`supabase-setup.md`](docs/guides/supabase-setup.md)

Quick version:
1. Set up a free Supabase project (Postgres + Auth)
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
3. Copy `backend/.env.example` → `backend/.env` and `frontend/.env.example` → `frontend/.env`, fill in your keys
4. `cd backend && uv sync && uv run alembic upgrade head`
5. Download sample filings: `uv run data/download.py` (from the repo root)
6. Convert to markdown: `uv run data/convert_to_markdown.py`
7. Load documents: `uv run python -m ingest.load_source_documents`
8. Chunk + embed: `uv run python -m ingest.chunk_and_embed --all`
9. Run the backend: `uv run uvicorn app.main:app --reload`
10. Run the frontend: `cd frontend && pnpm install && pnpm dev`

## Current data coverage

Currently indexed: Apple 10-Ks (2021–2025), Microsoft 10-Ks (2024–2025), with the remaining Microsoft, NVIDIA, Amazon, and Google filings pending — ingestion is limited by the free tier of the Gemini API, which caps embedding requests per day.

## Notes on this build

The backend originally used OpenAI for the LLM and embeddings; it's been migrated here to run entirely on Google Gemini instead, including:

- `PydanticAI`'s native `GoogleModel` for the agent (tool calling, structured output)
- `google-genai`'s embedding API for both document ingestion and query embeddings
- A custom retry/backoff layer to handle Gemini's free-tier rate limits during ingestion

## Out of scope

No trading recommendations, no external data beyond the ingested filings, no multi-tenant setup, no billing, no mobile app.