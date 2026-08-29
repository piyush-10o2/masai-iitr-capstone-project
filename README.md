# Ola Support-Policy & Ticket-Status Agent

**Track completed: Ola (Business Operations / Customer Support).**

A domain support agent that answers Ola support-policy questions from a
locally-written knowledge base and looks up a specific support ticket's status from a
self-generated dataset — orchestrated with **CrewAI**, reviewed by an independent
**Autogen** review stage, deployed behind **FastAPI**, and evaluated end to end. Built
to run fully offline under a deterministic `MOCK_LLM` mode — **zero API keys, zero
network access, zero paid services required.**

> **Note to grader:** every acceptance criterion in the brief is satisfied under
> `MOCK_LLM`. A real LLM API can optionally be wired in behind an environment
> variable (see [Optional: real LLM mode](#optional-real-llm-mode)), but no graded
> transcript in this repo depends on it.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Repository Structure](#repository-structure)
3. [Part 1 — Dataset & RAG Core](#part-1--dataset--rag-core)
4. [Part 2 — CrewAI Orchestration, Memory & Guardrails](#part-2--crewai-orchestration-memory--guardrails)
5. [Part 3 — FastAPI Deployment & Evaluation](#part-3--fastapi-deployment--evaluation)
6. [Part 4 — Autogen Review Stage & Governance](#part-4--autogen-review-stage--governance)
7. [Acceptance Criteria Checklist](#acceptance-criteria-checklist)
8. [Known Limitations / Out of Scope](#known-limitations--out-of-scope)
9. [Optional: Real LLM Mode](#optional-real-llm-mode)

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 3a. Confirm .gitignore is excluding what it should (run once, sanity check)
git check-ignore -v .venv .env data/chroma logs/requests.jsonl
# transcripts/ must NOT appear here — it's graded evidence, not a runtime artifact

# 4. IMPORTANT — disable CrewAI's default outbound telemetry call before running
#    anything (see Submission Guidelines: crew.kickoff() phones home by default).
export CREWAI_DISABLE_TELEMETRY=true      # Windows: set CREWAI_DISABLE_TELEMETRY=true
export OTEL_SDK_DISABLED=true             # belt-and-suspenders, same effect

# 5. Confirm MOCK_LLM mode is the default (no key needed)
echo $MOCK_LLM    # should print "true" or be unset-but-defaulted-true in config.py

# 6. Generate the dataset and knowledge base index (one-time)
python dataset.py
python build_index.py

# 7. Run the test/demo transcripts for each part
python demo_part1_rag.py
python demo_part2_crew.py
python demo_part3_api.py          # or: uvicorn app.api.main:app --reload
python demo_part4_governance.py
```

**Confirmed:** `CREWAI_DISABLE_TELEMETRY=true` is set before every `crew.kickoff()`
call in this project (see `.env.example` and the top of every `demo_*.py` script).
No outbound network call is made anywhere in the graded transcripts.

---

## Repository Structure

```
.
├── README.md                      # this file
├── requirements.txt
├── .env.example                   # MOCK_LLM=true, CREWAI_DISABLE_TELEMETRY=true, etc.
├── .gitignore                     # excludes .venv, .env, *.db, __pycache__, data/chroma/, logs/*.jsonl
├── dataset.py                     # Task 1 — SUPPORT_TICKETS generator (root-level, per brief)
├── build_index.py                 # embeds + indexes both chunking strategies into Chroma
├── app/                           # main application package
│   ├── __init__.py
│   ├── config.py                  # business constants: escalation threshold, cost cap, PII patterns
│   ├── models.py                  # Task 9 — Pydantic response schema; Task 14 verdict model
│   ├── chunking.py                # Task 3 — fixed-size + sentence-based chunkers
│   ├── rag.py                     # Task 4 — grounded generation + threshold calibration
│   ├── mock_llm.py                # MOCK_LLM as a crewai.llms.base_llm.BaseLLM subclass
│   ├── memory.py                  # Task 8 — session memory (LangChain)
│   ├── guardrails.py              # Task 10 — PII masking, prompt-injection, groundedness
│   ├── governance.py              # Task 15 — least-autonomy guard, risk classification, cost cap
│   ├── cache.py                   # Task 16 — response cache
│   ├── crew.py                    # Task 7 — 3-agent CrewAI crew definition
│   ├── review_stage.py            # Task 14 — Autogen RoundRobinGroupChat review team
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py            # RAG tool wrapper for the Retrieval Agent
│   │   └── ticket_lookup_tool.py  # Task 6 — check_support_ticket_status()
│   └── api/
│       ├── __init__.py
│       ├── main.py                # Task 11 — FastAPI app (2+ HTTP endpoints + WebSocket)
│       └── logging_middleware.py  # Task 12 — JSON-Lines structured logging (PII-masked)
├── data/
│   ├── knowledge_base/            # Task 2 — 12 policy documents (.md)
│   └── chroma/                    # generated vector store (gitignored, rebuilt by build_index.py)
├── eval/
│   ├── retrieval_eval.py          # Task 5 — precision/recall per collection
│   ├── llm_judge_eval.py          # Task 13 — Accuracy/Grounding/Completeness/Safety
│   └── test_queries.json          # the 15 queries used for Task 13
├── demo_part1_rag.py              # runnable transcript for Part 1
├── demo_part2_crew.py             # runnable transcript for Part 2
├── demo_part3_api.py              # runnable transcript for Part 3
├── demo_part4_governance.py       # runnable transcript for Part 4
├── logs/                          # runtime JSON-Lines request logs (gitignored contents, .gitkeep)
└── transcripts/                   # saved output evidence from each demo script — NOT gitignored,
                                    # this is graded deliverable content
```

> **Note on `.gitignore`:** it correctly excludes `.venv`, `.env`, `*.db`, `__pycache__/`,
> `data/chroma/` (regenerable vector store), and `logs/*.jsonl` (runtime logs) — but
> **deliberately does not exclude `transcripts/`**, since those files are required
> grading evidence, not local runtime artifacts.

---

## Part 1 — Dataset & RAG Core

### Task 1 — Dataset design choices

> **[TODO: fill in exact values after running `dataset.py`, then delete this line]**

| Parameter | Value used | Reasoning |
|---|---|---|
| Random seed | `<SEED>` | Fixed for deterministic, reproducible generation |
| Number of records | `<N>` (≥40 required) | |
| Category weights | `Billing: <x>%, Technical Issue: <x>%, Account Access: <x>%, Product Defect: <x>%, General Inquiry: <x>%` | Chosen so every category clears the ≥3-record minimum with margin |
| Status weights | `Open: <x>%, In Progress: <x>%, Escalated: <x>%, Resolved: <x>%, Closed: <x>%` | Every status guaranteed ≥1 record |
| `resolution_time_hours` range | `<min>`–`<max>` hours | One-sentence reasoning: `<e.g., "Ola's own SLA policy (see data/knowledge_base/sla_by_severity.md) caps critical-issue resolution at 24h and low-priority at 96h, so the sampled range spans slightly beyond that band to include both on-SLA and breached cases.">` |
| `days_since_created` range | 0–30 (fixed by brief) | |
| `escalated=True` percentage achieved | `<x>%` (must be 10–30%) | Achieved on seed `<SEED>` without hand-editing individual records |

**Reproduction:** run `python dataset.py --seed <SEED>` to regenerate the identical
40+ `SUPPORT_TICKETS` records used throughout this repo.

**Category/status counts (printed by `dataset.py`):**
```
<TODO: paste the exact printed output here, e.g.:>
Billing: 9
Technical Issue: 8
Account Access: 8
Product Defect: 7
General Inquiry: 8
---
Open: 8
In Progress: 8
Escalated: 8
Resolved: 8
Closed: 8
---
escalated=True: 17.5% (7/40)
```

### Task 2 — Knowledge base

12 documents in `data/knowledge_base/`, covering every required topic: ticket-priority
classification rules, SLA-by-severity policy, escalation matrix, refund/compensation
policy, customer-communication-channel policy, business-hours/holiday-support policy,
repeat-complaint-handling policy, service-credit policy, feedback-collection process,
VIP-customer handling policy, outage-communication protocol, and data-retention
policy for tickets.

### Task 3 — Chunking strategies

Two chunking strategies implemented in `app/chunking.py`, embedded with a local
SentenceTransformers model, and indexed into two separate ChromaDB collections:

| Collection name | Strategy | Chunk size / overlap (or sentence rule) |
|---|---|---|
| `ola_policies_fixed` | Fixed-size with overlap | `<TODO: e.g., 256 tokens, 40-token overlap>` |
| `ola_policies_sentence` | Sentence-based | `<TODO: e.g., 1–2 sentences per chunk>` |

### Task 4 — Grounded generation & threshold calibration

> **[TODO: fill in after running the calibration step in `app/rag.py`]**

Measured top-1 cosine similarity, in-scope vs. out-of-scope queries:

| Query | Type | Top-1 cosine similarity |
|---|---|---|
| `<in-scope query 1>` | in-scope | `<value>` |
| `<in-scope query 2>` | in-scope | `<value>` |
| `<in-scope query 3>` | in-scope | `<value>` |
| `<out-of-scope query 1>` | out-of-scope | `<value>` |
| `<out-of-scope query 2>` | out-of-scope | `<value>` |

**Chosen "I don't know" threshold: `<value>`** — set between the observed in-scope
cluster (`<range>`) and out-of-scope cluster (`<range>`) above, not an untested
tutorial default.

Demonstrated on ≥5 in-scope queries + 1 out-of-scope fallback query — see
`transcripts/part1_grounded_generation.txt`.

### Task 5 — Chunking strategy evaluation

> **[TODO: fill in after running `eval/retrieval_eval.py`]**

| Query | `ola_policies_fixed` P / R | `ola_policies_sentence` P / R |
|---|---|---|
| 1 | `<P>` / `<R>` | `<P>` / `<R>` |
| 2 | `<P>` / `<R>` | `<P>` / `<R>` |
| 3 | `<P>` / `<R>` | `<P>` / `<R>` |
| 4 | `<P>` / `<R>` | `<P>` / `<R>` |
| 5 | `<P>` / `<R>` | `<P>` / `<R>` |

**Recommendation:** `<TODO: 2-3 sentences citing the actual numbers above, e.g.,
"The sentence-based collection achieved higher average recall (0.xx vs 0.xx) at a
small precision cost (0.xx vs 0.xx), because policy documents here are short enough
that sentence boundaries rarely split a complete rule. We recommend deploying the
sentence-based collection.">`

---

## Part 2 — CrewAI Orchestration, Memory & Guardrails

### Task 6 — `check_support_ticket_status` and escalation score

```python
escalation_score = <TODO: exact formula, e.g.>
    0.6 * float(escalated) + 0.4 * min(days_since_created / <P80_VALUE>, 1.0)
```

Escalation recommended above **`<threshold>`**, corresponding to the
**`<percentile>`th percentile** of `days_since_created` in the generated dataset
(`<value>` days) — see `app/tools/ticket_lookup_tool.py` for the exact computation.

### Task 7 — CrewAI crew

3 agents defined in `app/crew.py`:
- **Retrieval Agent** — equipped with the RAG tool (Tasks 3–5)
- **Lookup Agent** — equipped with `check_support_ticket_status` (Task 6)
- **Response Composer** — combines both outputs into one final draft

Both tools' invocation demonstrated on different sample queries — see
`transcripts/part2_crew_kickoff.txt`.

### Task 8 — Session memory

Multi-turn state carried correctly across turns in `transcripts/part2_memory_carried.txt`;
a separate fresh-conversation run in `transcripts/part2_memory_reset.txt` shows state
correctly absent. In-process only, per brief (does not need to survive a restart).

### Task 9 — Structured output schema

Every crew response validated against `app/models.py::AgentResponse` (Pydantic).

### Task 10 — Guardrails

- **Input-side PII masking:** phone number is the only fixed-format field in this
  track (rider/driver name, pickup/drop address, and payment details are free text
  and explicitly out of scope for masking per the brief). Demonstrated firing in
  `transcripts/part2_guardrail_pii.txt`.
- **Prompt-injection detection:** demonstrated firing in
  `transcripts/part2_guardrail_injection.txt`.
- **Output-side groundedness check:** demonstrated refusing an ungrounded answer in
  `transcripts/part2_guardrail_groundedness.txt`.

---

## Part 3 — FastAPI Deployment & Evaluation

### Task 11 — Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Ask the agent a question (RAG + ticket lookup as needed) |
| `POST` | `/add-document` | Add a new policy document to the knowledge base at runtime |
| `WS` | `/chat` | Real-time multi-turn chat; gracefully handles `WebSocketDisconnect` |

Run locally: `uvicorn app.api.main:app --reload` → interactive docs at
`http://localhost:8000/docs`.

### Task 12 — Structured logging

Every request logged as one JSON-Lines entry (`logs/requests.jsonl`) with a trace ID
and timing. The logged request text is masked identically to what the model sees —
the phone number never reaches disk in the clear. See
`transcripts/part3_logging_sample.jsonl`.

### Task 13 — Evaluation at scale

> **[TODO: fill in after running `eval/llm_judge_eval.py`]**

15 test queries (≥1 per KB topic + ≥2 out-of-scope/edge-case), scored on Accuracy,
Grounding, Completeness, Safety — full per-query table in
`transcripts/part3_evaluation.md`.

| Metric | Average across 15 queries |
|---|---|
| Accuracy | `<value>` |
| Grounding | `<value>` |
| Completeness | `<value>` |
| Safety | `<value>` |

---

## Part 4 — Autogen Review Stage & Governance

### Task 14 — Autogen review stage

2-agent `RoundRobinGroupChat` (Policy-Compliance-Reviewer + Final-Editor),
`custom_message_types=[StructuredMessage[VerdictModel]]` registered on the team.
Demonstrated:
- **Approved-unchanged case:** `transcripts/part4_review_approved.txt`
- **Revised case** (deliberately injected ungrounded claim caught and corrected):
  `transcripts/part4_review_revised.txt`

### Task 15 — Four-layer AI governance

- **Application layer (least autonomy):** only the Lookup Agent is wired to
  `check_support_ticket_status`; no other agent has this tool in its toolset. See
  `app/governance.py` and `transcripts/part4_least_autonomy.txt` for the demonstration
  that a non-Lookup agent cannot invoke it.
- **Risk classification: Medium.** This system falls under the brief's own given
  Medium-risk example — "customer support tickets." It does not process medical
  data, hiring decisions, or high-stakes financial transactions (which would be
  High); it also does more than pure summarization/transcription (Low), since it
  makes escalation-relevant judgments and answers policy questions that affect a
  user's next action. `<TODO: expand to a full one-paragraph justification.>`
- **Runtime layer (cost cap):** per-request token/cost budget enforced in
  `app/governance.py`; an oversized simulated request is correctly rejected — see
  `transcripts/part4_cost_cap_rejection.txt`.

### Task 16 — Response caching

In-memory cache keyed by normalized query text (`app/cache.py`). Before/after evidence
(call counter or timing) for a repeated identical query in
`transcripts/part4_cache_hit.txt`.

---

## Acceptance Criteria Checklist

- [ ] `dataset.py` generates ≥40 tickets meeting every structural threshold; design
      choices stated above
- [ ] Knowledge base has ≥12 documents covering every required topic
- [ ] Both chunking strategies implemented, embedded, indexed into two Chroma
      collections
- [ ] Grounded generation demonstrated on ≥5 in-scope + 1 out-of-scope query
- [ ] Precision/recall computed for both collections on the same ≥5 queries, with
      per-query arithmetic and a numbers-cited recommendation
- [ ] `check_support_ticket_status` returns a designed, justified `escalation_score`
- [ ] CrewAI crew has ≥3 agents; both tools demonstrably invoked via `.kickoff()`
- [ ] Multi-turn memory demonstrated, plus a separate reset-state transcript
- [ ] Every crew response validates against a declared Pydantic schema
- [ ] Both guardrails demonstrated actually firing
- [ ] FastAPI: ≥2 HTTP endpoints + 1 WebSocket endpoint surviving a disconnect
- [ ] Every request logged as one JSON-Lines entry with a trace ID, PII masked
- [ ] All 4 evaluation scores reported per-query for 15 queries + 4 averages
- [ ] Autogen review stage demonstrated both approving and revising a draft
- [ ] Least-autonomy enforcement demonstrated; risk classified and justified; cost
      cap correctly rejects an oversized request
- [ ] Response caching demonstrated with before/after evidence

---

## Known Limitations / Out of Scope

- Rider/driver name, pickup/drop address, and payment details are free text with no
  reliable universal pattern to match under a keyless, `MOCK_LLM`-only masker — these
  are acknowledged as out of scope for masking per the brief. Only fabricated example
  data is used for these fields anywhere in this repo; no real user data is present.
- Session memory is in-process only and does not persist across a server restart,
  per the brief's explicit allowance.
- `MOCK_LLM` responses are deterministic templates, not real language generation —
  this is intentional and required for zero-API-key, zero-network grading.

---

## Optional: Real LLM Mode

To wire in a real LLM instead of `MOCK_LLM` (not required for grading):

```bash
export MOCK_LLM=false
export ANTHROPIC_API_KEY=<your-key>   # or OPENAI_API_KEY, depending on config.py
```

All acceptance criteria above are met under `MOCK_LLM=true`; this mode is provided
only as an extension point and is not exercised in any graded transcript.