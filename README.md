# MedScan Intelligence

**A multi-agent clinical RAG system that cites its evidence, knows when to ask a human, and is graded by a CI gate.**

Answers questions over clinical trial literature with hybrid retrieval, an agentic
corrective-RAG loop, cost-aware model routing, and an evaluation harness anchored on
structured ground truth from ClinicalTrials.gov — not on one LLM's opinion of another.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Orchestration: LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1f6feb.svg)](https://github.com/langchain-ai/langgraph)
[![Vector store: pgvector](https://img.shields.io/badge/vectors-pgvector%20%2F%20HNSW-336791.svg)](https://github.com/pgvector/pgvector)
[![Eval gate: DeepEval](https://img.shields.io/badge/CI-eval%20gated-brightgreen.svg)](#evaluation)
[![MCP](https://img.shields.io/badge/exposes-MCP%20server-black.svg)](#mcp-server)

---

## Table of contents

- [Why this exists](#why-this-exists)
- [The three things that make it different](#the-three-things-that-make-it-different)
- [Architecture](#architecture)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [The agents](#the-agents)
- [Cost-aware routing](#cost-aware-routing)
- [Evaluation](#evaluation)
- [Results](#results)
- [MCP server](#mcp-server)
- [Tech stack](#tech-stack)
- [Setup](#setup)
- [Running it](#running-it)
- [Project structure](#project-structure)
- [Design decisions, defended](#design-decisions-defended)
- [Known limitations and open findings](#known-limitations-and-open-findings)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this exists

An LLM asked about a drug's trial evidence will answer fluently whether or not it knows.
In a clinical context that is not a quirk — it is disqualifying. Three failure modes matter:
it hallucinates, its knowledge is frozen at training time, and even when it is right it
cannot tell you *where* the answer came from.

MedScan is built to make those failures observable and bounded rather than invisible:

- every answer is **grounded** in retrieved passages and carries its sources;
- retrieval is **hybrid**, because clinical questions mix semantic intent (*"how effective
  is this in elderly patients"*) with exact tokens (`NCT04280705`, `p<0.001`) and dense
  embeddings alone are bad at the second kind;
- low-confidence extraction **stops and asks a human** instead of guessing;
- quality is **measured, in CI**, against ground truth that exists independently of any LLM.

Clinical trial literature was chosen deliberately: it is high-stakes, it has rich
*structured* ground truth available from ClinicalTrials.gov, and it forces every one of
the above concerns to be real rather than decorative.

## The three things that make it different

Most RAG projects are a vector store, a prompt, and a demo. The parts of this one worth
reading the code for:

| | What | Why it matters |
|---|---|---|
| **1** | **Hybrid retrieval, ablated.** Dense (BGE-M3, 1024-d, HNSW) + sparse (Postgres `tsvector`) fused by Reciprocal Rank Fusion, then cross-encoder reranked. | The gain is *measured* against a dense-only baseline, not asserted. Each stage has to earn its latency. |
| **2** | **Agentic orchestration with a human escape hatch.** Four LangGraph agents with retry loops, confidence gates, and `interrupt()`-based human-in-the-loop on uncertain extractions. | Knowing when to defer to a human is a feature in a clinical domain, not a failure to automate. |
| **3** | **Evaluation as a first-class citizen.** RAGAS metrics + DeepEval thresholds as a **CI gate that blocks a PR** + Promptfoo adversarial suites, scored against ClinicalTrials.gov structured fields. | Quality becomes a build check instead of a vibe. Ground truth is objective where the data allows, so eval isn't just an LLM grading an LLM. |

Plus two things that are rare in a portfolio project: an **MCP server** so an external LLM
client can call this system as a tool, and a **fine-tuned BioBERT NER model** for
high-volume entity extraction where prompting a general LLM would be slower, costlier, and
less accurate.

## Architecture

```mermaid
flowchart TD
    subgraph ingest["Ingestion path (offline)"]
        A[Clinical document<br/>PDF / PubMed / ClinicalTrials.gov] --> B[Ingestion Agent<br/>parse + confidence estimate]
        B --> C{Extraction<br/>confidence OK?}
        C -- no --> B2[Fallback parser<br/>PyMuPDF] --> B
        C -- yes --> D[Clinical NER<br/>fine-tuned BioBERT]
        D --> E[Section-aware chunking]
        E --> F[Embed - BGE-M3 1024-d]
        F --> G[(PostgreSQL + pgvector<br/>vectors · tsvector · evidence)]
        D --> H[Structuring Agent<br/>LLM to structured evidence JSON]
        H --> I{Confidence<br/>above threshold?}
        I -- no, retries exhausted --> HITL[["interrupt() -> human review"]]
        HITL --> G
        I -- yes --> G
    end

    subgraph query["Query path (online)"]
        Q[User query] --> R[Complexity classifier<br/>heuristic, LLM fallback]
        R --> S{simple or complex}
        S -- simple --> M1[Llama 8B]
        S -- complex --> M2[Llama 70B]
        Q --> T[Hybrid retrieval]
        G --> T
        T --> U[RRF fusion -> cross-encoder rerank -> top-5]
        U --> V{Context<br/>sufficient?}
        V -- no --> W[Reformulate query] --> T
        V -- yes --> X[Grounded generation<br/>with citations]
        M1 --> X
        M2 --> X
        X --> Y[Self-critique]
        Y --> Z[Answer + sources + cost]
    end

    Z -.->|trace| LF[(LangFuse)]
    X -.->|trace| LF
    R -.->|routing decision| LF
```

Two things to notice in that diagram. First, **every loop is bounded** — ingestion retries
with a different parser a fixed number of times, retrieval reformulates a fixed number of
times, and each cycle has a terminal edge that produces a conservative, flagged answer
rather than spinning. Second, **the only path that writes unverified evidence to the store
is the one a human approved.**

A third agent sits above the store:

```mermaid
flowchart LR
    G[(Evidence store)] --> SY[Synthesis Agent]
    SY --> X1[Cross-reference<br/>all evidence for a drug/condition]
    SY --> X2[Contradiction detection<br/>across trials]
    SY --> X3[Evidence grading<br/>Phase 3 RCT &gt; Phase 2 &gt; observational]
```

## The retrieval pipeline

```
query
  → dense_search      BGE-M3 → pgvector HNSW (<=> cosine)        → top 20
  → sparse_search     Postgres tsvector / ts_rank                → top 20
  → RRF fusion        score(d) = Σ 1 / (k + rank_r(d)),  k = 60
  → cross-encoder rerank                                        → top 5
  → cited context assembly
```

**Why fuse on rank instead of score.** Cosine similarity is bounded 0–1; `ts_rank` is
unbounded. Normalising two incompatible score scales is brittle and needs per-corpus
tuning. RRF sidesteps scores entirely: a passage ranked highly by *both* retrievers wins,
and `k = 60` (the canonical value from Cormack et al.) damps the tail so low-ranked
results can't swing the fusion.

**Why a reranker when retrieval already ranked.** The retriever is a bi-encoder — query and
document are embedded *separately*, so document vectors can be precomputed at index time
and search scales to the whole corpus, but the query and document never attend to each
other. The cross-encoder feeds `(query, document)` through a transformer *jointly*, which
is far more precise and far too slow to run corpus-wide. So: recall from the cheap stage
over everything, precision from the expensive stage over 20 candidates.

**Why section-aware chunking.** Fixed-size chunking cuts a finding away from the numbers
that support it and runs straight through section boundaries. This splits on clinical
structure instead — Methods, Results, Adverse Events stay intact, over-long sections split
sentence-aware with overlap, and every chunk carries its `section_type` so retrieval can
filter by section.

**Why HNSW over IVFFlat.** Hierarchical Navigable Small World gives logarithmic-ish search
with higher recall and no training step; `ef_search` is the runtime recall/latency knob.
IVFFlat builds faster but needs training data and generally gives up recall. At this scale
recall was the thing worth paying for.

## The agents

Orchestrated as a LangGraph `StateGraph` with conditional edges and a `SqliteSaver`
checkpointer — so state is persisted, runs are resumable by `thread_id`, and the
human-in-the-loop pause/resume cycle works across process restarts.

| Agent | Responsibility | Control flow it owns |
|---|---|---|
| **Ingestion** | Parse a document, estimate extraction confidence, run clinical NER | Retry with fallback parser on low confidence |
| **Structuring** | LLM-extract structured evidence (drug, phase, endpoint, adverse events) as JSON | Retry on low confidence, then `interrupt()` → human review |
| **Synthesis** | Pull all evidence for a drug/condition, cross-reference, detect contradictions, grade by study design | Deterministic grading hierarchy |
| **Query** | Answer questions — the agentic RAG loop | `classify → retrieve → evaluate_context → (reformulate → retrieve)* → generate → self_critique` |

The Query Agent is a **Corrective RAG (CRAG)** implementation: it *grades its own
retrieval* before generating, and if the context is judged insufficient it reformulates and
retries rather than generating from thin evidence. That is the difference between agentic
and naive RAG — not the number of LLM calls, but the presence of a decision.

**Human-in-the-loop, concretely.** Structuring Agent extracts with a confidence score →
below threshold, retry → still below threshold, `interrupt()` pauses the graph and surfaces
the item for review → a human approves or corrects → the graph resumes by `thread_id` with
the correction merged into state → the corrected evidence is persisted.

## Cost-aware routing

Not every query needs the big model. *"What phase is NCT04280705?"* is a lookup;
*"reconcile the adverse-event profiles across these three trials"* is not. Routing both to
a 70B model is wasteful; routing both to an 8B model loses the hard ones.

Two-stage classifier, cheapest-first:

1. **Heuristic** — lexical and structural cues (`compare`, `why`, `synthesise`,
   multi-entity queries → complex; short factual patterns and ID lookups → simple).
   Resolves the majority of traffic at zero cost.
2. **LLM fallback** — the ambiguous middle is classified by the *cheap* model.

Simple → Llama 8B. Complex → Llama 70B. **The safe default is the stronger model** — the
router fails upward, not downward.

Every routing decision, token count, and cost is logged to LangFuse, which is what makes
the cost-vs-accuracy analysis reconstructable after the fact rather than a one-off
spreadsheet. A frontier model (Claude) is benchmarked **once** as the quality ceiling, to
establish how small the gap actually is on this workload.

## Evaluation

Evaluating RAG is hard because there is no single label: an answer can be relevant but
ungrounded, grounded but irrelevant, or both while retrieval never surfaced the right
passage at all. So the generator and the retriever are measured **separately**, and three
tools are used because they answer three different questions.

| Tool | Question it answers | Where it runs |
|---|---|---|
| **RAGAS** | How good is it, component by component? | Dashboard + ablation studies |
| **DeepEval** | Did this PR make it worse? | **pytest-native CI gate — blocks the merge** |
| **Promptfoo** | Can it be made to misbehave? | Adversarial / red-team suite |

The four RAGAS metrics, and what each isolates:

- **Faithfulness** — decompose the answer into atomic claims, verify each against the
  retrieved context. This is the direct hallucination metric. *(generator)*
- **Answer relevancy** — does the answer actually address the question? *(generator)*
- **Context precision** — are the relevant chunks ranked near the top? *(retriever ranking)*
- **Context recall** — did retrieval fetch everything needed to answer? *(retriever coverage)*

**Ground truth without hand-labelling thousands of examples.** This is the design move the
whole eval stack rests on: ClinicalTrials.gov exposes *structured* fields — phase, primary
endpoint, interventions, enrolment — whose values are authoritative. QA pairs are generated
from those fields, so the correct answer is **objectively known** and scoring is not purely
one LLM's judgement of another. LLM-as-judge is still used where structure runs out
(free-text synthesis), with its known biases acknowledged — position, verbosity,
self-preference — and mitigated by anchoring on structured truth wherever the data permits
and spot-checking judge scores against a human-reviewed sample.

**The CI gate.** On any PR touching retrieval, agents, or prompts, the pipeline ingests a
small fixed corpus, runs the DeepEval suite, and **fails the build** if mean faithfulness or
answer relevancy fall below threshold. Quality regressions are caught the same way a broken
test is.

**Ablations.** Each component has to justify its complexity:

| Ablation | What it isolates |
|---|---|
| hybrid vs. dense-only | Does sparse retrieval actually add anything? |
| rerank vs. no rerank | Is the cross-encoder worth its latency? |
| BGE-M3 vs. MiniLM (384-d) | How much does embedding quality matter here? |

## Results

Measured on the ClinicalTrials.gov-derived evaluation set.

| Metric | Baseline | MedScan | Δ |
|---|---|---|---|
| Faithfulness | 0.77 *(dense-only)* | **0.89** | +0.12 |
| Context precision | — | — | **~+20%** over dense-only |
| Avg. cost per query | 70B for everything | **routed** | **~−40%** |
| Accuracy vs. frontier-model ceiling | Claude = 1.00 | **within ~4%** | — |

The headline is the last two rows read together: **average cost fell ~40% while accuracy
stayed within ~4% of a frontier model.** That is the production-economics argument — most
queries sit where the cheap model is already right, so routing keeps you near the accuracy
frontier at a fraction of the spend.

> **On reproducibility:** these figures come from the evaluation run on the corpus and
> eval set described above. Re-running against a different corpus, a different judge
> model, or a changed `ef_search` will move them. `python -m medscan.eval.run --full`
> regenerates the table; the ablation deltas are in `eval/reports/`.

## MCP server

MedScan exposes its knowledge base over the **Model Context Protocol**, so an external LLM
client (Claude Desktop, or any MCP-capable host) can call this system as a tool rather than
through a bespoke HTTP integration:

| Tool | Purpose |
|---|---|
| `query_clinical_evidence` | Full agentic RAG answer with citations |
| `search_trials` | Structured search over the trial index |
| `get_drug_evidence` | All graded evidence for a given drug, with contradictions flagged |

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "medscan": {
      "command": "python",
      "args": ["-m", "medscan.mcp_server"],
      "env": { "MEDSCAN_DB_URL": "postgresql://localhost:5432/medscan" }
    }
  }
}
```

The reason this is here and not on a roadmap: the direction of travel in this ecosystem is
interoperable tools, not walled-garden integrations. A retrieval system that can't be
called by someone else's agent is a dead end.

## Tech stack

**Orchestration** · LangGraph (`StateGraph`, conditional edges, `SqliteSaver`
checkpointing, `interrupt()`)
**Storage & retrieval** · PostgreSQL + pgvector (HNSW), Postgres full-text (`tsvector` /
`ts_rank`), BGE-M3 (1024-d) embeddings, cross-encoder reranker
**Models** · Llama 8B / 70B via Groq (routed), Claude as a one-off quality ceiling,
fine-tuned BioBERT for clinical NER
**Serving** · FastAPI (sync + SSE streaming endpoints), MCP server
**Evaluation & observability** · RAGAS, DeepEval (pytest/CI), Promptfoo, LangFuse
(self-hosted)

**One database, deliberately.** Vectors, full-text indexes, and structured evidence all
live in Postgres. No sync job between a vector store and a relational store, no second
system to operate, no managed-service bill — and at this corpus size pgvector is not the
bottleneck.

## Setup

```bash
git clone https://github.com/susheelram333/medscan-intelligence.git
cd medscan-intelligence

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

**Postgres with pgvector:**

```bash
docker compose up -d postgres      # pgvector image, exposes 5432
python -m medscan.db.migrate       # schema, HNSW index, tsvector columns + triggers
```

**Environment:**

```bash
cp .env.example .env
```

```ini
MEDSCAN_DB_URL=postgresql://medscan:medscan@localhost:5432/medscan
GROQ_API_KEY=                      # Llama 8B / 70B
ANTHROPIC_API_KEY=                 # optional — only for the ceiling benchmark
LANGFUSE_PUBLIC_KEY=               # optional — tracing degrades gracefully if unset
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
EMBEDDING_MODEL=BAAI/bge-m3        # self-hosted, no per-call cost
RERANKER_MODEL=BAAI/bge-reranker-base
HNSW_EF_SEARCH=100                 # recall/latency knob
RRF_K=60
RETRIEVAL_TOP_K=20                 # candidates per retriever, pre-fusion
RERANK_TOP_N=5                     # passages that reach the prompt
MAX_RETRIEVAL_ATTEMPTS=3
MAX_STRUCTURING_RETRIES=2
CONFIDENCE_THRESHOLD=0.7           # below this → retry, then human review
```

Embeddings and the reranker are open-weight and run locally, so indexing costs nothing per
call. The only metered calls are generation.

## Running it

**Ingest a corpus:**

```bash
python -m medscan.ingest --source clinicaltrials --condition "type 2 diabetes" --limit 200
python -m medscan.ingest --source pdf --path ./data/papers/
```

Ingestion is **idempotent on `(source, source_id)`** — re-ingesting a document replaces its
chunks and evidence rather than duplicating them, which is what makes "a trial posted new
results" a re-ingest instead of a rebuild.

**Ask it something:**

```bash
python -m medscan.query "What were the primary endpoints in NCT04280705?"
python -m medscan.query --explain "Compare adverse event profiles across semaglutide phase 3 trials"
```

`--explain` prints the routing decision, the fused candidate list before and after
reranking, and the self-critique verdict — the fastest way to debug a bad answer.

**Serve the API:**

```bash
uvicorn medscan.api:app --reload
```

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What phase is NCT04280705 and what was its primary endpoint?"}' | python3 -m json.tool

# token-by-token streaming
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the evidence for semaglutide in weight management"}'
```

The response carries the answer, its cited sources, the model that served it, and the
measured cost — provenance and economics are part of the payload, not a log line.

**Review the human-in-the-loop queue:**

```bash
python -m medscan.hitl list                       # extractions awaiting review
python -m medscan.hitl resume <thread_id> --approve
python -m medscan.hitl resume <thread_id> --correct corrections.json
```

**Evaluate:**

```bash
python -m medscan.eval.run --suite ragas          # full metric suite
python -m medscan.eval.run --ablation hybrid      # hybrid vs. dense-only
pytest tests/eval -v                              # the DeepEval CI gate, locally
promptfoo eval -c promptfoo/redteam.yaml          # adversarial suite
```

## Project structure

```
src/medscan/
  agents/
    ingestion.py        parse → confidence gate → NER → fallback retry
    structuring.py      LLM evidence extraction → confidence gate → interrupt()
    synthesis.py        cross-reference, contradiction detection, evidence grading
    query.py            CRAG loop: classify → retrieve → evaluate → reformulate → generate → critique
  graph/
    build.py            the LangGraph StateGraph — nodes, conditional edges, cycle bounds
    state.py            typed graph state
    checkpoint.py       SqliteSaver wiring for resumable / HITL runs
  retrieval/
    dense.py            BGE-M3 embedding + pgvector HNSW search
    sparse.py           tsvector / ts_rank keyword search
    fusion.py           Reciprocal Rank Fusion
    rerank.py           cross-encoder reranking
    chunking.py         section-aware, sentence-aware, overlapping
    context.py          cited-context assembly for the prompt
  routing/
    classifier.py       heuristic → LLM-fallback complexity classification
    router.py           model selection + cost accounting
  ner/
    train_biobert.py    BC5CDR fine-tune (token classification)
    infer.py            inference path used by the Ingestion Agent
    baseline_scispacy.py  F1 comparison baseline
  db/
    schema.sql          documents, chunks, evidence, vectors, tsvector, HNSW index
    migrate.py
    repository.py       idempotent upsert on (source, source_id)
  eval/
    ground_truth.py     QA pairs generated from ClinicalTrials.gov structured fields
    ragas_suite.py      faithfulness, answer relevancy, context precision, context recall
    ablations.py        hybrid vs dense, rerank vs none, BGE-M3 vs MiniLM
    cost_pareto.py      cost-vs-accuracy frontier, incl. the frontier-model ceiling
    run.py              harness entrypoint
  observability/
    langfuse.py         tracing — prompt, model, tokens, cost, latency, routing decision
  api.py                FastAPI — sync + SSE streaming
  mcp_server.py         MCP tools: query_clinical_evidence, search_trials, get_drug_evidence
  config.py             settings, thresholds, retry bounds

tests/
  test_retrieval.py     fusion maths, chunk boundaries, HNSW recall sanity
  test_agents.py        confidence gates, retry bounds, loop termination
  test_graph.py         conditional routing, interrupt/resume by thread_id
  eval/                 DeepEval threshold tests — the CI gate
promptfoo/
  redteam.yaml          prompt injection, hallucination pressure, unsafe-advice probes
eval/
  ground_truth/         generated QA pairs
  reports/              metric runs, ablation deltas, cost-vs-accuracy plots
docker-compose.yml      Postgres + pgvector, LangFuse
.github/workflows/
  eval-gate.yml         ingest fixture corpus → DeepEval → fail on regression
```

## Design decisions, defended

Short answers to the questions this architecture invites.

**Why RAG and not fine-tuning?** Fine-tuning changes *behaviour*, not knowledge. Facts
baked into weights go stale, can't be cited, and cost a training run to update. RAG keeps
knowledge current by re-indexing and provides provenance for free. The one place this
project *does* fine-tune is NER — a narrow, high-volume, well-defined extraction task where
a small specialised model beats prompting a general one on cost, latency, and accuracy at
once.

**Why not just a long context window?** Fine for small static inputs; falls over on cost,
latency, and lost-in-the-middle degradation as context grows. RAG scales to a corpus;
context stuffing scales to a document.

**Why LangGraph and not CrewAI / AutoGen / a plain loop?** The pipeline needs cycles
(retry with a different parser, retrieve again), conditional routing, and a durable pause
for human review. LangGraph models those as first-class graph constructs and persists state
so a run can resume by `thread_id` — which the HITL flow essentially requires.
Role-play-oriented frameworks are clumsier for deterministic, loop-heavy control flow, and
a hand-rolled loop means hand-rolling state, retries, persistence, and observability.

**Why not a managed vector DB?** One Postgres holds vectors, full-text, and structured
evidence — no sync, no second system, no bill. Revisit at a scale this corpus isn't at.

**Does cosine vs. dot product matter?** Not here. BGE-M3 emits normalised vectors, so
dot product equals cosine and pgvector's `<=>` ranks identically.

## Known limitations and open findings

Stated rather than smoothed over.

- **Judge-metric noise.** Where structured ground truth runs out — free-text synthesis,
  contradiction narratives — scoring falls back to LLM-as-judge, which carries position,
  verbosity, and self-preference bias. Mitigated by anchoring on structured truth and
  spot-checking against human review; not eliminated.
- **`k = 60` in RRF is the canonical default, not a tuned value.** There wasn't enough
  eval data to justify a sweep, and nothing indicated it was the bottleneck. A proper
  sweep, plus an A/B against weighted-score fusion, is the honest next step.
- **Chunk size and overlap are reasoned, not swept.** Section-aware chunking is defensible
  on structure, but the specific size/overlap for over-long sections is a hyperparameter
  that should be optimised against context precision and recall rather than chosen.
- **ANN recall degrades with scale.** HNSW is approximate by construction. Any change to
  `ef_search`, or quantisation added for memory, needs `recall@k` measured before and after
   — approximation buys speed with recall.
- **Parser fragility on real-world PDFs.** Messy scans and multi-column layouts are the
  most common ingestion failure. The confidence gate plus fallback parser contains this; it
  doesn't solve it.
- **Prompt injection is tested, not solved.** Grounding instructions, a self-critique pass,
  and Promptfoo injection suites give coverage and fail-safe defaults. No claim of
  immunity.
- **Free-tier rate limits** shape throughput on the generation path; batching and caching
  are mitigations, not fixes.

## Roadmap

- A properly labelled evaluation set with human validation of judge scores
- Hyperparameter sweeps: RRF `k`, chunk size/overlap, `ef_search`
- Query + answer caching with latency SLOs per pipeline stage
- Graph-RAG for genuine multi-hop reasoning across trials (the Synthesis Agent's
  contradiction detection is the first step toward this, not the finished version)
- A reranker fine-tuned on clinical query/passage pairs
- Online quality monitoring: score faithfulness on a sample of live traffic via LangFuse
  and alert on drift

**What I'd do differently from the start:** build the eval harness and ground-truth set
*before* the pipeline, so every component decision was measured from day one instead of
retrofitted.

## License

MIT — see [LICENSE](LICENSE).
