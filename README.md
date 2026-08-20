# 🧠 Adaptive Conversational Memory

> **An adaptive memory architecture for conversational AI that goes beyond traditional semantic retrieval by combining semantic, episodic, procedural, and temporal memory with hybrid retrieval, adaptive reranking, conflict resolution, consolidation, lifecycle management, and controlled forgetting.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-507%20Passed-success)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-20%2F20%20Cases-success)]()
[![Recall@5](https://img.shields.io/badge/Adaptive%20Recall%405-1.000-brightgreen)]()
[![Baseline Recall@5](https://img.shields.io/badge/Baseline%20Recall%405-0.800-orange)]()

---

## 📌 Overview

Most Retrieval-Augmented Generation (RAG) systems treat retrieved information primarily as a collection of semantically similar documents.

That approach works well for static knowledge, but conversational AI requires something more sophisticated.

A user may have:

- Current preferences
- Previous preferences
- Repeated behaviors
- Past events
- Procedural knowledge
- Time-dependent information
- Conflicting memories
- Frequently accessed information
- Information that should eventually be forgotten

For example:

> "Which database do I currently prefer?"

A traditional semantic retriever may return both:

- "I previously used MongoDB."
- "I now prefer PostgreSQL."

Both statements are semantically relevant, but only one represents the user's **current preference**.

This project addresses that problem by introducing an **Adaptive Conversational Memory Architecture** that dynamically routes queries, retrieves memories from multiple memory systems, fuses candidates, reranks them using multiple signals, resolves conflicts, builds context, and finally generates a response.

---

# 🎯 Project Objective

The primary objective is to design a conversational memory system capable of:

1. Understanding different types of memories.
2. Storing memories using appropriate representations.
3. Routing queries to the most relevant memory subsystem.
4. Combining results from multiple retrieval mechanisms.
5. Ranking memories using more than semantic similarity.
6. Handling conflicting and outdated information.
7. Consolidating repeated information.
8. Managing memory lifecycle and controlled forgetting.
9. Building relevant conversational context.
10. Evaluating retrieval quality against a baseline system.

---

# 🏗️ System Architecture

## Memory Ingestion Flow

```text
                         ┌──────────────────────┐
                         │   User Conversation  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Memory Extraction  │
                         │      using LLM       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Memory Classifier   │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │  Semantic   │       │  Episodic   │       │ Procedural  │
       │   Memory    │       │   Memory    │       │   Memory    │
       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Importance /         │
                         │ Confidence Scoring   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Conflict Detection   │
                         │ & Resolution         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Memory Consolidation │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │          Memory Storage         │
                    │                                 │
                    │ Vector │ Temporal │ Graph/Proc │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Memory Lifecycle     │
                         │ & Controlled Forget. │
                         └──────────────────────┘
```

---

# 🔎 Adaptive Retrieval Architecture

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Memory Router     │
                         │                      │
                         │ Semantic             │
                         │ Episodic             │
                         │ Procedural            │
                         │ Temporal              │
                         └──────────┬───────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────┐
               │          Memory-Specific Retrieval       │
               │                                          │
               │ Vector Retrieval                         │
               │ Temporal Retrieval                       │
               │ Graph Retrieval                          │
               │ Procedural Retrieval                     │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Candidate Fusion   │
                         │                      │
                         │ Deduplication        │
                         │ Score Normalization   │
                         │ Hybrid Fusion         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Adaptive Reranker    │
                         │                      │
                         │ Retrieval Score      │
                         │ Query Relevance      │
                         │ Importance            │
                         │ Recency               │
                         │ Source Priority       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Conflict Detection   │
                         │ & Resolution         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Context Builder   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Response Generator  │
                         │        + LLM         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Final Response     │
                         └──────────────────────┘
```

---

# 🧩 Core System Components

## 1. Memory Extraction

Conversational information is extracted into structured memory representations.

Examples:

```text
"I prefer PostgreSQL now."
"I previously used MongoDB."
"I worked on an AI project yesterday."
"My usual deployment process involves..."
```

The extracted information is converted into structured memory objects.

---

## 2. Memory Classification

The system classifies memories according to their role.

### Semantic Memory

Stable facts, preferences, interests, and knowledge.

Example:

```text
I prefer Python for machine learning projects.
```

### Episodic Memory

Past events and experiences.

Example:

```text
Yesterday I worked on my adaptive conversational memory project.
```

### Procedural Memory

Knowledge about how the user performs tasks.

Example:

```text
I first implement the experiment, then run its tests,
and finally run the complete pytest suite.
```

### Temporal Memory

Information whose relevance depends strongly on time.

Example:

```text
I currently use Go.
I previously used Python.
```

---

## 3. Memory Router

The Memory Router analyzes the query and determines which memory subsystem should be prioritized.

Examples:

```text
"What did I work on yesterday?"
        ↓
     Episodic

"How do I deploy my application?"
        ↓
    Procedural

"What database do I currently prefer?"
        ↓
     Semantic
     + Temporal signals
```

---

## 4. Vector Memory

Semantic memories are represented using embeddings and stored in a vector store.

The vector retrieval system provides the initial semantic candidate set.

This is useful for queries where meaning is more important than exact keyword matching.

---

## 5. Temporal Memory

Temporal information is handled separately so that the system can distinguish between:

```text
Previous preference
        ↓
Current preference
        ↓
Future / latest information
```

This prevents older memories from being treated as equally relevant when the user explicitly asks about current or recent information.

---

## 6. Graph-Based Memory

Graph structures allow relationships between memories, entities, and procedural states to be represented explicitly.

This is particularly useful when the relationship between pieces of information matters rather than just their semantic similarity.

---

## 7. Hybrid Retrieval

Instead of relying on a single retrieval mechanism, candidate results from different sources are combined.

The hybrid retrieval layer performs:

```text
Multiple Retrieval Sources
          ↓
     Deduplication
          ↓
   Score Normalization
          ↓
     Score Fusion
          ↓
   Candidate Ranking
```

This creates a larger candidate pool for the adaptive reranker.

---

## 8. Adaptive Reranking

The reranker is one of the key components of the system.

Instead of ranking memories purely according to retrieval similarity, the system considers multiple signals:

```text
Final Score =
    Retrieval Relevance
  + Query Relevance
  + Importance
  + Recency
  + Source Priority
```

Current deterministic ranking configuration:

| Signal | Weight |
|---|---:|
| Retrieval Score | 0.55 |
| Query Relevance | 0.15 |
| Importance | 0.12 |
| Recency | 0.10 |
| Source Priority | 0.08 |

This allows a memory with slightly lower embedding similarity to outrank a less useful memory when other signals indicate that it is more relevant.

---

## 9. Conflict Detection & Resolution

Conversational memory naturally contains contradictions.

Example:

```text
Old:
I use MongoDB.

New:
I now prefer PostgreSQL.
```

The system identifies memories belonging to the same conflict group and resolves the conflict while preserving historical information.

The goal is not to simply delete the old memory.

Instead:

```text
Historical Memory
       +
Current Memory
       ↓
Conflict Resolution
       ↓
Current preference is prioritized
Historical information remains available
```

---

## 10. Memory Consolidation

Repeated information can be consolidated into stronger memory representations.

For example:

```text
Python used in project A
Python used in project B
Python used in project C
```

can support the higher-level understanding:

```text
Python is repeatedly used across my projects.
```

This reduces redundant memory representations while preserving useful information.

---

## 11. Memory Lifecycle

Memories are not treated as permanently static objects.

Conceptually:

```text
Created
   ↓
Active
   ↓
Accessed / Updated
   ↓
Consolidated
   ↓
Potentially Forgotten
```

The system also contains forgetting mechanisms for controlled memory management.

---

## 12. Context Building

After retrieval and reranking, the most relevant memories are converted into a context representation.

```text
Retrieved Memories
       ↓
Ranked Memories
       ↓
Conflict Resolution
       ↓
Context Builder
       ↓
LLM Context
```

This prevents irrelevant memories from unnecessarily occupying the model's context.

---

## 13. Response Generation

The final context is passed to the response-generation layer.

```text
User Query
     +
Relevant Memory Context
     ↓
     LLM
     ↓
Final Response
```

---

# 🧠 Why This Architecture?

Traditional semantic RAG:

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
Top-K Documents
 ↓
LLM
```

Adaptive Conversational Memory:

```text
Query
 ↓
Memory Routing
 ↓
Multiple Memory Systems
 ↓
Hybrid Retrieval
 ↓
Candidate Fusion
 ↓
Adaptive Reranking
 ↓
Conflict Resolution
 ↓
Context Construction
 ↓
LLM
```

The adaptive architecture provides additional mechanisms for:

- Changing preferences
- Historical information
- Procedural knowledge
- Time-sensitive memories
- Conflicting memories
- Repeated information
- Memory importance
- Memory recency
- Memory lifecycle

---

# 📊 Evaluation

The system includes a dedicated retrieval benchmark containing **20 evaluation cases** covering:

- Semantic retrieval
- Episodic retrieval
- Procedural retrieval
- Temporal retrieval
- Conflict handling
- Consolidation
- Noise
- Mixed queries

The adaptive retriever is evaluated against a baseline retriever.

## Final Benchmark Results

| Metric | Baseline | Adaptive | Improvement |
|---|---:|---:|---:|
| **Recall@5** | 0.800 | **1.000** | **+25.00%** |
| **Precision@5** | 0.160 | **0.232** | **+44.79%** |
| **Hit@5** | 0.800 | **1.000** | **+25.00%** |
| **MRR** | 0.545 | **0.674** | **+23.70%** |
| **NDCG@5** | 0.608 | **0.754** | **+24.02%** |

### Key Result

> **Adaptive Recall@5 improved from 0.800 to 1.000, achieving a 25% relative improvement over the baseline.**

The adaptive system achieved **1.000 Recall@5 and 1.000 Hit@5 across all 20 benchmark cases.**

---

# 🧪 Testing

The project contains a comprehensive automated test suite covering the major system components.

Final validation:

```text
507 / 507 tests passed
```

The complete retrieval benchmark was validated with:

```text
20 / 20 cases
```

Validation flow:

```text
Unit Tests
    ↓
Component Tests
    ↓
Memory Pipeline Tests
    ↓
Retrieval Benchmark
```

---

# 📈 Benchmark Interpretation

### Recall@5

```text
Baseline   ████████░░  0.800
Adaptive   ██████████  1.000
```

### Hit@5

```text
Baseline   ████████░░  0.800
Adaptive   ██████████  1.000
```

### NDCG@5

```text
Baseline   ██████░░░░  0.608
Adaptive   ████████░░  0.754
```

The improvement is not limited to whether the correct memory appears in the retrieved set. MRR and NDCG also improve, indicating better ranking quality.

---

# 🗂️ Project Structure

```text
Adaptive-Conversational-Memory/
│
├── data/
│   ├── conversations/
│   ├── evaluation/
│   └── memories/
│
├── scripts/
│   ├── populate_graph_store.py
│   ├── populate_temporal_store.py
│   ├── populate_vector_store.py
│   └── run_benchmark.py
│
├── src/
│   ├── classification/
│   ├── conflict/
│   ├── consolidation/
│   ├── embeddings/
│   ├── evaluation/
│   ├── forgetting/
│   ├── lifecycle/
│   ├── llm/
│   ├── models/
│   ├── pipeline/
│   ├── retrieval/
│   ├── routing/
│   └── storage/
│
├── tests/
│   ├── test_adaptive_memory.py
│   ├── test_adaptive_retriever.py
│   ├── test_baseline_rag.py
│   ├── test_conflict.py
│   ├── test_consolidation.py
│   ├── test_forgetting.py
│   ├── test_hybrid_retriever.py
│   ├── test_memory_pipeline.py
│   ├── test_reranker.py
│   ├── test_router.py
│   └── ...
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone <your-repository-url>
cd Adaptive-Conversational-Memory
```

## Create a virtual environment

```bash
python -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Run the complete test suite

```bash
python -m pytest
```

Expected:

```text
507 passed
```

## Run the retrieval benchmark

```bash
PYTHONPATH=. python scripts/run_benchmark.py
```

The benchmark compares the baseline retrieval system against the adaptive retrieval system.

---

# 📊 Evaluation Metrics

### Recall@K

Measures how many relevant memories are retrieved within the top K results.

### Precision@K

Measures the proportion of retrieved results that are relevant.

### Hit@K

Measures whether at least one relevant memory appears in the top K.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant memory is ranked.

### NDCG@K

Measures ranking quality while considering the position of relevant results.

---

# 🔬 Design Principles

### 1. Memory is heterogeneous

Not every memory should be represented or retrieved in the same way.

### 2. Relevance is multidimensional

Semantic similarity alone is insufficient.

### 3. Time matters

A recent preference may be more relevant than an older conflicting preference.

### 4. Conflicts should be resolved, not blindly deleted

Historical information can remain useful.

### 5. Retrieval should be adaptive

Different queries require different memory strategies.

### 6. Memory requires lifecycle management

Memories should be created, accessed, updated, consolidated, and potentially forgotten.

### 7. Retrieval should be measurable

The system includes an explicit evaluation framework rather than relying only on qualitative examples.

---

# 🛠️ Technology Stack

```text
Language
    Python

AI / ML
    Sentence Embeddings
    Hugging Face Models
    LLM-based Memory Extraction
    LLM-based Response Generation

Retrieval
    Vector Retrieval
    Hybrid Retrieval
    Graph Retrieval
    Temporal Retrieval
    Adaptive Reranking

Storage
    Vector Store
    Temporal Store
    Graph Store
    SQLite-based Memory Storage

Evaluation
    Pytest
    Recall@K
    Precision@K
    Hit@K
    MRR
    NDCG
```

---

# 🎓 Project Highlights

This project demonstrates practical implementation of:

- Conversational AI memory architectures
- Retrieval-Augmented Generation
- Semantic search
- Hybrid retrieval
- Embedding-based retrieval
- Query routing
- Reranking
- Temporal reasoning
- Conflict resolution
- Memory consolidation
- Controlled forgetting
- Graph-based memory
- Evaluation-driven development
- End-to-end AI pipeline design

---

# 💡 Example

Consider the following memories:

```text
Memory 1:
I previously used MongoDB.

Memory 2:
I now prefer PostgreSQL.

Memory 3:
I usually use PostgreSQL for relational projects.
```

Query:

```text
Which database do I currently prefer?
```

Instead of simply returning the most semantically similar memories, the adaptive system considers:

```text
Semantic relevance
        +
Query relevance
        +
Importance
        +
Recency
        +
Memory source
        +
Conflict resolution
```

The system can therefore prioritize the current preference while preserving the historical memory.

---

# 🧠 Adaptive vs Traditional Retrieval

| Capability | Traditional Semantic RAG | Adaptive Conversational Memory |
|---|---|---|
| Semantic retrieval | ✅ | ✅ |
| Multiple memory types | ❌ | ✅ |
| Query routing | ❌ | ✅ |
| Temporal retrieval | Limited | ✅ |
| Procedural memory | Limited | ✅ |
| Graph memory | Limited | ✅ |
| Hybrid retrieval | Limited | ✅ |
| Adaptive reranking | Limited | ✅ |
| Importance scoring | ❌ | ✅ |
| Recency scoring | ❌ | ✅ |
| Conflict resolution | ❌ | ✅ |
| Memory consolidation | ❌ | ✅ |
| Controlled forgetting | ❌ | ✅ |
| Retrieval benchmark | Varies | ✅ |

---

# 📌 Current Status

```text
┌────────────────────────────────────────────┐
│        ADAPTIVE MEMORY PROJECT             │
├────────────────────────────────────────────┤
│                                            │
│  Memory Architecture       ✅ Complete     │
│  Query Routing             ✅ Complete     │
│  Hybrid Retrieval          ✅ Complete     │
│  Adaptive Reranking        ✅ Complete     │
│  Conflict Resolution       ✅ Complete     │
│  Memory Consolidation      ✅ Complete     │
│  Memory Lifecycle          ✅ Complete     │
│  Controlled Forgetting     ✅ Complete     │
│  End-to-End Pipeline       ✅ Validated    │
│  Automated Tests           ✅ 507/507      │
│  Benchmark                 ✅ 20/20        │
│  Adaptive Recall@5         ✅ 1.000        │
│                                            │
└────────────────────────────────────────────┘
```

---

# 📈 Final Result

```text
Baseline Recall@5
        0.800
          │
          │ +25%
          ▼
Adaptive Recall@5
        1.000
```

Additional improvements:

```text
Precision@5   0.160 → 0.232
Hit@5         0.800 → 1.000
MRR           0.545 → 0.674
NDCG@5        0.608 → 0.754
```

The benchmark demonstrates measurable improvement in both **retrieval coverage and ranking quality**.

---

# 🚧 Future Improvements

Potential future extensions include:

- Learned reranking models
- Neural conflict resolution
- Larger conversational datasets
- Online memory learning
- User-specific memory policies
- More advanced temporal reasoning
- Long-term memory compression
- Reinforcement-based retrieval optimization
- Multi-user memory isolation
- Distributed memory storage
- Production-scale vector and graph infrastructure
- Large-scale evaluation across real conversational datasets

---

# 📜 License

This project is intended for academic and research purposes.

Add your preferred license here if the repository will be publicly distributed.

---

# ⭐ Summary

**Adaptive Conversational Memory** is a multi-layer memory architecture designed to make conversational AI systems more reliable when dealing with persistent, changing, temporal, procedural, and conflicting user information.

Rather than treating memory as a simple vector database, the system combines:

```text
Memory Classification
        ↓
Memory Routing
        ↓
Specialized Retrieval
        ↓
Hybrid Fusion
        ↓
Adaptive Reranking
        ↓
Conflict Resolution
        ↓
Context Construction
        ↓
LLM Response Generation
        ↓
Memory Lifecycle Management
```

With a final benchmark result of:

> **Recall@5: 1.000 vs 0.800 baseline**

and:

> **507/507 automated tests passing**

the project provides an evaluation-driven implementation of an adaptive conversational memory architecture.
