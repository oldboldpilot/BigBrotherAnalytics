# BigBrotherAnalytics - Claude AI Guide

**Project:** High-performance AI-powered trading intelligence platform
**Phase:** Tier 1 POC (Weeks 1-12) - DuckDB-first validation
**Budget:** $30,000 Schwab margin account
**Goal:** Prove daily profitability ($150+/day) before scaling

## Core Architecture

**Three Interconnected Systems:**
1. **Market Intelligence Engine** - Multi-source data ingestion, NLP, impact prediction, graph generation
2. **Correlation Analysis Tool** - Statistical relationships, time-lagged correlations, leading indicators
3. **Trading Decision Engine** - Options day trading (initial focus), explainable decisions, risk management

**Technology Stack (Tier 1 POC):**
- **Languages:** C++23 (core), Python 3.14+ GIL-free (ML), Rust (optional)
- **Database:** DuckDB ONLY (PostgreSQL deferred to Tier 2 after profitability)
- **Parallelization:** MPI, OpenMP, UPC++ (32+ cores)
- **ML/AI:** PyTorch, Transformers, XGBoost, SHAP
- **GPU:** CUDA 13.0, vLLM for inference

## Critical Principles

1. **DuckDB-First:** Zero setup time. PostgreSQL ONLY after proving profitability.
2. **Options First:** Algorithmic options day trading before stock strategies.
3. **Explainability:** Every trade decision must be interpretable and auditable.
4. **Validation:** Free data (3-6 months) before paid subscriptions.
5. **Speed:** Microsecond-level latency for critical paths.

## Key Documents

- **PRD:** `docs/PRD.md` - Complete requirements and cost analysis
- **Database Strategy:** `docs/architecture/database-strategy-analysis.md` - DuckDB-first rationale
- **Playbook:** `playbooks/complete-tier1-setup.yml` - Environment setup (DuckDB, no PostgreSQL)
- **Architecture:** `docs/architecture/*` - Detailed system designs

## Current Status (as of 2025-11-06)

**Completed:**
- ✅ Planning phase complete (architecture, PRD, database strategy)
- ✅ DuckDB-first decision made and documented
- ✅ Ansible playbook updated for DuckDB-only Tier 1
- ✅ All documentation updated to reflect DuckDB-first approach

**Next Steps:**
- [ ] Run ansible playbook to set up Tier 1 environment
- [ ] Implement C++23 core components (options pricing, correlation engine)
- [ ] Build Python ML pipeline with DuckDB
- [ ] Validate with 10 years of free historical data

## AI Orchestration System

**For structured development, use the AI orchestration system:**

```
+------------------+
|   Orchestrator   | ← Coordinates all agents
+------------------+
        ↓
+------------------+
|    PRD Writer    | ← Requirements
+------------------+
        ↓
+------------------+
| System Architect | ← Architecture
+------------------+
        ↓
+------------------+
|  File Creator    | ← Implementation
+------------------+
        ↓
+---------------------------+
| Self-Correction (Hooks)   | ← Validation
| Playwright + Schema Guard |
+---------------------------+
```

**Available Agents:**
- `PROMPTS/orchestrator.md` - Coordinates multi-agent workflows
- `PROMPTS/prd_writer.md` - Updates PRD and requirements
- `PROMPTS/architecture_design.md` - Designs system architecture
- `PROMPTS/file_creator.md` - Generates implementation code
- `PROMPTS/self_correction.md` - Validates and auto-fixes code
- `PROMPTS/code_review.md` - Reviews code quality
- `PROMPTS/debugging.md` - Debugs issues systematically

**Workflows:**
- `WORKFLOWS/feature_implementation.md` - Implement new features
- `WORKFLOWS/bug_fix.md` - Fix bugs systematically

**See `ai/README.md` for complete orchestration guide.**

## AI Assistant Guidelines

When helping with this project:
1. Always check database strategy first - use DuckDB for Tier 1, not PostgreSQL
2. Reference `ai/MANIFEST.md` for current goals and active agents
3. Check `ai/IMPLEMENTATION_PLAN.md` for task status and checkpoints
4. Use workflows in `ai/WORKFLOWS/` for repeatable processes
5. **For complex tasks, use the Orchestrator** (`PROMPTS/orchestrator.md`)
6. Focus on validation speed - POC has $30k at stake
