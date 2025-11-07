# BigBrotherAnalytics AI Orchestration System

This directory contains the AI agent orchestration system for structured, high-quality development of BigBrotherAnalytics.

---

## Overview

The AI orchestration system coordinates multiple specialized AI agents to handle complex development tasks with consistency, quality, and automation.

### Orchestration Hierarchy

```
+------------------+
|   Orchestrator   | ← Coordinates all agents
+------------------+
        |
        v
+------------------+
|    PRD Writer    | ← Requirements & specifications
+------------------+
        |
        v
+------------------+
| System Architect | ← Architecture design
+------------------+
        |
        v
+------------------+
|  File Creator    | ← Implementation
+------------------+
        |
        v
+---------------------------+
| Self-Correction (Hooks)   | ← Validation & quality
| Playwright + Schema Guard |
+---------------------------+
```

---

## Directory Structure

```
ai/
├── README.md                    # This file
├── CLAUDE.md                    # Always-loaded guide for AI assistants
├── MANIFEST.md                  # Project goals and active agents
├── IMPLEMENTATION_PLAN.md       # Detailed task breakdown and checkpoints
├── PROMPTS/                     # Agent-specific prompts
│   ├── orchestrator.md          # Coordinates multi-agent workflows
│   ├── prd_writer.md            # Creates/updates requirements docs
│   ├── architecture_design.md   # Designs system architecture
│   ├── file_creator.md          # Generates implementation code
│   ├── self_correction.md       # Validates and auto-fixes code
│   ├── code_review.md           # Reviews code quality
│   └── debugging.md             # Debugs issues systematically
└── WORKFLOWS/                   # End-to-end workflow guides
    ├── feature_implementation.md  # Implement new features
    └── bug_fix.md                 # Fix bugs systematically
```

---

## Agent Descriptions

### 1. Orchestrator (`PROMPTS/orchestrator.md`)
**Purpose:** Coordinate complex multi-agent workflows

**When to Use:**
- Implementing complex features requiring multiple agents
- Large refactoring efforts
- System-wide changes

**Capabilities:**
- Task decomposition
- Agent coordination
- Progress tracking
- Error handling and recovery

**Example:**
```
I need to implement the Options Pricing Engine end-to-end.
Please orchestrate the implementation with all necessary agents.
```

---

### 2. PRD Writer (`PROMPTS/prd_writer.md`)
**Purpose:** Create and maintain Product Requirements Documents

**When to Use:**
- New feature requests
- Requirement changes
- Clarifying scope
- Documenting decisions

**Capabilities:**
- Captures complete requirements
- Defines success metrics
- Documents constraints
- Maintains changelog

**Example:**
```
Update the PRD to include real-time options pricing requirements
with performance targets.
```

---

### 3. System Architect (`PROMPTS/architecture_design.md`)
**Purpose:** Design system architecture and component interactions

**When to Use:**
- New component design
- System integration
- Performance optimization
- Technology decisions

**Capabilities:**
- Component architecture design
- API design (C++ and Python)
- Database schema design
- Performance analysis
- Risk assessment

**Example:**
```
Design the architecture for the Correlation Engine using MPI and OpenMP.
Target: 1000x1000 correlation matrix in < 10 seconds.
```

---

### 4. File Creator (`PROMPTS/file_creator.md`)
**Purpose:** Generate production-quality implementation code

**When to Use:**
- Implementing designs
- Creating new components
- Generating tests
- Boilerplate generation

**Capabilities:**
- C++23 code generation
- Python 3.14+ code generation
- Unit test generation
- CMake/build configuration
- Python bindings (pybind11)

**Example:**
```
Generate the Options Pricing Engine implementation based on
docs/architecture/options-pricing-engine.md
```

---

### 5. Self-Correction (`PROMPTS/self_correction.md`)
**Purpose:** Automated validation, testing, and quality assurance

**When to Use:**
- After code generation
- Before commits
- Pre-push validation
- CI/CD pipeline

**Capabilities:**
- Static analysis (clang-tidy, ruff, mypy)
- Schema validation (DuckDB)
- Unit/integration/performance testing
- Auto-fixing (formatting, linting)
- Security scanning

**Example:**
```
Run self-correction validation on the Options Pricing Engine
implementation before committing.
```

---

### 6. Code Reviewer (`PROMPTS/code_review.md`)
**Purpose:** Review code for quality, performance, and correctness

**When to Use:**
- Post-implementation review
- Pre-commit checks
- Performance audits
- Security reviews

**Capabilities:**
- C++23 best practices review
- Python code quality review
- Financial calculation verification
- Performance analysis
- Security checks

**Example:**
```
Review src/cpp/options/trinomial_tree.cpp for performance
and correctness.
```

---

### 7. Debugger (`PROMPTS/debugging.md`)
**Purpose:** Systematic debugging and issue resolution

**When to Use:**
- Bug reports
- Performance issues
- Crashes
- Test failures

**Capabilities:**
- Root cause analysis
- Memory leak detection
- Performance profiling
- Fix implementation
- Regression test generation

**Example:**
```
Debug the segfault in correlation engine when processing
1000+ symbols with MPI.
```

---

## Workflows

### Feature Implementation (`WORKFLOWS/feature_implementation.md`)

Complete workflow for implementing new features:
1. Orchestrator coordinates agents
2. PRD Writer updates requirements (if needed)
3. System Architect designs architecture
4. File Creator generates implementation
5. Self-Correction validates code
6. Commit to git and push

**Estimated Time:** 1-4 hours

---

### Bug Fix (`WORKFLOWS/bug_fix.md`)

Systematic workflow for fixing bugs:
1. Orchestrator analyzes bug report
2. Debugger identifies root cause
3. File Creator implements fix
4. Self-Correction validates fix
5. Commit with regression test

**Estimated Time:** 30 minutes to 4 hours

---

## Usage Patterns

### Simple Task (Single Agent)

For straightforward tasks, invoke a single agent directly:

```
Please review src/python/data_ingestion/yahoo_finance.py
using the code review prompt. Focus on error handling.
```

### Complex Task (Orchestrated)

For complex tasks, let Orchestrator coordinate:

```
I need to implement the complete Market Intelligence Engine
with NLP, sentiment analysis, and impact prediction.

Please orchestrate the full implementation using:
- PRD Writer (update requirements)
- System Architect (design architecture)
- File Creator (generate code)
- Self-Correction (validate)
```

---

## Integration with Development

### Local Development

```bash
# 1. Implement feature using AI agents
# (follow feature_implementation.md workflow)

# 2. Self-correction runs automatically via git hooks
# (installed by scripts/install_hooks.py)

# 3. Commit and push
git commit -m "feat: Add new feature"
git push origin master
```

### CI/CD Integration

The Self-Correction system integrates with GitHub Actions:

```yaml
# .github/workflows/validate.yml
# - Runs on every push
# - Blocks PR merge if validation fails
# - Reports results in PR comments
```

---

## Best Practices

### 1. Always Start with Orchestrator for Complex Tasks
Let the Orchestrator decide which agents to invoke and in what order.

### 2. Keep Agents Focused
Each agent has a specific responsibility. Don't mix concerns.

### 3. Validate Early and Often
Run Self-Correction after every code generation, not just before commit.

### 4. Document Decisions
PRD Writer and System Architect should document all architectural decisions.

### 5. Test Thoroughly
File Creator should generate comprehensive tests, not just happy paths.

---

## Configuration

### Agent Settings

Agents can be configured via environment variables or config files:

```bash
# Example: .env
AI_ORCHESTRATOR_MODE=sequential  # or parallel
AI_AUTO_FIX=true                 # Auto-fix simple issues
AI_VALIDATION_LEVEL=strict       # or relaxed
```

### Git Hooks

Install self-correction hooks:

```bash
python scripts/install_hooks.py

# Hooks will run on:
# - pre-commit: Format, lint, unit tests
# - pre-push: Full validation, integration tests
```

---

## Troubleshooting

### Agent Not Found

**Problem:** "Agent X not available"

**Solution:** Check that `PROMPTS/[agent_name].md` exists

---

### Validation Failing

**Problem:** Self-Correction reports failures

**Solution:**
1. Review validation report
2. Check which specific checks failed
3. Fix issues manually or route back to appropriate agent
4. Re-run validation

---

### Orchestration Hangs

**Problem:** Orchestrator stuck or not progressing

**Solution:**
1. Check for circular dependencies between agents
2. Verify all required inputs are available
3. Check agent-specific error logs

---

## Metrics and Monitoring

Track AI orchestration effectiveness:

```python
metrics = {
    'features_implemented': 0,
    'bugs_fixed': 0,
    'auto_fixes_applied': 0,
    'validation_success_rate': 0.0,
    'average_time_per_feature_hours': 0.0,
}
```

---

## Extending the System

### Adding New Agents

1. Create `PROMPTS/new_agent.md` with agent specification
2. Update `PROMPTS/orchestrator.md` with new agent details
3. Create workflow using the new agent (optional)
4. Update this README

### Adding New Workflows

1. Create `WORKFLOWS/new_workflow.md`
2. Document step-by-step process
3. Include example execution
4. Update this README

---

## References

- **PRD:** `docs/PRD.md`
- **Architecture:** `docs/architecture/*`
- **Implementation Plan:** `ai/IMPLEMENTATION_PLAN.md`
- **Manifest:** `ai/MANIFEST.md`

---

## Version History

**v1.0.0 (2025-11-06):**
- Initial AI orchestration system
- 7 specialized agents
- 2 complete workflows
- Self-correction with git hooks

---

**For questions or issues with the AI orchestration system, consult this README or the specific agent/workflow documentation.**
