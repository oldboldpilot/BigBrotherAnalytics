# Quick Start: Using AI Agents After Cloning

**After cloning BigBrotherAnalytics to a new machine, the AI agents are ready to use immediately.**

---

## Step 1: Clone Repository

```bash
git clone https://github.com/oldboldpilot/BigBrotherAnalytics.git
cd BigBrotherAnalytics
```

---

## Step 2: Verify AI System

```bash
# Check AI directory exists
ls -la ai/

# Should show:
# - PROMPTS/ (7 agent prompts)
# - WORKFLOWS/ (2 workflow guides)
# - README.md, CLAUDE.md, MANIFEST.md, IMPLEMENTATION_PLAN.md
```

---

## Step 3: AI Agents Are Ready!

**No installation needed for AI agents** - they're markdown files that work immediately.

### Using with AI Assistant (e.g., Claude Code)

**Simple Task:**
```
Please review the correlation engine code using the code review prompt
from ai/PROMPTS/code_review.md
```

**Complex Task:**
```
I need to implement the Options Pricing Engine end-to-end.
Please use the orchestration system in ai/PROMPTS/orchestrator.md
to coordinate implementation.
```

**Following a Workflow:**
```
Please follow the feature implementation workflow in
ai/WORKFLOWS/feature_implementation.md to build the Market Intelligence Engine.
```

---

## Step 4: Set Up Development Environment (Optional)

**Only needed when you're ready to run code:**

```bash
# Install C++23, Python 3.14+, DuckDB, etc.
ansible-playbook playbooks/complete-tier1-setup.yml
```

**But AI agents work BEFORE environment setup** - you can plan, design, and generate code immediately after cloning!

---

## What's Included (Portable)

✅ **Immediately Available After Clone:**
- 7 AI agent prompts with complete C++23 standards
- 2 end-to-end workflows
- Complete PRD with all requirements
- 5 architecture documents
- Implementation plan with task breakdown
- All coding standards (fluent APIs, modules, trailing return, Core Guidelines, STL-first)

⏸️ **Requires Environment Setup (Later):**
- C++23 compiler
- Python 3.14+
- Build tools
- DuckDB

---

## Examples

### Example 1: New Developer Joins Team

```bash
# Day 1 - Clone repo
git clone https://github.com/oldboldpilot/BigBrotherAnalytics.git
cd BigBrotherAnalytics

# Read AI guides
cat ai/CLAUDE.md          # Quick reference
cat ai/README.md          # Full orchestration guide
cat ai/MANIFEST.md        # Current project status

# Start working with AI assistant immediately
# "Please review ai/IMPLEMENTATION_PLAN.md and show me what tasks are pending"
```

### Example 2: Moving to Production Server

```bash
# On production server
git clone https://github.com/oldboldpilot/BigBrotherAnalytics.git
cd BigBrotherAnalytics

# AI agents work immediately
# Generate code using AI agents:
# "Please implement correlation engine using ai/PROMPTS/file_creator.md"

# Then set up environment:
ansible-playbook playbooks/complete-tier1-setup.yml
```

### Example 3: Different AI Assistant

The prompts work with any AI assistant:
- Claude Code (this is designed for it)
- ChatGPT Code Interpreter
- GitHub Copilot Chat
- Any AI assistant that can read markdown files

Just point the AI to the appropriate prompt file.

---

## Verification

After cloning, verify everything is ready:

```bash
# Check file count
find ai/ -name "*.md" | wc -l
# Should output: 13

# Check total documentation size
wc -l ai/**/*.md ai/*.md | tail -1
# Should be ~5,200 lines

# Quick test - read a prompt
cat ai/PROMPTS/orchestrator.md | head -20
```

---

## All Set!

The AI orchestration system is **100% portable** and ready to use on any machine immediately after `git clone`. No installation, no configuration, no external dependencies.

**Just clone and start building with AI assistance!** 🚀
