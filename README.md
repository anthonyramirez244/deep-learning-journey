# Deep Learning Journey

A self-directed roadmap from classical ML through neural networks, LLM APIs, and tool-using agents, in a straight line toward one goal: a working concurrent multi-agent system. Six phases, each one built specifically to leave the next phase something real to extend — not six unrelated exercises.

## How each phase was built

Every phase followed the same two-step process: scaffold first (a folder, a field-journal guide, starter notebooks with the workflow laid out step by step), then solve and run for real only once the shape of the phase was settled. Executing for real mattered — several of the most useful findings in this repo only showed up by actually running the code against a live API, not by reading it:

- **Phase 3** found that `claude-opus-5` defaults to extended thinking, which can silently consume a small `max_tokens` budget before any visible output appears — and that a chain-of-thought grading script can badly understate a prompt's real accuracy if it grades the *first* mentioned answer instead of the final one.
- **Phase 4** found that two of its own safety-guardrail tests didn't actually trigger — not because the guardrails were broken, but because the model was capable enough to recognize a bad-faith task and decline it outright, rather than needing the safety net to catch it.
- **Phase 5** confirmed the fix for Phase 4's finding: a rubric-based test the model can't see in advance produces a real catch-and-revise cycle, where a "does the model just avoid the trap" test doesn't.
- **Phase 6** carried every one of those lessons into its own starting point *proactively*, not reactively — and became the first phase where nothing needed fixing mid-run.

Each phase's actual results, bugs, and design decisions were logged in [`HANDOFF.md`](./HANDOFF.md) and [`Final Project.md`](./Final%20Project.md) as they happened, not written up after the fact.

## Phases

| Phase | Topic | Real result |
|---|---|---|
| [Phase 1](./Phase%201) | Classical ML | Iris, Titanic, House Prices |
| [Phase 2](./Phase%202) | Neural Networks | MNIST (99.29% CNN test accuracy), Sentiment Analysis, CIFAR-10 (78.9%, beat the target range) |
| [Phase 3](./Phase%203) | Large Pretrained Models | LLM API basics, prompt engineering, embeddings search, a RAG pipeline |
| [Phase 4](./Phase%204) | Tool Use & Single-Agent Systems | A tool-calling agent, a ReAct loop, a sandboxed self-correcting coding agent |
| [Phase 5](./Phase%205) | Sequential Multi-Agent Coordination | Researcher→writer handoffs, planner→worker chains, a capped critic feedback loop, and the sequential version of the final project |
| [Phase 6](./Phase%206) | Concurrent Multi-Agent Systems | The same pipeline, made concurrent — **93.8s sequential → 21.0s concurrent, a real 4.5x speedup** |

## The final project: Researcher Model

Phase 6's final project — a planner, concurrent critic-gated researchers, a lock-protected shared store, a writer agent, and a supervisor handling retries, timeouts, and cost tracking — turned out real enough to live on its own. It's now a standalone tool:

**→ [researcher-model](https://github.com/anthonyramirez244/researcher-model)**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your Anthropic API key
```
