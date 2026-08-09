# Deep Learning Journey

A self-directed roadmap from classical ML through neural networks, LLM APIs, and tool-using agents — working toward a final concurrent multi-agent project.

Each phase is a self-contained folder with Jupyter notebooks and written notes. See [`Final Project.md`](./Final%20Project.md) for the target end-state and design decisions carried through the later phases.

## Phases

| Phase | Topic | Contents |
|---|---|---|
| [Phase 1](./Phase%201) | Classical ML | Iris, Titanic, House Prices |
| [Phase 2](./Phase%202) | Neural Networks | MNIST, Sentiment Analysis, CIFAR-10 |
| [Phase 3](./Phase%203) | Large Pretrained Models | LLM API basics, prompt engineering, embeddings search, RAG pipeline |
| [Phase 4](./Phase%204) | Tool Use & Single-Agent Systems | Tool-calling agent, ReAct loop, coding agent |
| Phase 5 (planned) | Multi-Agent Orchestration | Sequential version of the final project |
| Phase 6 (planned) | Concurrent Multi-Agent Systems | Async version — the final project |

## Final project: Concurrent Research Assistant

Given a broad question, a planner agent breaks it into subtopics, multiple researcher agents investigate concurrently, a critic agent reviews and can send work back for revision, and a writer agent synthesizes the final report — with a supervisor handling retries, timeouts, and cost tracking. Full design notes in [`Final Project.md`](./Final%20Project.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your Anthropic API key
```
