Final Project Notes (Written by Anthony)

Target for after Phase 6. Full phase-by-phase roadmap lives in `~/Downloads/ml-to-multiagent-roadmap.md`.

## The project: Concurrent Research Assistant

Give it a broad question (e.g. "compare the top 5 EV battery technologies") and it:

1. A planner agent breaks the question into subtopics (one per battery technology).
2. Five researcher agents run concurrently, each investigating one subtopic (web search, tool use, maybe reading PDFs).
3. As each researcher finishes, its output feeds into a shared results store — this is where race conditions and out-of-order completion get handled for real.
4. A critic agent reviews each research output for quality/completeness and can kick a task back to its researcher for another pass — reuses the Phase 5 feedback loop, but now inside a concurrent system.
5. Once all subtopics are done, a writer agent synthesizes everything into a final report.
6. A supervisor monitors progress, handles retries if an agent times out or an API call fails, and tracks total token/cost spend across the whole run.

**Why this project specifically:**
- Requires real concurrency (5 researchers running at once), not fake concurrency that could've been sequential.
- Has a natural shared resource (the results store), so locking/race conditions are unavoidable, not theoretical.
- Has real failure modes worth handling (a researcher times out, a critic loops forever) — which is most of what makes concurrent agent systems hard in practice.
- The output (a written report) is genuinely useful, so it's worth iterating on rather than abandoning once it "works."

## Design decisions from review (apply when building Phase 5/6)

1. **Build Phase 5 and Phase 6 as the same codebase.** Phase 5 = this exact pipeline, sequential (plain for-loop, one researcher at a time). Phase 6 = turn that loop into `asyncio.gather` + a locking layer around the shared store. Don't build two separate projects — the diff between the two versions *is* the concurrency lesson.
2. **Hard-cap the critic↔researcher revision loop.** e.g. max 2 revision passes, then the critic accepts with noted caveats or escalates to the supervisor. Design this in from the start — in Phase 6 a stuck critic loop also holds up a worker slot while other researchers are trying to finish, so it's a concurrency bug, not just an agent-design nuisance.
3. **Keep the shared results store in-process** — an `asyncio.Lock` around a plain dict, not a real database. A database solves the race condition for you; the point is to learn why the race condition existed.
4. **Start tracking token/cost spend in Phase 3**, the first time a real metered API gets called — not bolted on as a Phase 6 supervisor feature. Build the habit early so it's not a retrofit.
5. **Add structured logging as an explicit Phase 6 requirement** — a trace ID per subtopic and a timestamped log per agent-transition. With 5 concurrent agents, retries, and a critic loop, the hard part is figuring out *why* a run went wrong after the fact.
6. **Treat PDF-reading for researcher agents as optional stretch scope**, not core. Web search + text tools alone still fully exercises Phases 4-6; PDF parsing is orthogonal to the orchestration lessons this project is actually for.

## Progress log

- 2026-08-08: Phase 1 (Classical ML) complete and reviewed.
- 2026-08-08: Phase 2 (Neural Networks) in progress — MNIST specimen solved and reviewed, sentiment analysis specimen in progress.
- 2026-08-09: Phase 3 (Working With Large Pretrained Models) scaffolded — folder, venv packages (anthropic, python-dotenv, sentence-transformers), `.env`/`.gitignore`, and 4 starter notebooks created. Not yet attempted. Phase 2's CIFAR-10 specimen still finishing in the background.
- 2026-08-09: Phase 3 solved (code written for all 4 notebooks — API basics, prompt engineering, embeddings search, RAG pipeline) but not executed/run yet, per request. Phase 4 (Tool Use & Single-Agent Systems) scaffolded — folder + 3 starter notebooks (tool-calling agent, ReAct loop, coding agent). No new packages needed, reuses the Anthropic SDK from Phase 3.
