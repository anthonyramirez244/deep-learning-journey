Handoff — ML → Multi-Agent Roadmap (read this first in a new session)

Workspace: `~/Desktop/Machine Learning/` — open `Machine Learning.code-workspace` in VS Code.
Roadmap spec (phase-by-phase projects/tools/benchmarks): `~/Downloads/ml-to-multiagent-roadmap.md`
Final project design + review notes: `Final Project.md` (workspace root)
Shared Python env: `.venv/` at workspace root, one `.env` for the Anthropic API key (gitignored).

## Conventions established so far — follow these without re-deriving them

- Model: `claude-opus-5` everywhere. Pricing used for cost math: $5/$25 per MTok (input/output).
- Each phase gets: a folder, a "field journal" HTML guide published as a Claude Artifact (paper/pine-green naturalist theme, localStorage checkbox progress), and starter notebooks with markdown workflow-step headers + empty code cells.
- Two-step pattern per phase: (1) scaffold the folder + guide + empty notebooks first, (2) only solve/execute notebooks when explicitly asked.
- When solving notebooks: real working code, **no comments** (explicit preference), verified by `compile()` before calling it done. Execute for real when asked to "solve and run"; when asked to "solve, don't run," only write the code.
- Kernel for all notebooks: "Machine Learning (Roadmap)" (shared across phases, registered once).
- API key lives in `.env`, loaded via `python-dotenv`, never printed/hardcoded. `.gitignore` already excludes it.

## Progress by phase

**Phase 1 — Classical ML: done.** Iris/Titanic/House Prices solved by the user, reviewed in detail. A few minor non-blocking gaps were noted (e.g. Iris split wasn't actually stratified despite the label) but nothing was required to fix.

**Phase 2 — Neural Networks: done, all 3 specimens executed with real results.**
- MNIST: MLP 97.38% val acc, CNN 99.29% test acc.
- Sentiment (IMDB): bag-of-embeddings 80.2% val acc beat the LSTM (79.3% val, 81.1% test) despite the LSTM taking ~22x longer to train — the intended lesson landed.
- CIFAR-10: CNN+BatchNorm, 78.9% test accuracy (beat the 70-75% guide target). Most confused pair: dog→cat (203 cases). This finished in the background and was **not yet reported to the user before this handoff** — worth mentioning if picking this up fresh.

**Phase 3 — Working With Large Pretrained Models: done, all 4 notebooks executed for real against the live API.**
- 01 API basics: real key confirmed working, session cost tracking correct (~$0.006 for the notebook's calls).
- 02 prompt engineering: zero-shot/few-shot hit 100% on the fixed 8-message eval set, CoT/strict-format hit 87.5% — the one common miss across all four ("Do you offer a student discount?", true label `other`) is a genuinely ambiguous case, not a bug.
- 03 embeddings search: semantic search correctly beats keyword search on the intended contrast query ("How did people first travel to Earth's natural satellite?" — keyword search gets pulled toward unrelated chunks by generic word overlap, semantic search nails Apollo 11).
- 04 RAG pipeline: retrieval precision@3 = 6/6 on the eval set; unanswerable-question test correctly triggers "I don't know" instead of hallucinating. Interesting real finding: on "how much oxygen does the Amazon produce," RAG faithfully repeats the provided (commonly-cited but oversimplified) 20% figure, while the no-RAG answer actually corrects that popular myth with more nuance — good evidence that grounding makes answers more citable, not automatically more correct.

Two real bugs were found and fixed by actually running this against the live API (not caught by the earlier "verified to compile" pass):
1. `claude-opus-5` defaults to extended thinking. Low `max_tokens` budgets (100–200) could get fully consumed by thinking tokens before any visible text was produced, silently returning empty strings. Fixed by passing `thinking={"type": "disabled"}` on every `messages.create()` call across notebooks 01, 02, and 04 — appropriate here since none of these Phase 3 exercises are about extended thinking itself.
2. Notebook 02's `extract_category` regex grabbed the *first* category word mentioned anywhere in the model's chain-of-thought reasoning, not the actual final answer — badly understating CoT's real accuracy. Fixed to prefer the "Category: X" line, with the old first-match behavior as fallback for the other prompt styles. Also had to raise that cell's `max_tokens` 200→700 since the CoT prompt was getting truncated before reaching its conclusion at lower budgets.

Environment note: `sentence_transformers`' default download path uses HF's newer `hf-xet` chunked transfer protocol, which hung indefinitely (0 bytes transferred) in this environment. Fixed by setting `HF_HUB_DISABLE_XET=1`, which falls back to plain HTTPS. Worth setting that env var proactively before Phase 5/6 if they use HF models too.

**Phase 4 — Tool Use & Single-Agent Systems: done, all 3 notebooks executed for real against the live API.** Safety pass (from before execution) found and fixed real issues:
- Calculator `eval()` could be driven into unbounded exponentiation (`"9**9**9**9"`) — fixed with a length cap + `**` block, verified locally.
- Several agent loops (`call_with_tools`, `call_with_tools_safe`, `react_loop_basic`, `react_loop_logged`) had no step cap at all — all now capped (`max_steps`, raises `RuntimeError` past the limit).
- The coding agent's sandboxing (timeout + forbidden-pattern blocklist) was only wired into 1 of 3 exercises — now baked into the single `run_python` function used everywhere.
- Known remaining limitation, accepted as-is: the blocklist is substring-based, not a real sandbox — fine for this notebook's own benign self-directed use, not for untrusted input.

Same `thinking={"type": "disabled"}` fix from Phase 3 was applied proactively to all `messages.create()` calls across all 3 notebooks before running (same root cause as Phase 3's bug: `claude-opus-5` defaults to extended thinking, which can silently eat small `max_tokens` budgets before a tool_use or text block appears).

Results, all executed for real:
- 01 tool-calling agent: calculator + lookup tool both worked correctly, including the multi-tool `call_with_tools` wrapper. Real, notable finding: the deliberate "make a tool fail" test (`error_case + 5`) never actually exercised the `is_error` path — the model recognized the expression was invalid and declined to call the calculator at all, reasoning about it in text instead. The error-handling code itself is still correctly wired, just not exercised by this prompt against this model.
- 02 ReAct loop: multi-step tool chaining worked (price lookups → arithmetic → correct final answer), step-by-step logging worked, distractor tool (`get_weather`) correctly ignored. Same pattern as 01: the "confirm the step cap actually triggers" test (an unreachable "count to 1,000,000" goal) did NOT trigger the cap — the model recognized the task was infeasible after step 1 and refused outright rather than blindly looping. `Cap triggered: False` here is the good outcome, not a bug; the cap logic itself is sound, just untested by a model that no longer needs it.
- 03 coding agent: Fibonacci computed correctly with a clear note on indexing convention; the deliberately-broken median code threw a real `IndexError`, and the agent read the traceback and fixed it (correct median 15.5, matched `statistics.median`); sandbox tests confirmed both the timeout and forbidden-pattern blocks fire correctly, normal execution unaffected; linear regression task correctly reached for numpy and reported slope/intercept/R² with sensible interpretation.

Recurring theme worth remembering for Phase 5/6 design: `claude-opus-5` is capable enough that several of these notebooks' "does the safety mechanism actually trigger" tests no longer trigger, because the model just doesn't take the bait a weaker model would have. The mechanisms are still correctly implemented and would fire if needed — this is model capability outpacing the test design, not fragility.

**Phase 5 — Multi-Agent Coordination (Sequential): done, all 4 notebooks solved and executed for real against the live API.** Folder + 4 notebooks (`01_researcher_writer_handoff.ipynb`, `02_planner_sequential_workers.ipynb`, `03_critic_feedback_loop.ipynb`, `04_sequential_research_assistant.ipynb`), plus a published field journal guide. No new packages needed (reused `anthropic` + `python-dotenv`).

Results, all executed for real:
- 01 researcher→writer handoff: schema-enforced JSON handoffs worked cleanly across all 4 battery-tech topics; ~$0.11 total.
- 02 planner→sequential workers: real dependency confirmed (`Results differ meaningfully: True` on the with/without-context ablation); planner cap held at 5 even on a deliberately oversized "chef guide" prompt; ~$0.15 total.
- 03 critic feedback loop: unlike Phase 4's traps, this rubric-based one produced a *real* catch — attempt 1 blew the 40-word limit 3x over, used the wrong battery figure, and omitted the price entirely; attempt 2 fixed all of it and was approved. The 2-revision cap itself didn't trigger this run (worker self-corrected), which the notebook's own step 6 cell reports honestly rather than claiming success it didn't have.
- 04 sequential research assistant (the actual final-project pipeline, sequential form): ran end-to-end on "compare Django/Flask/FastAPI/Tornado for a 10k req/s REST API" — 5 subtopics researched and critic-approved (0 revisions needed), synthesized into a coherent comparative report with a real markdown benchmark table, 0 retries needed, 12 total API calls, ~$0.23 for the full run.

Two real bugs surfaced only by actually running Phase 5, both variations on "the planner/researcher writes more than the token budget allows, breaking the JSON parse":
1. `run_planner`'s default `max_tokens=800` was too tight for a 6-subtask itinerary plan — `json.loads` failed on an unterminated string mid-generation. Fixed by raising to `max_tokens=2000` (empirically the model needs 600-900 output tokens for a well-formed plan; 2000 gives real headroom without being wasteful).
2. Specimen 04's `run_researcher`, given the planner's own multi-clause technical subtopic descriptions (e.g. "cover architecture, benchmarks, deployment, caching, scaling..."), tried to write an exhaustive essay regardless of token budget — even 1800 tokens wasn't enough. Chasing a bigger budget was the wrong fix (unbounded demand vs. bounded supply); the real fix was bounding the *demand*: the researcher's system prompt now asks for **exact** counts ("exactly 4 key_facts", "exactly 1-2 sentences") rather than "at most," which the model reliably honors, combined with `max_tokens=1200` as a safety margin. Also worth noting for later phases: `output_config`'s JSON schema does **not** support `maxItems` on arrays (`400 invalid_request_error` if you try) — array length has to be controlled by instruction, not schema.

Every notebook's setup cell bakes in this session's lessons from Phase 3/4 directly, instead of patching them in after a bug surfaces:
- A shared `call_model(messages, tools=None, max_tokens=800, output_schema=None)` wrapper defaults to `thinking={"type": "disabled"}` on every call.
- The same wrapper makes schema-enforced JSON output (`output_schema=...` → `output_config`) a one-liner, so agent-to-agent handoffs default to typed JSON instead of regex-parsed prose — the direct fix for Phase 3's CoT grading bug.
- `call_cost()` is present from the first cell, continuing the Phase 3/4 habit instead of bolting cost tracking on later.
- Specimen 03's step 4 explicitly calls out the Phase 4 lesson about safety-cap tests: use an objective, checkable rubric a good-faith attempt is likely to genuinely miss, not a trap the model can just decline (which is what silently invalidated two of Phase 4's own guardrail tests).

Specimen 04 (`04_sequential_research_assistant.ipynb`) is deliberately the literal sequential version of the Final Project's Concurrent Research Assistant — planner → one-researcher-at-a-time loop → shared in-process store (built as its own small class so Phase 6 can wrap it in an `asyncio.Lock` without a rewrite) → capped critic loop → writer → lightweight supervisor with retries and cost tracking. Per the final-project review note: Phase 6 should extend this exact codebase (loop → `asyncio.gather`, dict → locked store), not start a second project.

**Phase 6 — Concurrent Multi-Agent Systems: done, all 4 notebooks solved and executed for real against the live API. This closes out the entire roadmap — Phases 1-6 are all complete.** Folder + 4 notebooks (`01_parallel_workers.ipynb`, `02_shared_store_concurrency.ipynb`, `03_task_queue_supervisor.ipynb`, `04_concurrent_research_assistant.ipynb`), plus a published field journal guide. No new packages (`asyncio` is stdlib, `AsyncAnthropic` ships in the already-installed `anthropic` package).

Results, all executed for real:
- 01 parallel workers: sequential 22.0s vs. concurrent 5.6s summarizing 7 documents — a real, measured 3.9x speedup. `return_exceptions=True` correctly isolated a simulated failure (6/7 succeeded, 1 reported cleanly) without losing the successful results.
- 02 shared store under concurrency: the race condition reliably reproduced (50 increments → final value 1, 49 lost) and was reliably fixed by `asyncio.Lock` (final value 50, 0 lost) — not a one-off, structurally guaranteed by the `sleep(0)` yield point between read and write. Real concurrent agent writes into the locked store: 5/5 entries present, no losses.
- 03 task queue & supervisor: dynamic load balancing actually showed up unprompted (4/2/2 task split across 3 workers, not an even 3/3/2) — the pull model in action, not asserted. Completion order genuinely differed from submission order. The poison task was retried once then failed permanently, exactly as designed. The SLOWPOKE task timed out twice (initial + 1 retry) without stalling the 4 other tasks running alongside it.
- 04 concurrent research assistant (the actual final project, now concurrent): the whole Phase 5 pipeline converted cleanly — same planner, same critic, same writer, sequential loop → `asyncio.gather`, `ResearchStore` → locked, supervisor gained retry+timeout. Structured logging with trace IDs worked as designed: raw interleaved log plus a clean filter-back-to-one-subtopic demonstration. The simulated flaky-subtopic failure was caught by the supervisor and succeeded on retry. **The headline result, measured fairly in the same session on the same 5 subtopics: sequential research phase 93.8s, concurrent research phase 21.0s — a real 4.5x speedup.** Full end-to-end run: 5/5 subtopics researched and approved on first attempt, 0 retries needed, $0.221 total cost, 44.2s wall-clock for the complete pipeline (planner + research + writer).

No new bugs surfaced by running Phase 6 — the lessons baked in proactively from Phase 5 (generous `max_tokens`, exact-count instructions, async client, semaphore rate limiting) held up cleanly across all 4 notebooks with zero fixes needed mid-run, a first for this project.

Setup cells bake in Phase 3-5's lessons plus what's new for real concurrency:
- `call_model` is now `async`, built on `AsyncAnthropic` instead of the sync client — `asyncio.gather` needs genuine awaitable coroutines, wrapping a blocking sync call wouldn't actually overlap requests.
- A `Semaphore(5)` caps in-flight requests, since `gather` fires every call at once by default — the fastest way to get rate-limited.
- Default `max_tokens` raised to 1200 (Phase 5's fix, applied proactively this time), `thinking` disabled by default, schema-enforced JSON handoffs via `output_schema` — all carried forward unchanged.
- The guide explicitly reminds: `output_config`'s JSON schema doesn't support `maxItems` — bound array/response length by instruction, not schema.

Specimen 04 (`04_concurrent_research_assistant.ipynb`) is explicitly framed as a **conversion** of Phase 5 Specimen 04, not a new build, per the final-project design decision: planner and writer carry over unchanged, the sequential critic-gated research loop becomes `asyncio.gather`, `ResearchStore` gets Specimen 02's lock, and the supervisor gains Specimen 03's retry-once/per-task-timeout pattern. Step 7 explicitly asks for an honest wall-clock comparison against the Phase 5 sequential run — including reporting if concurrency doesn't actually help, rather than assuming it will.

Also fixed proactively this time (learned from Phase 5): `Machine Learning.code-workspace` only listed folders through Phase 5, so Phase 6 wouldn't have shown up in VS Code without a manual edit — added it to the workspace file's `folders` list before reporting the scaffold as done, instead of waiting for the user to hit the same problem again.

## Published field journal artifacts (Claude Artifacts, private)
- Phase 1: https://claude.ai/code/artifact/d00b95b2-5a7a-4fd2-ac75-09bd48c5227f
- Phase 2: https://claude.ai/code/artifact/bbbb9993-7d65-4cae-9263-e027948bc93f
- Phase 3: https://claude.ai/code/artifact/dc20d8a5-09c1-41ee-acc1-87bf2f85b40c
- Phase 4: https://claude.ai/code/artifact/39c976ab-09cf-48b6-ab8a-ff5d14ca7027
- Phase 5: https://claude.ai/code/artifact/01f84837-0b48-40b3-a3a1-109065de5d9c
- Phase 6: https://claude.ai/code/artifact/cf451fd8-0806-4e54-bb20-c6eaeac936f9
- "Watching It Think" (live Phase 2 MLP visualization): https://claude.ai/code/artifact/f2edeb5a-4f2c-414f-9187-396310a1db89

## Open loose ends for the next session
**The roadmap is complete — Phases 1 through 6 are all done and executed for real.** There is no mandatory next phase. Anything from here is optional follow-up:
1. A live "Watching It Think" MLP visualization (draw a digit, watch real Phase 2 weights fire) was published as a Claude Artifact — worth the same treatment for the CNN (feature-map view) if the user wants it.
2. The Concurrent Research Assistant (Phase 6 Specimen 04) is real, working code — could be lifted out of the notebook into a standalone script/small CLI tool if the user wants to actually use it rather than leave it as a learning exercise.
3. Possible extensions beyond the original roadmap, only if the user asks: a real task queue backend (Celery, per the roadmap's own "if you want to go further" note), a real vector/document store instead of the in-process dict, or trying an agent framework (AutoGen/CrewAI) now that the fundamentals are built by hand and it'd be clear what such a framework is actually doing for you.
4. `Machine Learning.code-workspace`'s `folders` list is up to date through Phase 6 — if any future phase gets added, remember to update it at the same time (Phase 5 caught the user out on this once).

## How to resume
Point the new session at this file (or just say "check HANDOFF.md") plus `Final Project.md` for the fuller design notes. Claude Code's own memory for this project (`ml_multiagent_roadmap.md` in its memory store) also carries a summary of this, but this file is the durable, portable copy.
