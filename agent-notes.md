## 2025-11-29
- Added semantic scaffolding: new semantic traits/config/store, registration APIs, and bijection enforcement between e-classes and semantic IDs.
- Extended `EGraph` to track semantic configurations/stores; initial unit tests passing.
- Added TDD scaffolding for semantic constructors: simple arithmetic constructor tests in `src/semantic.rs`.
- Implemented `EGraph::semantic_add` to create/merge eq classes based on semantic IDs and added insertion tests ensuring semantic merges happen on equal semantics and stay distinct otherwise.
- Added semantic merge plumbing with decomposer hooks and counting test; semantic store supports rebind/unbind to maintain bijection during merges.
- Prototyped upward semantic recomputation and added (ignored) regression test while iterating; semantic merge now triggers recompute and handles missing semantics more defensively.
- Began DSL exposure: registering a read-only `sem_<EqSort>` function when adding a semantic domain, storing its backend id in `SemanticStore`, and populating it alongside semantic bindings. All tests pass (one recompute test still ignored).
## 2025-12-10
- User asked if they can write a conversation log for a Codex conversation; preparing concise guidance.
## 2025-12-10
- Wired semantics into normal DSL execution: added `semantic_rebuild_all` to compute/bind semantics for backend-inserted rows, reconcile canonical ids, and trigger semantic merges/decomposers; invoked after rule/action/expression evaluation so schedules see semantic merges.
- Added strict mismatch policy (semantic merge errors when IDs differ) and tests for both conflict errors and rebuild path; semantic merge tests adjusted accordingly.
- Hooked semantic recomputation into rule runs to propagate `updated` when semantics change; added new backend-insertion semantic merge test and ensured semantic rebuild handles deferrals.
