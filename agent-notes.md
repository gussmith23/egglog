## 2025-11-29
- Added semantic scaffolding: new semantic traits/config/store, registration APIs, and bijection enforcement between e-classes and semantic IDs.
- Extended `EGraph` to track semantic configurations/stores; initial unit tests passing.
- Added TDD scaffolding for semantic constructors: simple arithmetic constructor tests in `src/semantic.rs`.
- Implemented `EGraph::semantic_add` to create/merge eq classes based on semantic IDs and added insertion tests ensuring semantic merges happen on equal semantics and stay distinct otherwise.
- Added semantic merge plumbing with decomposer hooks and counting test; semantic store supports rebind/unbind to maintain bijection during merges.
- Prototyped upward semantic recomputation and added (ignored) regression test while iterating; semantic merge now triggers recompute and handles missing semantics more defensively.
- Began DSL exposure: registering a read-only `sem_<EqSort>` function when adding a semantic domain, storing its backend id in `SemanticStore`, and populating it alongside semantic bindings. All tests pass (one recompute test still ignored).
