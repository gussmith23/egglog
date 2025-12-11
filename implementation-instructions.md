Here’s a concrete plan, step-by-step, for grafting the paper’s *semantic e-graph* machinery onto egglog, focusing on:

* swapping opaque e-class IDs for **semantic IDs**, and
* adding **decomposers** (δ) that fire extra merges on semantic equality.
* testing along the way.

Please ensure you are testing at each step. Read and implement the Testing section in parallel with the Implementation section.

# Implementation

## 0. What the paper wants us to implement

The semantic e-graph in the paper does three key things:

1. Each e-class carries a **semantic identifier X ∈ D** (sei d), from some domain D_g indexed by its type g, instead of an opaque integer id.
2. **Semantic constructors** `»f… : D_{g1} × … × D_{gn} → D_g` compute parent semantic IDs from child semantic IDs on insertion (`sadd`). They may:

   * evaluate concretely (e.g., `1+4` and `2+3` both map to `5`), or
   * fall back to fresh symbolic values when not interpretable.
3. **Semantic decomposers** `δ_{g1,g2}: D_{g1}×D_{g2} → P(⋃_g' D_{g'}×D_{g'})` take two semantic IDs we’re equating and spit out more pairs of semantic IDs that must also be equal (e.g., elementwise vector alignment, complex numbers → pair of real/imag parts, etc.).

Operationally, we have two mutating operations over the e-graph:

* `sadd(e = f(C1,…,Cn))`: compute semantic ID from children, reuse or create an e-class for that semantic ID.
* `smerge(C1≃C2)`: merge their node sets, then:

  * primitive case: join semantic IDs via some `union(X1,X2)`;
  * structured case: use decomposer to push merges down, then propagate back up.

Semantic consistency invariant: **one-to-one mapping between e-classes and semantic IDs**.

We want that in egglog’s engine.

---

## 1. Where it fits into egglog today

From the Rust docs:

* `EGraph` is the main engine: holds sorts, function tables, rules, etc.([egraphs-good.github.io][1])
* Equality is handled via `EqSort` (implements `Sort`), representing classical e-classes; the engine maintains congruence closure internally.([egraphs-good.github.io][2])
* Values are represented by `Value`, and sorts are `ArcSort`.
* From the engine’s perspective, an “e-class id” for an `EqSort` is just a `Value` in that sort; union-find and hash-consing are internal and not part of the public API.

So: egglog already has *eq sorts* and e-class IDs, but they are **opaque numerical-ish identifiers**. We want to *layer semantic IDs* on top of that.

Rather than literally replacing all internal IDs with semantic IDs (which would be invasive), I’d:

* keep the existing union-find / numeric ids,
* but enforce that **each eq-class has exactly one semantic ID** and there is a **bijection between semantic IDs and eq-classes**, as in the paper.
* maintain a *semantic side-table* and hook it into e-class creation and merging.

---

## 2. Data structures to add

### 2.1 Representing semantic IDs

To avoid fancy generic gymnastics, I’d *reuse egglog’s existing `Value` representation* and its sorts:

* For each `EqSort` `S`, choose a **semantic domain sort** `D_S : ArcSort` (e.g., `I64Sort`, `VecSort`, tuples via container sorts, etc.). This lines up nicely with the paper’s `D = {D_g}` family of domains.

Then define:

```rust
/// Registered semantic info for a particular EqSort.
pub struct SemanticSortConfig {
    /// The eq sort whose e-classes we’re decorating.
    pub eq_sort: Arc<EqSort>,
    /// The sort of its semantic identifiers D_g.
    pub sem_sort: ArcSort,

    /// Constructor: computes semantic ID for a parent from child semantic IDs.
    pub constructor: Arc<dyn SemanticConstructor>,

    /// Optional per-(g1,g2) decomposers for this semantic domain.
    pub decomposers: Vec<SemanticDecomposerEntry>,
}

pub struct SemanticDecomposerEntry {
    pub lhs_sem_sort: ArcSort,
    pub rhs_sem_sort: ArcSort,
    pub func: Arc<dyn SemanticDecomposer>,
}
```

Where `SemanticConstructor` and `SemanticDecomposer` are user-implementable traits:

```rust
pub trait SemanticConstructor: Send + Sync {
    /// Given the semantic IDs of the children, compute the parent semantic ID.
    ///
    /// - `fun` is the egglog function representing the constructor application.
    /// - `children` are pairs (sort, semantic_value) for each child.
    ///
    /// Returns:
    ///   - `sem_value`: the semantic ID for the result.
    ///   - `extra_sem_classes`: optional additional semantic IDs that should
    ///      become singleton e-classes (fallback structured case).
    fn construct(
        &self,
        eg: &mut EGraph,
        fun: &ResolvedFunction,
        children: &[(ArcSort, Value)],
    ) -> (Value /*sem_value*/, Vec<(ArcSort, Value)> /*extra_sem_classes*/);
}

/// `δ_{g1,g2}` from the paper: given two semantics we are equating, return
/// additional semantic pairs that must also be merged.
pub trait SemanticDecomposer: Send + Sync {
    /// Both semantic IDs are values of the domain sorts registered in `SemanticDecomposerEntry`.
    ///
    /// Returns list of pairs (sort, lhs_sem, rhs_sem) that must be unified.
    fn decompose(
        &self,
        eg: &EGraph,
        lhs_sem: Value,
        rhs_sem: Value,
    ) -> Vec<(ArcSort /*child_sem_sort*/, Value, Value)>;
}
```

These traits are intentionally pretty “raw”: they operate directly on `Value` and `ArcSort` so they can handle polymorphic domains (tuples, vectors, etc.) and use the rest of egglog’s infrastructure (container values, base values).

### 2.2 Per-e-class semantic state

Inside the engine (not exposed in public API), add:

```rust
/// For eq sort S:
struct SemanticStore {
    sem_sort: ArcSort,

    /// Map eq-class id -> semantic ID (Value in sem_sort).
    class_sem: HashMap<Value, Value>,

    /// Map semantic ID -> eq-class id, enforcing 1–1 mapping.
    sem_class: HashMap<Value, Value>,
}
```

And in `EGraph`:

```rust
pub struct EGraph {
    // existing fields...
    semantic_sorts: HashMap<String /*EqSort name*/, SemanticSortConfig>,
    semantic_stores: HashMap<String /*EqSort name*/, SemanticStore>,
}
```

Invariants to maintain (matching the paper’s semantic consistency invariant):

* Every e-class `c` of a semantic-enabled `EqSort` has exactly one semantics `X ∈ D`.
* The mapping is bijective: `c ↔ X`.

If we need multiple semantic views for the same eq sort (multi-join semantics, cf. “multi-semantic identifiers” in the paper), we can extend `SemanticStore` to hold a *set* of semantic IDs per class, but I’d ignore that initially (they say they don’t need it for their main system).

---

## 3. Rust API for users

Goal: user configures semantics from Rust; DSL just benefits from it.

### 3.1 Registering a semantic domain for an eq sort

Add methods to `EGraph`:

```rust
impl EGraph {
    /// Attach a semantic domain to an EqSort.
    pub fn add_semantic_domain(
        &mut self,
        eq_sort: Arc<EqSort>,
        sem_sort: ArcSort,
        constructor: Arc<dyn SemanticConstructor>,
    ) {
        // insert into semantic_sorts, create a SemanticStore with empty maps.
    }

    /// Add a decomposer δ_{g1,g2} for this semantic domain.
    pub fn add_semantic_decomposer(
        &mut self,
        eq_sort_name: &str,
        lhs_sem_sort: ArcSort,
        rhs_sem_sort: ArcSort,
        decomposer: Arc<dyn SemanticDecomposer>,
    ) {
        // push into the corresponding SemanticSortConfig.decomposers
    }
}
```

Usage pattern from a Rust client:

```rust
let mut eg = EGraph::default();
let int_eq = eg.get_arcsort_by(|s| s.name() == "IntEq").downcast::<EqSort>().unwrap();
let int_sem = eg.get_sort::<I64Sort>(); // semantic domain D_Int = i64

eg.add_semantic_domain(
    int_eq.clone(),
    int_sem.clone(),
    Arc::new(MyIntSemanticConstructor::new()),
);

// If you had a structured vector semantics:
let vec_sem = eg.get_sort::<VecSort>(); // domain of D_Vec
eg.add_semantic_decomposer(
    "VecEq",
    vec_sem.clone(),
    vec_sem.clone(),
    Arc::new(MyVecDecomposer::new()),
);
```

Concrete implementor example (arithmetic from Example 3.1 in the paper):

```rust
struct IntArithConstructor;
impl SemanticConstructor for IntArithConstructor {
    fn construct(
        &self,
        eg: &mut EGraph,
        fun: &ResolvedFunction,
        children: &[(ArcSort, Value)],
    ) -> (Value, Vec<(ArcSort, Value)>) {
        // assume all children have sem_sort = I64Sort
        let sem_sort = eg.get_sort::<I64Sort>();
        let ints: Vec<i64> = children
            .iter()
            .map(|(_, v)| eg.value_to_base::<i64>(*v))
            .collect();

        let res = match fun.name.as_str() {
            "+" => ints[0] + ints[1],
            "*" => ints[0] * ints[1],
            _   => {
                // Fallback: treat as uninterpreted, use a hash-based symbolic value
                let sym = hash_symbolic(fun, &ints);
                return (eg.base_to_value(sym), vec![]);
            }
        };

        (eg.base_to_value(res), vec![])
    }
}
```

For decomposers (vector elementwise equality as in Example 3.4):

```rust
struct VecEqDecomposer;
impl SemanticDecomposer for VecEqDecomposer {
    fn decompose(
        &self,
        eg: &EGraph,
        lhs_sem: Value,
        rhs_sem: Value,
    ) -> Vec<(ArcSort, Value, Value)> {
        let vec_sort = eg.get_sort::<VecSort>();
        let lhs = eg.value_to_container::<VecContainer>(lhs_sem);
        let rhs = eg.value_to_container::<VecContainer>(rhs_sem);
        assert_eq!(lhs.len(), rhs.len());

        lhs.iter()
            .zip(rhs.iter())
            .map(|(lv, rv)| (vec_sort.inner_sorts()[0].clone(), *lv, *rv))
            .collect()
    }
}
```

This matches the paper’s δ for fixed-length vectors.

---

## 4. Hooking semantics into egglog’s core

Now the meat: we need to modify e-graph **insertion** and **merging** to use the semantic information.

I’ll mimic the paper’s `sadd` and `smerge` operations.

### 4.1 Insertion: semantic constructors (`sadd`)

Right now, when egglog inserts a new “constructor call” that produces an eq sort, it:

1. Creates a new e-node (function + child e-class IDs).
2. Does congruence-closure: if an identical e-node already exists, reuse its e-class; otherwise create a new e-class.

We want to extend that with semantics:

**New insertion path for eq sorts with semantics:**

Let `f : g1×…×gn → g` be a constructor / function which *returns* an eq sort `S_g` with registered semantic domain `D_S`.

Given child e-classes `C1,…,Cn` with semantic IDs `X1,…,Xn`:

1. Look up `X_i = semantic_stores[S_i].class_sem[Ci]` using the store for the child eq sort (or treat primitive/other sorts as providing direct semantics via `Value`).
2. Call the registered `SemanticConstructor`:

   ```rust
   let (sem_id, extra_sem_classes) = constructor.construct(
       &mut eg,
       &fun,
       &child_sem_pairs,
   );
   ```
3. If `sem_id` is already present in `semantic_stores[S_g].sem_class`, take the existing eq-class `C_existing` and **merge** the syntactic e-node into that class (classical congruence closure).
   This is the “immediate semantic merge” from the paper: equal semantics *skip* explicit rewrites.
4. Otherwise, create a fresh e-class `C_new` for the e-node, and record:

   ```rust
   semantic_stores[S_g].class_sem.insert(C_new, sem_id);
   semantic_stores[S_g].sem_class.insert(sem_id, C_new);
   ```
5. For any `extra_sem_classes` returned by the constructor (fallback structured case – e.g. symbolic vectors or tuples), create new singleton e-classes in the appropriate eq sorts and associate them with those semantic IDs; this mirrors the paper’s “structured fallback” case.

Crucially, this is done **in addition to** existing hashcons-based e-node lookup. The semantics can unify classes even when syntactic hashes differ, which is exactly what they want.

Implementation-wise, the hook lives where the engine calls something like “insert function row / constructor result value for an eq sort”; we intercept that path and route it through the semantic logic when a `SemanticSortConfig` exists for the output sort.

### 4.2 Merging: semantic decomposers (`smerge`)

Whenever the engine decides to union two e-classes `C1, C2` of the same `EqSort` `S`, we want to:

* merge them as usual, and
* run semantic merge + decomposer logic.

In the paper, `smerge(2₁≃X1, 2₂≃X2)` does:

1. **Merge** node sets: `2* = 2₁ ∪ 2₂`.
2. If primitive domain:

   * compute `X* = union(X1,X2)` (some least-upper-bound or tag/replace structure), and set semantics of class to `X*`.
3. If structured:

   * apply decomposer `δ(X1,X2)`, which returns `[(Y1,Y1'), …]` in some (possibly different) domains,
   * recursively merge the classes whose semantic IDs are `Y1,Y1'`, etc.,
   * recompute parent semantics upward.

To do this in egglog:

**Extend the core union operation for eq sorts with semantics:**

Assume we’re merging `class_a` and `class_b` for eq sort `S` with domain sort `D_S`.

1. Get current semantic IDs:

   ```rust
   let store = &mut semantic_stores[S];
   let sem_a = store.class_sem[&class_a];
   let sem_b = store.class_sem[&class_b];
   ```

2. Use union-find to pick a representative class `class_rep` (existing code).

3. Decide new semantic ID:

   * If we treat all domains as “primitive” by default, we can call a domain-specific `union_semantics(sem_a, sem_b)` that is part of the constructor’s implementation (e.g., “take lattice join”, “pick one and tag”, etc.). This function is equivalent to the paper’s `union(X₁,X₂)` for primitive domains.
   * For structured domains, instead of computing a direct `X*`, we will let decomposers handle downward propagation and then later recompute semantics by re-running constructors for parent nodes.

4. Update the store:

   ```rust
   // remove old entries
   store.sem_class.remove(&sem_a);
   if sem_b != sem_a { store.sem_class.remove(&sem_b); }

   // insert new mapping
   store.class_sem.insert(class_rep, new_sem);
   store.sem_class.insert(new_sem, class_rep);
   ```

5. **Decomposer step**:

   For each registered `SemanticDecomposerEntry` for this domain where the semantic sorts match the runtime sorts of `sem_a` and `sem_b`, call:

   ```rust
   let pairs = entry.func.decompose(&eg, sem_a, sem_b);
   for (child_sem_sort, child_sem1, child_sem2) in pairs {
       // Find or create the child eq-classes that have those semantic IDs.
       let child_eq_sort = semantic_sorts
           .values()
           .find(|cfg| cfg.sem_sort == child_sem_sort)
           .expect("no eq sort for given semantic domain");

       let child_store = &mut semantic_stores[child_eq_sort.eq_sort.name()];
       let class1 = child_store.sem_class[&child_sem1];
       let class2 = child_store.sem_class[&child_sem2];

       self.merge_eclasses(child_eq_sort.eq_sort.clone(), class1, class2);
   }
   ```

   This recursive `merge_eclasses` call re-enters this same semantic merge routine and classical union-find; termination follows from the same monotone equivalence relation argument as in the paper.

6. **Upward recomputation**:

   When semantic IDs of child classes change (due to merging), we may need to recompute parent semantic IDs and possibly trigger further merges (the “upward propagation” in the complex-number example).

   I’d implement this lazily:

   * Maintain, per eq sort, a reverse index from child class → parent function rows (egglog already has to know enough about functional dependencies to support rebuilders; we can reuse that).
   * When a class’s semantic ID changes, enqueue its parents for semantic recomputation:

     ```rust
     for parent in parents_of(class_rep) {
         let old_sem = store.class_sem[&parent_class];
         let new_sem = constructor.construct(...child_sem_ids...).0;
         if new_sem != old_sem {
             // update store, and if another class already has new_sem, merge them
         }
     }
     ```
   * This is analogous to existing e-analysis / rebuild passes in egglog, but driven by semantic IDs.

This gives us `smerge` semantics more or less as in the paper without ripping out egglog’s internal representation.

---

## 5. How a Rust user would *actually* plug in a new semantic domain

Putting it together, a user wanting to use semantic IDs for some eq sort `Expr` could:

1. **Define the semantic domain** as an egglog sort, e.g.:

   * `I64Sort` for integer semantics,
   * `VecSort<I64Sort>` for vector semantics,
   * or a custom container/base sort they implemented via `BaseValue` / `ContainerValue`.

2. **Register it with the engine**:

   ```rust
   let mut eg = EGraph::default();

   let expr_eq: Arc<EqSort> = eg.get_arcsort_by(|s| s.name() == "Expr").downcast().unwrap();
   let sem_sort: ArcSort = eg.get_sort::<I64Sort>(); // D_Expr = Int semantics

   eg.add_semantic_domain(
       expr_eq.clone(),
       sem_sort.clone(),
       Arc::new(ExprArithConstructor::new()),
   );
   ```

3. Optionally **register decomposers** for structured semantics:

   ```rust
   let vec_sem = eg.get_sort::<VecSort>();
   eg.add_semantic_decomposer(
       "Expr",     // eq sort name
       vec_sem.clone(),
       vec_sem.clone(),
       Arc::new(VecEqDecomposer),
   );
   ```

After that, **no DSL changes are strictly required** for the semantics to take effect: any egglog program that creates `Expr` nodes will use the semantic constructors / decomposers implicitly.

---

## 6. How to expose semantic IDs in the egglog DSL

You said it’s fine if *defining* semantic IDs stays in Rust, but you want ideas on how to *use* them from the DSL.

I’d do the following **lightweight** surface extensions (all implemented on top of the machinery above):

### 6.1 Auto-generated semantic functions

For each eq sort `S` with semantic domain `D_S`, synthesize an internal function:

```egglog
(sort Sem_S)        ; optional, if D_S is not a built-in sort
(function sem_S S D_S)
```

And maintain a **functional dependency** `sem_S(S) -> D_S` in the engine, so that:

* whenever two `S`-classes merge, `sem_S` stays consistent, and
* rule matching on `sem_S` is safe.

Then in DSL, users can write rules that use semantics:

```egglog
; Example: constant folding guarded by semantic information
(rule (=> ((Expr x) (sem_Expr x v)
          (v > 0))             ; semantic test
         (PosExpr x)))         ; some derived predicate
```

Here `sem_Expr` is read-only: no DSL construct should allow the user to assign to it; it’s entirely backed by the Rust semantic domain.

### 6.2 Semantic equality / constraints

Even though semantic equality is implicitly enforced (two nodes with equal sem IDs are merged), sometimes it’s useful to *ask* about semantics:

* Add a DSL primitive `sem_eq_S(x, y)` that just desugars to `sem_S(x) == sem_S(y)` over `D_S`.
* Allow conditional guards using semantic values (e.g. range checks, alignment properties) by exposing the semantic sort `D_S` and its primitives in the usual way (Rust primitives / `add_primitive!`).

This lets users write high-level rules that depend on semantics, which is exactly what the paper’s motivating examples do, but without having to model semantics as ordinary egglog tables by hand.

### 6.3 Decomposer-driven rules (optional)

In most cases, decomposers will act entirely under the hood, but if you want to give DSL users visibility:

* Add debug commands like `print_sem_class` that dump the semantic ID plus the induced decomposed pairs.
* Maybe expose a relation `Decomp_S(x, y, child_x, child_y)` for debugging; but I’d start without any DSL exposure here and keep decomposers purely engine-level.

---

## 7. Summary of the integration plan

Very concretely:

1. **Add semantic metadata to `EGraph`:**

   * `SemanticSortConfig` per semantic-enabled `EqSort`.
   * `SemanticStore` with `class_sem` and `sem_class` maps.

2. **Define Rust traits for user-provided semantics:**

   * `SemanticConstructor` implements `»f…` for the domain.
   * `SemanticDecomposer` implements `δ_{g1,g2}`.

3. **Extend `EGraph` API:**

   * `add_semantic_domain(eq_sort, sem_sort, constructor)`.
   * `add_semantic_decomposer(eq_sort_name, lhs_sem_sort, rhs_sem_sort, decomposer)`.

4. **Hook into core operations:**

   * On insertion of a node whose output sort is a semantic-enabled `EqSort`, compute semantic ID via `SemanticConstructor`, look up existing class by semantic ID, and merge or create accordingly (`sadd`).
   * On union of two e-classes of a semantic-enabled `EqSort`, update semantic store, run appropriate decomposers, recursively merge implied child classes, and recompute parent semantics (`smerge` + downward/upward propagation).

5. **Optional DSL exposure:**

   * Auto-generate read-only functions `sem_S : S → D_S`.
   * Allow rules to pattern-match or guard on semantic values.
   * Maybe add small helper primitives like `sem_eq_S`.

This respects the paper’s semantics pretty closely, without demanding a full rearchitecture of egglog’s internals, and gives you a clean Rust interface to experiment with different semantic domains and decomposers.

[1]: https://egraphs-good.github.io/egglog/docs/egglog/struct.EGraph.html "EGraph in egglog - Rust"
[2]: https://egraphs-good.github.io/egglog/docs/egglog/sort/index.html "egglog::sort - Rust"


# Testing

## 0. Groundwork: protect existing behavior

Before touching semantics:

1. **Snapshot current behavior** for a representative egglog program (or several) that uses plain `EqSort`:

   * Normalization of simple arithmetic.
   * Equality saturation example(s) already in the repo.
   * DSL regression tests (if they exist).

2. Add smoke tests like:

```rust
#[test]
fn existing_equivalence_examples_still_hold() {
    // Build a small egglog program with the public API.
    // Run saturation / evaluation.
    // Assert final equalities / tables have the expected shape.
}
```

These give you a baseline that must continue to pass once semantics are wired in.

---

## 1. Semantic store + registration (no engine hooks yet)

### Goal

Confirm that the semantic metadata and storage behave correctly on their own.

### Tests to write **first**

#### 1.1 Basic registration

```rust
#[test]
fn can_register_semantic_domain_for_eq_sort() {
    // Arrange
    let mut eg = EGraph::default();
    let eq_sort = eg.add_eq_sort("Expr");
    let sem_sort = eg.add_sort::<I64Sort>("IntSem");

    // Act
    eg.add_semantic_domain(eq_sort.clone(), sem_sort.clone(), Arc::new(DummyConstructor));

    // Assert
    assert!(eg.semantic_sorts.contains_key("Expr"));
    let cfg = &eg.semantic_sorts["Expr"];
    assert!(Arc::ptr_eq(&cfg.eq_sort, &eq_sort));
    assert!(Arc::ptr_eq(&cfg.sem_sort, &sem_sort));
}
```

#### 1.2 Decomposer registration

```rust
#[test]
fn can_register_decomposer_for_semantic_domain() {
    let mut eg = EGraph::default();
    let expr_eq = eg.add_eq_sort("Expr");
    let sem_sort = eg.add_sort::<VecSort>("VecSem");

    eg.add_semantic_domain(expr_eq.clone(), sem_sort.clone(), Arc::new(DummyConstructor));

    eg.add_semantic_decomposer(
        "Expr",
        sem_sort.clone(),
        sem_sort.clone(),
        Arc::new(DummyDecomposer),
    );

    let cfg = &eg.semantic_sorts["Expr"];
    assert_eq!(cfg.decomposers.len(), 1);
    assert!(Arc::ptr_eq(&cfg.decomposers[0].lhs_sem_sort, &sem_sort));
}
```

#### 1.3 Store invariant: bijection

Write tests purely against `SemanticStore`:

```rust
#[test]
fn semantic_store_enforces_bijection() {
    let mut store = SemanticStore::new(sem_sort.clone());
    let class1 = Value::from_u32(1);
    let class2 = Value::from_u32(2);
    let x = Value::from_i64(5);

    store.bind(class1, x);
    assert_eq!(store.class_sem[&class1], x);
    assert_eq!(store.sem_class[&x], class1);

    // Attempting to bind the same semantic ID to another class should reject/fail.
    let res = store.try_bind(class2, x);
    assert!(res.is_err());
}
```

> Only after these tests compile and fail (because the code doesn’t exist yet) do you implement `SemanticStore`, `add_semantic_domain`, `add_semantic_decomposer`, etc.

---

## 2. Constructors / insertion path (`sadd`)

Now test that **insertion** of nodes for a semantic-enabled eq sort uses semantic IDs to choose classes.

### 2.1 Unit tests for a simple `SemanticConstructor`

Start with a trivial semantic domain: integers, mapping `(+ 2 3)` → `5`.

```rust
#[test]
fn int_arith_constructor_evaluates_plus_and_times() {
    let eg = EGraph::default();
    let ctor = IntArithConstructor::new();  // your test impl

    let plus_fun = fun("+");
    let children = vec![(int_sem_sort.clone(), int_val(2)), (int_sem_sort.clone(), int_val(3))];

    let (sem_id, extra) = ctor.construct(&mut eg.clone(), &plus_fun, &children);
    assert_eq!(sem_id, int_val(5));
    assert!(extra.is_empty());
}
```

Also write a test for **uninterpreted fallback**:

```rust
#[test]
fn int_arith_constructor_falls_back_for_unknown_fun() {
    let eg = EGraph::default();
    let ctor = IntArithConstructor::new();

    let weird_fun = fun("foo");
    let children = vec![(int_sem_sort.clone(), int_val(2))];
    let (sem_id1, _) = ctor.construct(&mut eg.clone(), &weird_fun, &children);
    let (sem_id2, _) = ctor.construct(&mut eg.clone(), &weird_fun, &children);

    // Should be deterministic but non-equal to any “normal” value.
    assert_eq!(sem_id1, sem_id2);
    assert_ne!(sem_id1, int_val(2));
}
```

### 2.2 Integration test: `sadd` merges equal semantics

Write tests that go through the actual `EGraph` insertion API, not just the constructor:

```rust
#[test]
fn insertion_merges_nodes_with_equal_semantics() {
    let mut eg = EGraph::default();

    let expr_eq = eg.add_eq_sort("Expr");
    let int_sem = eg.add_sort::<I64Sort>("IntSem");

    eg.add_semantic_domain(expr_eq.clone(), int_sem.clone(), Arc::new(IntArithConstructor::new()));

    // Insert "(+ 2 3)" and "(+ 1 4)" as Expr nodes.

    let c1 = eg.insert_expr_plus(int_val(2), int_val(3)); // some helper building the e-node
    let c2 = eg.insert_expr_plus(int_val(1), int_val(4));

    // Assert: they end up in the same e-class because semantics are both 5.
    assert_eq!(eg.find_class(c1), eg.find_class(c2));

    // Assert: semantic ID for the class is exactly 5.
    let store = &eg.semantic_stores["Expr"];
    let rep = eg.find_class(c1);
    assert_eq!(store.class_sem[&rep], int_val(5));
}
```

Also test that **different semantics → different classes**:

```rust
#[test]
fn insertion_does_not_merge_nodes_with_different_semantics() {
    // (+ 2 3) vs (+ 2 4) → 5 vs 6
}
```

And a test where **syntactic equality implies both**:

```rust
#[test]
fn semantics_respect_congruence() {
    // Insert the _same_ node twice; check that both hash-consing and semantics agree.
}
```

The code for `insert_expr_plus` can live in a test helper module, calling the internal function-building API.

---

## 3. Merge logic + decomposers (`smerge`)

Once inserts work, move to **merging** and **decomposition**.

### 3.1 Primitive semantic merge

For a primitive domain (like integers) define some `union_semantics` behavior (e.g. assert that merging different ints is illegal, or pick first, or assert equal).

Tests:

```rust
#[test]
fn merge_same_semantic_id_is_noop() {
    // Build two syntactically distinct Expr nodes that share the same semantic ID (5).
    // Merge their classes explicitly.
    // Assert:
    // - union-find representative is consistent,
    // - semantic_mappings still map to 5,
    // - the bijection invariant holds.
}
```

```rust
#[test]
fn merge_different_semantics_panics_or_handles_consistently() {
    // Build 2 and 3 as separate EqSort classes with semantics 2 and 3,
    // then force a merge at engine level.
    // Assert that:
    //   - Either: test expects a panic (if you choose to forbid it),
    //   - Or: union_semantics picks one and both classes now share that semantic.
}
```

These tests force you to clearly define behavior for the “merge inconsistent semantics” case.

### 3.2 Decomposer for structured domain: simple pair

Before going to vectors, test with a simple structured semantics: pairs `(a, b)`.

* Domain `D_Pair` is something like `(I64, I64)`.
* Decomposer: if `(a, b)` = `(c, d)`, then enforce `a = c` and `b = d` in the underlying domains.

Unit test for decomposer:

```rust
#[test]
fn pair_decomposer_returns_component_pairs() {
    let eg = EGraph::default();
    let dec = PairDecomposer::new();
    let pair_sort = eg.add_sort::<PairSort>("PairSem");

    let lhs = pair_val(1, 2);
    let rhs = pair_val(1, 2);
    let pairs = dec.decompose(&eg, lhs, rhs);

    assert_eq!(pairs.len(), 2);
    // Check that we get (1,1) and (2,2) in the right child semantic sort.
}
```

Integration test for **recursive merge**:

```rust
#[test]
fn decomposer_triggers_child_merges() {
    let mut eg = EGraph::default();

    // eq sort S for complex numbers; semantic domain D = Pair(Int, Int).
    let complex_eq = eg.add_eq_sort("Complex");
    let pair_sem = eg.add_sort::<PairSort>("PairSem");
    let int_sem = eg.add_sort::<I64Sort>("IntSem");

    eg.add_semantic_domain(complex_eq.clone(), pair_sem.clone(), Arc::new(ComplexConstructor::new()));
    eg.add_semantic_domain(/* eq sort for ReIm */, int_sem.clone(), Arc::new(IntConstructor::new()));
    eg.add_semantic_decomposer(
        "Complex",
        pair_sem.clone(),
        pair_sem.clone(),
        Arc::new(PairDecomposer::new()),
    );

    // Build two complex nodes z1, z2 with semantics (3, 4) and (3, 4),
    // but whose real/imag child classes are different.
    let z1 = eg.insert_complex(real1, imag1);
    let z2 = eg.insert_complex(real2, imag2);

    // Merge z1 and z2 at the complex level:
    eg.merge_eclasses(complex_eq.clone(), z1, z2);

    // Now assert:
    // - real1 and real2 classes merged,
    // - imag1 and imag2 classes merged,
    // i.e. the decomposer was invoked and propagated merges downwards.
}
```

You’ll have to build the helper methods `insert_complex`, etc., in the test harness.

### 3.3 Upward recomputation sanity

Write at least one integration test where:

1. You merge children (or their semantics) first.
2. That should change the semantic ID of a parent node.
3. The engine recomputes and then merges parents that now have equal semantics.

Example:

```rust
#[test]
fn child_merge_can_trigger_parent_semantic_merge() {
    // eq sort Expr with int semantic domain.

    // Build parents:
    let c1 = insert_plus(expr_for(1), expr_for(2)); // sem = 3
    let c2 = insert_plus(expr_for(3), expr_for(0)); // sem = 3

    // Suppose we start with semantics incomplete so that initially sem(c1) != sem(c2).
    // Then some child merge happens (e.g. 1+2 rewritten to 3, etc.) so that their semantics become equal.
    // After running the reconstruction / semantic recomputation:
    eg.rebuild_semantics();

    assert_eq!(eg.find_class(c1), eg.find_class(c2));
}
```

This forces you to implement some `rebuild_semantics` / incremental recomputation and hook it into existing rebuild passes.

---

## 4. DSL-level tests (using semantics via `sem_S`)

Once the engine hooks are solid, test **exposed DSL features** that use semantics.

### 4.1 Generated semantic function is consistent

Suppose you synthesize a function `sem_Expr : Expr → IntSem`:

```rust
#[test]
fn sem_function_reflects_internal_semantics() {
    // Build an egglog program via string input or the public DSL builder:
    //
    // (sort Expr)
    // (function plus Expr Expr Expr)
    // (function sem_Expr Expr IntSem)  ; auto-generated
    // ...
    //
    // Insert (+ 2 3) and query sem_Expr of that class.

    let mut eg = EGraph::default();
    load_program_with_sem_expr(&mut eg);

    let expr = build_plus_2_3(&mut eg);
    let sem_value = eg.eval_sem_expr(expr);

    assert_eq!(sem_value, int_val(5));

    // Also check that asking for sem_Expr on an equivalent representative gives the same result.
}
```

### 4.2 Rule guards using semantics

Add a test that uses a DSL rule that pattern-matches on the semantic value:

```rust
#[test]
fn rules_can_guard_on_semantics() {
    // DSL pseudo-code:
    //
    // (rule (=> ((Expr x) (sem_Expr x v) (> v 0))
    //           (Positive x)))
    //
    // Then:
    //   - build Expr node whose semantics is +1,
    //   - run saturation,
    //   - assert Positive(x) holds.

    let mut eg = EGraph::default();
    load_program_with_positive_rule(&mut eg);

    let x = build_expr_with_semantic(1, &mut eg);
    eg.run();

    assert!(eg.has_fact("Positive", &[x]));
}
```

This confirms that:

* `sem_Expr` is visible in the DSL,
* the semantic value is flowing correctly into guard predicates, and
* semantics and equality work together.

---

## 5. Property tests / fuzzing

After you’ve got deterministic unit/integration tests, add some **property-based tests** using something like `proptest` or `quickcheck`.

Examples:

### 5.1 Semantic functional correctness

For a simple arithmetic semantics:

* Randomly generate small expressions (trees) with operations you interpret (`+`, `*`) and leaves from a bounded integer range.
* Evaluate them two ways:

  1. via the semantic domain and constructors, and
  2. via a reference evaluator you write in the test.

Property:

```text
For all randomly generated Expr t,
  sem_Expr(t) == eval_reference(t)
```

If semantics also trigger merges, you can add:

```text
If sem_Expr(t1) == sem_Expr(t2), then find_class(t1) == find_class(t2)
```

### 5.2 Decomposer soundness

For structured domains like vectors:

* Randomly generate “vector” semantics and their constituent child semantics.
* When you decompose them and merge children, assert that:

  * the number and sorts of generated child pairs match expectations, and
  * recompute the parent semantics to ensure they didn’t become inconsistent.

---

## 6. Regression tests and performance checks

Once things are working:

### 6.1 Regression tests for bugs

Every time you hit a bug during dev (e.g., inconsistent semantic mappings, infinite recursion in decomposer, etc.), immediately:

1. Reduce it to a minimal egglog program or Rust-level scenario.
2. Add a regression test to `tests/semantic_regressions.rs`:

```rust
#[test]
fn regression_looping_decomposer_case_2025_11_29() {
    // Build the minimal case that used to loop.
    // Assert that it now terminates and the final e-graph shape is reasonable.
}
```

### 6.2 Performance sanity

You don’t need full benchmark infra immediately, but do add at least:

* A test with a moderately large expression DAG relying on semantics for deduplication, asserting it finishes within some reasonable time/size bounds.
* A test where decomposers generate many merges but still terminates within time.

Rust’s `#[ignore]` with manual running is fine for heavier perf checks, but still write them early.

---

## 7. Development order (TDD workflow)

To keep things realistic and manageable:

1. **Phase 0: Baseline**

   * Write tests covering current engine behavior.
   * Run them; they all pass.
2. **Phase 1: Registration & store**

   * Write tests in section 1.
   * They fail (not implemented).
   * Implement `SemanticStore`, `add_semantic_domain`, `add_semantic_decomposer` until green.
3. **Phase 2: Constructors & insertion**

   * Write constructor unit tests and `sadd` tests.
   * Implement constructor trait and hook into insertion path, updating semantic store.
4. **Phase 3: Merge + decomposers**

   * Write primitive merge tests, pair/vector decomposer tests, upward recomputation tests.
   * Implement merge semantics, decomposer handling, and semantic rebuild.
5. **Phase 4: DSL exposure**

   * Write tests that rely on `sem_S` in DSL rules.
   * Implement auto-generation of `sem_S` functions and wiring to engine semantics.
6. **Phase 5: Properties & perf**

   * Add proptests / fuzz, perf sanity tests.
   * Use them as safety net while refactoring / optimizing.

At each phase, don’t move on until:

* New tests pass.
* Old tests (baseline + previous phases) still pass.

That gives you a tight, test-driven loop and pretty high confidence that semantic identifiers + decomposers + DSL integration behave the way the paper intends and don’t quietly break existing egglog behavior.
