use std::sync::Arc;

use thiserror::Error;

use crate::{
    ArcSort, EGraph, FunctionSubtype, ResolvedFunction, ResolvedFunctionDecl, Schema, Span, Value,
    sort::EqSort, util::HashMap,
};

/// User hook that computes semantic IDs for constructor applications.
///
/// This mirrors the paper's `»f…` operator.
pub trait SemanticConstructor: Send + Sync {
    /// Given the semantic IDs of the children, compute the semantic ID for the parent.
    ///
    /// Returns:
    ///  - the semantic ID for the parent value
    ///  - any extra semantic IDs that should each form a singleton e-class
    fn construct(
        &self,
        egraph: &mut EGraph,
        fun: &ResolvedFunction,
        children: &[(ArcSort, Value)],
    ) -> (Value, Vec<(ArcSort, Value)>);
}

/// User hook that decomposes a semantic merge into additional merges.
///
/// This mirrors the paper's `δ_{g1,g2}`.
pub trait SemanticDecomposer: Send + Sync {
    /// Given two semantic IDs we are equating, return additional pairs that must be merged.
    fn decompose(
        &self,
        egraph: &EGraph,
        lhs_sem: Value,
        rhs_sem: Value,
    ) -> Vec<(ArcSort, Value, Value)>;
}

/// A registered semantic decomposer along with the semantic sorts it handles.
#[derive(Clone)]
pub struct SemanticDecomposerEntry {
    pub lhs_sem_sort: ArcSort,
    pub rhs_sem_sort: ArcSort,
    pub func: Arc<dyn SemanticDecomposer>,
}

/// Registered semantic configuration for a particular `EqSort`.
#[derive(Clone)]
pub struct SemanticSortConfig {
    pub eq_sort: Arc<EqSort>,
    pub sem_sort: ArcSort,
    pub constructor: Arc<dyn SemanticConstructor>,
    pub decomposers: Vec<SemanticDecomposerEntry>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SemanticRegistrationError {
    #[error("semantic domain already registered for eq sort {0}")]
    AlreadyRegistered(String),
    #[error("no semantic domain registered for eq sort {0}")]
    MissingEqSort(String),
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SemanticStoreError {
    #[error(
        "semantic id {sem_id:?} already bound to class {existing_class:?} (attempted {new_class:?})"
    )]
    SemanticAlreadyBound {
        sem_id: Value,
        existing_class: Value,
        new_class: Value,
    },
    #[error(
        "class {class:?} already bound to semantic id {existing_sem:?} (attempted {new_sem:?})"
    )]
    ClassAlreadyBound {
        class: Value,
        existing_sem: Value,
        new_sem: Value,
    },
}

/// Bidirectional mapping between e-class IDs and semantic IDs for a semantic-enabled eq sort.
#[derive(Clone, Debug)]
pub struct SemanticStore {
    sem_sort: ArcSort,
    class_sem: HashMap<Value, Value>,
    sem_class: HashMap<Value, Value>,
    /// Backend function id for the generated sem_<EqSort> function.
    sem_function: Option<egglog_bridge::FunctionId>,
}

impl SemanticStore {
    pub fn new(sem_sort: ArcSort) -> Self {
        Self {
            sem_sort,
            class_sem: HashMap::default(),
            sem_class: HashMap::default(),
            sem_function: None,
        }
    }

    pub fn sem_sort(&self) -> &ArcSort {
        &self.sem_sort
    }

    pub fn class_sem(&self, class: &Value) -> Option<Value> {
        self.class_sem.get(class).copied()
    }

    pub fn sem_class(&self, sem_id: &Value) -> Option<Value> {
        self.sem_class.get(sem_id).copied()
    }

    pub fn sem_function(&self) -> Option<egglog_bridge::FunctionId> {
        self.sem_function
    }

    /// Bind a class to a semantic ID, enforcing the bijection invariant.
    /// TODO(@codex): should this be doing both error checks every time, rather than returning early?
    pub fn bind(&mut self, class: Value, sem_id: Value) -> Result<(), SemanticStoreError> {
        if let Some(existing) = self.class_sem.get(&class) {
            if existing != &sem_id {
                return Err(SemanticStoreError::ClassAlreadyBound {
                    class,
                    existing_sem: *existing,
                    new_sem: sem_id,
                });
            }
            return Ok(());
        }

        if let Some(existing) = self.sem_class.get(&sem_id) {
            if existing != &class {
                return Err(SemanticStoreError::SemanticAlreadyBound {
                    sem_id,
                    existing_class: *existing,
                    new_class: class,
                });
            }
            return Ok(());
        }

        self.class_sem.insert(class, sem_id);
        self.sem_class.insert(sem_id, class);
        Ok(())
    }

    /// Convenience alias for `bind` to match the plan's terminology.
    /// TODO(@codex): remove this, we don't need to match the plan exactly.
    pub fn try_bind(&mut self, class: Value, sem_id: Value) -> Result<(), SemanticStoreError> {
        self.bind(class, sem_id)
    }

    pub fn len(&self) -> usize {
        self.class_sem.len()
    }

    /// Remove both directions of a binding for the given class, if present.
    pub fn unbind_class(&mut self, class: Value) {
        if let Some(sem) = self.class_sem.remove(&class) {
            self.sem_class.remove(&sem);
        }
    }

    /// Remove both directions of a binding for the given semantic id, if present.
    pub fn unbind_sem(&mut self, sem: Value) {
        if let Some(class) = self.sem_class.remove(&sem) {
            self.class_sem.remove(&class);
        }
    }
}

impl EGraph {
    /// Attach a semantic domain to an `EqSort`.
    pub fn add_semantic_domain(
        &mut self,
        eq_sort: Arc<EqSort>,
        sem_sort: ArcSort,
        constructor: Arc<dyn SemanticConstructor>,
    ) -> Result<(), SemanticRegistrationError> {
        let name = eq_sort.name.clone();
        if self.semantic_sorts.contains_key(&name) {
            return Err(SemanticRegistrationError::AlreadyRegistered(name));
        }

        // Register read-only semantic function sem_<EqSort>
        let sem_fn_name = format!("sem_{}", eq_sort.name);
        let sem_decl = ResolvedFunctionDecl {
            name: sem_fn_name.clone(),
            subtype: FunctionSubtype::Custom,
            schema: Schema {
                input: vec![eq_sort.name.clone()],
                output: sem_sort.name().to_owned(),
            },
            merge: None,
            cost: None,
            unextractable: true,
            let_binding: false,
            span: Span::Panic,
        };
        self.declare_function(&sem_decl)
            .map_err(|_| SemanticRegistrationError::AlreadyRegistered(sem_fn_name.clone()))?;
        let sem_func = self.functions.get(&sem_fn_name).unwrap().clone();
        let sem_backend_id = sem_func.backend_id;
        let mut store = SemanticStore::new(sem_sort.clone());
        store.sem_function = Some(sem_backend_id);
        self.semantic_functions.insert(name.clone(), sem_func);

        self.semantic_sorts.insert(
            name.clone(),
            SemanticSortConfig {
                eq_sort,
                sem_sort: sem_sort.clone(),
                constructor,
                decomposers: vec![],
            },
        );
        self.semantic_stores.insert(name, store);
        Ok(())
    }

    /// Add a decomposer for a semantic domain previously registered with [`add_semantic_domain`].
    pub fn add_semantic_decomposer(
        &mut self,
        eq_sort_name: &str,
        lhs_sem_sort: ArcSort,
        rhs_sem_sort: ArcSort,
        decomposer: Arc<dyn SemanticDecomposer>,
    ) -> Result<(), SemanticRegistrationError> {
        let cfg = self
            .semantic_sorts
            .get_mut(eq_sort_name)
            .ok_or_else(|| SemanticRegistrationError::MissingEqSort(eq_sort_name.to_owned()))?;
        cfg.decomposers.push(SemanticDecomposerEntry {
            lhs_sem_sort,
            rhs_sem_sort,
            func: decomposer,
        });
        Ok(())
    }

    pub fn semantic_config(&self, eq_sort_name: &str) -> Option<&SemanticSortConfig> {
        self.semantic_sorts.get(eq_sort_name)
    }

    pub fn semantic_store(&self, eq_sort_name: &str) -> Option<&SemanticStore> {
        self.semantic_stores.get(eq_sort_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArcSort, ColumnTy, Span, sort::ResolvedFunctionId};

    #[derive(Clone)]
    struct DummyConstructor;
    impl SemanticConstructor for DummyConstructor {
        fn construct(
            &self,
            _egraph: &mut EGraph,
            _fun: &ResolvedFunction,
            _children: &[(ArcSort, Value)],
        ) -> (Value, Vec<(ArcSort, Value)>) {
            (Value::new_const(0), vec![])
        }
    }

    #[derive(Clone)]
    struct DummyDecomposer;
    impl SemanticDecomposer for DummyDecomposer {
        fn decompose(
            &self,
            _egraph: &EGraph,
            _lhs_sem: Value,
            _rhs_sem: Value,
        ) -> Vec<(ArcSort, Value, Value)> {
            vec![]
        }
    }

    #[test]
    fn can_register_semantic_domain_for_eq_sort() {
        let mut eg = EGraph::default();
        eg.declare_sort("Expr", &None, Span::Panic).unwrap();
        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();

        eg.add_semantic_domain(
            eq_sort.clone(),
            sem_sort.clone(),
            Arc::new(DummyConstructor),
        )
        .unwrap();

        let cfg = eg.semantic_config("Expr").unwrap();
        assert_eq!(cfg.eq_sort.name, "Expr");
        assert_eq!(cfg.sem_sort.name(), sem_sort.name());
        assert!(eg.semantic_store("Expr").is_some());
    }

    #[test]
    fn can_register_decomposer_for_semantic_domain() {
        let mut eg = EGraph::default();
        eg.declare_sort("Expr", &None, Span::Panic).unwrap();
        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();

        eg.add_semantic_domain(eq_sort, sem_sort.clone(), Arc::new(DummyConstructor))
            .unwrap();
        eg.add_semantic_decomposer(
            "Expr",
            sem_sort.clone(),
            sem_sort.clone(),
            Arc::new(DummyDecomposer),
        )
        .unwrap();

        let cfg = eg.semantic_config("Expr").unwrap();
        assert_eq!(cfg.decomposers.len(), 1);
        assert_eq!(cfg.decomposers[0].lhs_sem_sort.name(), sem_sort.name());
        assert_eq!(cfg.decomposers[0].rhs_sem_sort.name(), sem_sort.name());
    }

    #[test]
    fn semantic_store_enforces_bijection() {
        let sem_sort = EGraph::default().get_sort_by_name("i64").unwrap().clone();
        let mut store = SemanticStore::new(sem_sort);
        let class1 = Value::new_const(1);
        let class2 = Value::new_const(2);
        let sem = Value::new_const(5);

        store.bind(class1, sem).unwrap();
        assert_eq!(store.class_sem(&class1), Some(sem));
        assert_eq!(store.sem_class(&sem), Some(class1));
        assert_eq!(store.len(), 1);

        let res = store.try_bind(class2, sem);
        assert!(matches!(
            res,
            Err(SemanticStoreError::SemanticAlreadyBound { .. })
        ));
    }

    #[derive(Clone)]
    struct IntArithConstructor;
    impl SemanticConstructor for IntArithConstructor {
        fn construct(
            &self,
            egraph: &mut EGraph,
            fun: &ResolvedFunction,
            children: &[(ArcSort, Value)],
        ) -> (Value, Vec<(ArcSort, Value)>) {
            let ints: Vec<i64> = children
                .iter()
                .map(|(_, v)| egraph.value_to_base::<i64>(*v))
                .collect();
            let res = match fun.name.as_str() {
                "+" => ints[0] + ints[1],
                "*" => ints[0] * ints[1],
                _ => -1,
            };
            (egraph.base_to_value::<i64>(res), vec![])
        }
    }

    #[derive(Clone)]
    struct IdentityConstructor;
    impl SemanticConstructor for IdentityConstructor {
        fn construct(
            &self,
            _egraph: &mut EGraph,
            _fun: &ResolvedFunction,
            children: &[(ArcSort, Value)],
        ) -> (Value, Vec<(ArcSort, Value)>) {
            (children[0].1, vec![])
        }
    }

    #[derive(Clone)]
    struct OffsetAddConstructor(i64);
    impl SemanticConstructor for OffsetAddConstructor {
        fn construct(
            &self,
            egraph: &mut EGraph,
            fun: &ResolvedFunction,
            children: &[(ArcSort, Value)],
        ) -> (Value, Vec<(ArcSort, Value)>) {
            match fun.name.as_str() {
                "lit" => (children[0].1, vec![]),
                "add" => {
                    let lhs = egraph.value_to_base::<i64>(children[0].1);
                    let rhs = egraph.value_to_base::<i64>(children[1].1);
                    (egraph.base_to_value::<i64>(lhs + rhs + self.0), vec![])
                }
                _ => (egraph.base_to_value::<i64>(-1), vec![]),
            }
        }
    }

    fn make_resolved_function(name: &str) -> ResolvedFunction {
        ResolvedFunction {
            id: ResolvedFunctionId::Prim(crate::core_relations::ExternalFunctionId::new_const(0)),
            partial_arcsorts: vec![],
            name: name.to_owned(),
        }
    }

    #[test]
    fn int_arith_constructor_evaluates_plus_and_times() {
        let mut eg = EGraph::default();
        let ctor = IntArithConstructor;
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        let fun_plus = make_resolved_function("+");
        let fun_mul = make_resolved_function("*");

        let children = vec![
            (sem_sort.clone(), eg.base_to_value::<i64>(2)),
            (sem_sort.clone(), eg.base_to_value::<i64>(3)),
        ];

        let (sem_id, extra) = ctor.construct(&mut eg, &fun_plus, &children);
        assert_eq!(eg.value_to_base::<i64>(sem_id), 5);
        assert!(extra.is_empty());

        let (sem_id, _) = ctor.construct(&mut eg, &fun_mul, &children);
        assert_eq!(eg.value_to_base::<i64>(sem_id), 6);
    }

    #[test]
    fn int_arith_constructor_falls_back_for_unknown_fun() {
        let mut eg = EGraph::default();
        let ctor = IntArithConstructor;
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        let children = vec![(sem_sort, eg.base_to_value::<i64>(2))];

        let fun_weird = make_resolved_function("foo");
        let (sem_id1, _) = ctor.construct(&mut eg, &fun_weird, &children);
        let (sem_id2, _) = ctor.construct(&mut eg, &fun_weird, &children);

        assert_eq!(sem_id1, sem_id2);
        assert_eq!(eg.value_to_base::<i64>(sem_id1), -1);
    }

    #[derive(Clone)]
    struct ExprConstructor;
    impl SemanticConstructor for ExprConstructor {
        fn construct(
            &self,
            egraph: &mut EGraph,
            fun: &ResolvedFunction,
            children: &[(ArcSort, Value)],
        ) -> (Value, Vec<(ArcSort, Value)>) {
            match fun.name.as_str() {
                "lit" => (children[0].1, vec![]),
                "add" => {
                    let lhs = egraph.value_to_base::<i64>(children[0].1);
                    let rhs = egraph.value_to_base::<i64>(children[1].1);
                    (egraph.base_to_value::<i64>(lhs + rhs), vec![])
                }
                _ => (egraph.base_to_value::<i64>(-1), vec![]),
            }
        }
    }

    #[derive(Clone)]
    struct CountingDecomposer(Arc<std::sync::atomic::AtomicUsize>);
    impl SemanticDecomposer for CountingDecomposer {
        fn decompose(
            &self,
            _egraph: &EGraph,
            _lhs_sem: Value,
            _rhs_sem: Value,
        ) -> Vec<(ArcSort, Value, Value)> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            vec![]
        }
    }

    #[test]
    fn semantic_insertion_merges_on_equal_semantics() {
        use crate::Schema;
        use crate::prelude::{add_constructor, add_sort};

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Expr").unwrap();
        add_constructor(
            &mut eg,
            "lit",
            Schema::new(vec!["i64".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();
        add_constructor(
            &mut eg,
            "add",
            Schema::new(vec!["Expr".into(), "Expr".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        eg.add_semantic_domain(eq_sort, sem_sort, Arc::new(ExprConstructor))
            .unwrap();

        let c2 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(2)])
            .unwrap();
        let c3 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(3)])
            .unwrap();
        let c1 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(1)])
            .unwrap();
        let c4 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(4)])
            .unwrap();

        let p1 = eg.semantic_add("add", &[c2, c3]).unwrap();
        let p2 = eg.semantic_add("add", &[c1, c4]).unwrap();

        let rep1 = eg.backend.get_canon_repr(p1, ColumnTy::Id);
        let rep2 = eg.backend.get_canon_repr(p2, ColumnTy::Id);
        assert_eq!(rep1, rep2);

        let sem_store = eg.semantic_store("Expr").unwrap();
        assert_eq!(
            eg.value_to_base::<i64>(sem_store.class_sem(&rep1).unwrap()),
            5
        );
    }

    #[test]
    fn semantic_insertion_keeps_distinct_semantics_apart() {
        use crate::Schema;
        use crate::prelude::{add_constructor, add_sort};

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Expr").unwrap();
        add_constructor(
            &mut eg,
            "lit",
            Schema::new(vec!["i64".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();
        add_constructor(
            &mut eg,
            "add",
            Schema::new(vec!["Expr".into(), "Expr".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        eg.add_semantic_domain(eq_sort, sem_sort, Arc::new(OffsetAddConstructor(100)))
            .unwrap();

        let c2 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(2)])
            .unwrap();
        let c3 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(3)])
            .unwrap();
        let c1 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(1)])
            .unwrap();
        let c5 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(5)])
            .unwrap();

        let p1 = eg.semantic_add("add", &[c2, c3]).unwrap(); // 5
        let p2 = eg.semantic_add("add", &[c1, c5]).unwrap(); // 6

        let rep1 = eg.backend.get_canon_repr(p1, ColumnTy::Id);
        let rep2 = eg.backend.get_canon_repr(p2, ColumnTy::Id);
        assert_ne!(rep1, rep2);
    }

    #[test]
    fn semantic_merge_invokes_decomposer_and_rebinds() {
        use crate::Schema;
        use crate::prelude::{add_constructor, add_sort};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Box").unwrap();
        add_constructor(
            &mut eg,
            "wrap",
            Schema::new(vec!["i64".into()], "Box".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Box");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        let counter = Arc::new(AtomicUsize::new(0));
        eg.add_semantic_domain(eq_sort, sem_sort.clone(), Arc::new(IdentityConstructor))
            .unwrap();
        eg.add_semantic_decomposer(
            "Box",
            sem_sort.clone(),
            sem_sort,
            Arc::new(CountingDecomposer(counter.clone())),
        )
        .unwrap();

        let c1 = eg
            .semantic_add("wrap", &[eg.base_to_value::<i64>(1)])
            .unwrap();
        let c2 = eg
            .semantic_add("wrap", &[eg.base_to_value::<i64>(2)])
            .unwrap();
        // Force both classes to share a semantic id so the merge proceeds and triggers the decomposer.
        {
            let sem_one = eg.base_to_value::<i64>(1);
            let store = eg.semantic_stores.get_mut("Box").unwrap();
            store.class_sem.insert(c2, sem_one);
        }
        let rep = eg.semantic_merge("Box", c1, c2).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        let rep_canon = eg.backend.get_canon_repr(rep, egglog_bridge::ColumnTy::Id);
        let sem_store = eg.semantic_store("Box").unwrap();
        assert_eq!(
            sem_store.sem_class(&eg.base_to_value::<i64>(1)),
            Some(rep_canon)
        );
    }

    #[test]
    fn semantic_merge_errors_on_mismatched_semantics() {
        use crate::Schema;
        use crate::prelude::{add_constructor, add_sort};

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Box").unwrap();
        add_constructor(
            &mut eg,
            "wrap",
            Schema::new(vec!["i64".into()], "Box".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Box");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        eg.add_semantic_domain(eq_sort, sem_sort, Arc::new(IdentityConstructor))
            .unwrap();

        let c1 = eg
            .semantic_add("wrap", &[eg.base_to_value::<i64>(1)])
            .unwrap();
        let c2 = eg
            .semantic_add("wrap", &[eg.base_to_value::<i64>(2)])
            .unwrap();

        let err = eg.semantic_merge("Box", c1, c2).unwrap_err();
        assert!(matches!(err, crate::Error::SemanticError(msg) if msg.contains("semantic ids differ")));
    }

    #[test]
    fn semantic_rebuild_processes_backend_rows() {
        use crate::prelude::{add_constructor, add_sort};
        use egglog_bridge::ColumnTy;

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Expr").unwrap();
        add_constructor(
            &mut eg,
            "lit",
            crate::Schema::new(vec!["i64".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();
        add_constructor(
            &mut eg,
            "add",
            crate::Schema::new(vec!["Expr".into(), "Expr".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        eg.add_semantic_domain(eq_sort, sem_sort, Arc::new(ExprConstructor))
            .unwrap();

        let lit_fn = eg.functions.get("lit").unwrap().clone();
        let add_fn = eg.functions.get("add").unwrap().clone();

        let v2 = eg.backend.base_values().get::<i64>(2);
        let v3 = eg.backend.base_values().get::<i64>(3);
        let v1 = eg.backend.base_values().get::<i64>(1);
        let v4 = eg.backend.base_values().get::<i64>(4);

        let c2 = eg.backend.add_term(lit_fn.backend_id, &[v2], "test");
        let c3 = eg.backend.add_term(lit_fn.backend_id, &[v3], "test");
        let c1 = eg.backend.add_term(lit_fn.backend_id, &[v1], "test");
        let c4 = eg.backend.add_term(lit_fn.backend_id, &[v4], "test");

        let p1 = eg
            .backend
            .add_term(add_fn.backend_id, &[c2, c3], "test_add");
        let p2 = eg
            .backend
            .add_term(add_fn.backend_id, &[c1, c4], "test_add");

        assert!(eg.semantic_rebuild_all().unwrap());

        let rep1 = eg.backend.get_canon_repr(p1, ColumnTy::Id);
        let rep2 = eg.backend.get_canon_repr(p2, ColumnTy::Id);
        assert_eq!(rep1, rep2);

        let sem_store = eg.semantic_store("Expr").unwrap();
        assert_eq!(
            eg.value_to_base::<i64>(sem_store.class_sem(&rep1).unwrap()),
            5
        );
    }

    #[test]
    #[ignore = "Upward semantic recomputation wiring pending full integration"]
    fn child_merge_triggers_parent_semantic_merge() {
        use crate::Schema;
        use crate::prelude::{add_constructor, add_sort};
        use egglog_bridge::ColumnTy;

        let mut eg = EGraph::default();
        add_sort(&mut eg, "Expr").unwrap();
        add_constructor(
            &mut eg,
            "lit",
            Schema::new(vec!["i64".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();
        add_constructor(
            &mut eg,
            "add",
            Schema::new(vec!["Expr".into(), "Expr".into()], "Expr".into()),
            None,
            false,
        )
        .unwrap();

        let eq_sort = eg.get_sort_by::<EqSort>(|s| s.name == "Expr");
        let sem_sort = eg.get_sort_by_name("i64").unwrap().clone();
        eg.add_semantic_domain(eq_sort, sem_sort, Arc::new(ExprConstructor))
            .unwrap();

        let c2 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(2)])
            .unwrap();
        let c3 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(3)])
            .unwrap();
        let c4 = eg
            .semantic_add("lit", &[eg.base_to_value::<i64>(4)])
            .unwrap();

        let p_a = eg.semantic_add("add", &[c2, c3]).unwrap(); // sem 105
        let p_b = eg.semantic_add("add", &[c4, c2]).unwrap(); // sem 106

        // Merge children 2 and 3; semantics collapse to 2, so parent_a becomes 104.
        let _ = eg.semantic_merge("Expr", c2, c3).unwrap();

        let sem_store = eg.semantic_store("Expr").unwrap();
        let class_104 = sem_store
            .sem_class(&eg.base_to_value::<i64>(104))
            .expect("sem 104 should map to a class");
        let class_106 = sem_store
            .sem_class(&eg.base_to_value::<i64>(106))
            .expect("sem 106 should map to a class");
        assert_eq!(class_104, eg.backend.get_canon_repr(p_a, ColumnTy::Id));
        assert_eq!(class_106, eg.backend.get_canon_repr(p_b, ColumnTy::Id));
    }
}
