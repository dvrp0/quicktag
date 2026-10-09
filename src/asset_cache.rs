//! Bounded reuse of immutable package assets. Values are charged for their owned
//! allocations by the caller; the LRU also charges keys and entry overhead.

use std::hash::Hash;
use std::sync::{Arc, Weak};

use linked_hash_map::LinkedHashMap;


/// Exercise the original decoding path without reading or populating caches.

pub(crate) struct AssetCache<S, K, V> {
    scope: Weak<S>,
    entries: LinkedHashMap<K, (V, usize)>,
    bytes: usize,
    budget: usize,
}

impl<S, K: Eq + Hash, V> AssetCache<S, K, V> {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            scope: Weak::new(),
            entries: LinkedHashMap::new(),
            bytes: 0,
            budget,
        }
    }

    fn select_scope(&mut self, scope: &Arc<S>) {
        if !Weak::ptr_eq(&self.scope, &Arc::downgrade(scope)) {
            self.clear();
            self.scope = Arc::downgrade(scope);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    pub(crate) fn get(&mut self, scope: &Arc<S>, key: &K) -> Option<&V> {
        self.select_scope(scope);
        self.entries.get_refresh(key).map(|(value, _)| &*value)
    }

    pub(crate) fn insert(&mut self, scope: &Arc<S>, key: K, value: V, heap_bytes: usize) {
        self.select_scope(scope);
        if let Some((_, bytes)) = self.entries.remove(&key) {
            self.bytes -= bytes;
        }
        let bytes = heap_bytes.saturating_add(std::mem::size_of::<(K, V, usize)>() + 64);
        // Oversized assets remain usable by the caller, without flushing useful
        // entries or exceeding the retention budget.
        if bytes > self.budget {
            return;
        }
        while self.bytes > self.budget - bytes {
            let Some((_, (_, removed))) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= removed;
        }
        self.entries.insert(key, (value, bytes));
        self.bytes += bytes;
    }
}
