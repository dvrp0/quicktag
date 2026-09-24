//! Bounded reuse of immutable package assets. Values are charged for their owned
//! allocations by the caller; the LRU also charges keys and entry overhead.

use std::hash::Hash;
use std::sync::{Arc, Weak};

use linked_hash_map::LinkedHashMap;

#[cfg(test)]
thread_local! {
    static BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Exercise the original decoding path without reading or populating caches.
#[cfg(test)]
pub(crate) fn without_asset_cache<R>(run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BYPASS.set(self.0);
        }
    }
    let _restore = Restore(BYPASS.replace(true));
    run()
}

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
        #[cfg(test)]
        if BYPASS.get() {
            return None;
        }
        self.select_scope(scope);
        self.entries.get_refresh(key).map(|(value, _)| &*value)
    }

    pub(crate) fn insert(&mut self, scope: &Arc<S>, key: K, value: V, heap_bytes: usize) {
        #[cfg(test)]
        if BYPASS.get() {
            return;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_evicts_least_recent_asset_within_budget() {
        let scope = Arc::new(());
        let charge = std::mem::size_of::<(u32, u32, usize)>() + 64;
        let mut cache = AssetCache::new(charge * 2);
        cache.insert(&scope, 1, 10, 0);
        cache.insert(&scope, 2, 20, 0);
        assert_eq!(cache.get(&scope, &1), Some(&10));
        cache.insert(&scope, 3, 30, 0);
        assert_eq!(cache.get(&scope, &2), None);
        assert_eq!(cache.get(&scope, &1), Some(&10));
        assert_eq!(cache.get(&scope, &3), Some(&30));
        assert_eq!(cache.bytes, charge * 2);
    }

    #[test]
    fn replacement_and_oversized_assets_preserve_accounting() {
        let scope = Arc::new(());
        let mut cache = AssetCache::new(1024);
        cache.insert(&scope, 1, 10, 100);
        cache.insert(&scope, 1, 20, 200);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(
            cache.bytes,
            200 + std::mem::size_of::<(i32, i32, usize)>() + 64
        );
        cache.insert(&scope, 2, 30, 1024);
        assert_eq!(cache.get(&scope, &2), None);
        assert_eq!(cache.get(&scope, &1), Some(&20));
    }

    #[test]
    fn changing_package_scope_invalidates_same_tag() {
        let first = Arc::new(());
        let second = Arc::new(());
        let mut cache = AssetCache::new(1024);
        cache.insert(&first, 1, 10, 0);
        assert_eq!(cache.get(&second, &1), None);
        cache.insert(&second, 1, 20, 0);
        assert_eq!(cache.get(&second, &1), Some(&20));
        assert_eq!(cache.get(&first, &1), None);
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn cache_does_not_keep_package_scope_alive() {
        let scope = Arc::new(());
        let mut cache = AssetCache::new(1024);
        cache.insert(&scope, 1, 10, 0);
        assert_eq!(Arc::strong_count(&scope), 1);
        drop(scope);
        assert!(cache.scope.upgrade().is_none());
        assert_eq!(cache.get(&Arc::new(()), &1), None);
    }

    #[test]
    fn explicit_invalidation_discards_values_and_accounting() {
        let scope = Arc::new(());
        let mut cache = AssetCache::new(1024);
        cache.insert(&scope, 1, 10, 100);
        cache.clear();
        assert_eq!(cache.get(&scope, &1), None);
        assert_eq!(cache.bytes, 0);
        cache.insert(&scope, 1, 20, 200);
        assert_eq!(cache.get(&scope, &1), Some(&20));
    }
}
