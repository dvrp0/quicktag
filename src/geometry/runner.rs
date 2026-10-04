//! Runner shells follow their authored Pattern graph, including 64-bit links.
use super::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct RunnerShellAssembly {
    pub pattern: TagHash,
    pub nested_patterns: Vec<TagHash>,
    pub parts: Vec<RunnerShellPart>,
}

#[derive(Clone, Debug)]
pub struct RunnerShellPart {
    pub component: TagHash,
    /// Authored ancestry, from selected shell to the nearest owning Pattern.
    pub pattern_path: Vec<TagHash>,
    pub geometry: Vec<TagHash>,
}

impl RunnerShellAssembly {
    #[cfg(test)]
    pub fn contains(&self, tag: TagHash) -> bool {
        self.pattern == tag
            || self.nested_patterns.contains(&tag)
            || self
                .parts
                .iter()
                .any(|part| part.component == tag || part.geometry.contains(&tag))
    }
    pub fn resolve(cache: &TagCache, model: TagHash) -> Option<Self> {
        if package_manager().get_entry(model)?.reference != CLASS_PATTERN {
            return None;
        }
        let pattern = model;
        let mut queue = VecDeque::from([(pattern, vec![pattern])]);
        let mut seen = rustc_hash::FxHashSet::default();
        let mut seen_geometry = rustc_hash::FxHashSet::default();
        let mut parts = Vec::<RunnerShellPart>::new();
        let mut nested_patterns = vec![];
        while let Some((parent, path)) = queue.pop_front() {
            if !seen.insert((parent, path.clone())) {
                continue;
            }
            if parent != pattern
                && package_manager()
                    .get_entry(parent)
                    .is_some_and(|entry| entry.reference == CLASS_PATTERN)
            {
                if !nested_patterns.contains(&parent) { nested_patterns.push(parent); }
            }
            let mut geometry = vec![];
            for child in pattern_graph_children(cache, parent) {
                match package_manager()
                    .get_entry(child)
                    .map(|entry| entry.reference)
                {
                    Some(CLASS_PATTERN) if !path.contains(&child) => {
                        let mut child_path=path.clone(); child_path.push(child);
                        queue.push_back((child,child_path));
                    }
                    Some(CLASS_PATTERN_COMPONENT) => queue.push_back((child,path.clone())),
                    Some(CLASS_GEOMETRY_RESOURCE) => {
                        if seen_geometry.insert(child) {
                            geometry.push(child);
                        } else {
                            // Shared geometry with different object owners has no
                            // unique local scope. Keep geometry once; reject scope
                            // assignment rather than picking traversal order.
                            for part in &mut parts {
                                if part.geometry.contains(&child) && part.pattern_path != path {
                                    part.pattern_path.clear();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !geometry.is_empty() {
                geometry.sort_unstable();
                parts.push(RunnerShellPart {
                    component: parent,
                    pattern_path: path,
                    geometry,
                });
            }
        }
        (!parts.is_empty()).then_some(Self {
            pattern,
            nested_patterns,
            parts,
        })
    }

    pub fn geometry(&self) -> Vec<TagHash> {
        self.parts
            .iter()
            .flat_map(|part| part.geometry.iter().copied())
            .collect()
    }

    pub fn load(&self, cache: Arc<TagCache>) -> Option<GeometryTagPreview> {
        let entry = package_manager().get_entry(self.pattern)?;
        let mut model = load_model_preview_from_tags(
            cache.clone(),
            self.pattern,
            &entry,
            "Runner shell",
            self.geometry(),
            &[],
        );
        // Cosmetic channels belong to this shell's immediate components. Never
        // search reverse/shared graph edges, which can reach another skin.
        let palette = pattern_graph_children(&cache, self.pattern)
            .into_iter()
            .filter(|tag| {
                package_manager()
                    .get_entry(*tag)
                    .is_some_and(|entry| entry.reference == CLASS_PATTERN_COMPONENT)
            })
            .find_map(|tag| {
                package_manager()
                    .read_tag(tag)
                    .ok()
                    .and_then(|data| decode_weapon_skin_gear_dye_palette(&data))
            });
        if let (Some(palette), Some(wireframe)) = (palette, model.wireframe.as_mut()) {
            for range in &mut wireframe.material_ranges {
                let Some(technique) = range.technique else {
                    continue;
                };
                let Some(default) = technique_default_gear_dye_color(technique).or_else(|| {
                    range
                        .textures
                        .character_surface
                        .is_some()
                        .then_some([1.0; 4])
                }) else {
                    continue;
                };
                let Some(dye) = range
                    .gear_dye_change_color_index
                    .and_then(|index| palette.get(index as usize).copied())
                else {
                    continue;
                };
                range.textures.gear_dye_palette = Some(palette);
                range.textures.gear_dye = Some(dye);
                range.textures.gear_dye_default = Some(default);
            }
        }
        Some(GeometryTagPreview {
            kind: GeometryPreviewKind::Model(model),
        })
    }
}

#[cfg(test)]
pub fn runner_shell_assemblies(
    cache: &TagCache,
    containers: &[TagHash],
) -> Vec<RunnerShellAssembly> {
    containers
        .iter()
        .copied()
        .filter(|tag| {
            package_manager()
                .get_entry(*tag)
                .is_some_and(|entry| entry.reference == CLASS_PATTERN)
        })
        .filter(|tag| ancestor_pattern_roots(cache, *tag).is_empty())
        .filter_map(|tag| RunnerShellAssembly::resolve(cache, tag))
        .filter(|shell| !shell.nested_patterns.is_empty())
        .collect()
}
