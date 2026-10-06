//! Runner shells follow their authored Pattern graph, including 64-bit links.
use super::*;
use std::collections::VecDeque;

/// Package holding the entities the game spawns, runners included.
const RUNNER_ENTITY_PACKAGE: &str = "sr_sandbox";
/// Array of the slots an entity plugs child entities into (a runner: head, body).
const CLASS_ENTITY_SLOT_TABLE: u32 = 0x808032BA;

#[derive(Clone, Debug)]
pub struct RunnerShellAssembly {
    pub pattern: TagHash,
    pub nested_patterns: Vec<TagHash>,
    pub parts: Vec<RunnerShellPart>,
    /// The runner's own entity definition, when the caller knows which runner
    /// this shell belongs to. Its default gear is rendered with the shell.
    pub runner: Option<TagHash>,
    /// Row of the runner's dye table this skin uses, from its item definition.
    pub dye_row: Option<u32>,
    /// Geometry of the props the clip being played spawns.
    pub props: Vec<TagHash>,

}

#[derive(Clone, Debug)]
pub struct RunnerShellPart {
    pub component: TagHash,
    /// Authored ancestry, from selected shell to the nearest owning Pattern.
    pub pattern_path: Vec<TagHash>,
    pub geometry: Vec<TagHash>,
}

impl RunnerShellAssembly {
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
            runner: None,
            dye_row: None,
            props: vec![],
        })
    }

    pub fn geometry(&self) -> Vec<TagHash> {
        self.parts
            .iter()
            .flat_map(|part| part.geometry.iter().copied())
            .collect()
    }

    /// Geometry the runner wears by default: the gear its own entity definition
    /// carries, which no skin repeats (the Destroyer's forearm shield, the
    /// drones at Triage's thighs).
    ///
    /// The skin is skinned to its runner's body skeleton, and the entity
    /// definitions built on that skeleton in the game-entity package are the
    /// runner itself. Several are; the one with the least geometry is the bare
    /// runner, without the weapons and effects the fuller ones gather. Only
    /// geometry from that package is gear: the rest is shared placeholder
    /// content. Runners sharing one skeleton cannot be told apart this way, so
    /// when the bare definitions disagree nothing is added.
    fn default_gear(&self, cache: &TagCache, sources: &[AuthoredGeometryInput]) -> Vec<TagHash> {
        let Some(skeleton) = crate::animation::runner_body_skeleton(sources) else {
            return vec![];
        };
        let package = |tag: TagHash| package_manager().package_paths.get(&tag.pkg_id()).map(|path| path.name.clone());
        let worn = self.geometry();
        let candidates = crate::animation::definitions_of_skeleton(skeleton)
            .into_iter()
            .filter(|definition| package(*definition).as_deref() == Some(RUNNER_ENTITY_PACKAGE))
            .filter_map(|definition| {
                let mut gear = Self::resolve(cache, definition)?
                    .geometry()
                    .into_iter()
                    .filter(|geometry| package(*geometry) == package(definition) && !worn.contains(geometry))
                    .collect::<Vec<_>>();
                gear.sort_by_key(|tag| tag.0);
                log::debug!("runner definition {definition}: gear geometry {gear:?}");
                (!gear.is_empty()).then_some(gear)
            })
            .collect::<Vec<_>>();
        let Some(least) = candidates.iter().map(Vec::len).min() else {
            return vec![];
        };
        let mut bare = candidates.into_iter().filter(|gear| gear.len() == least).collect::<Vec<_>>();
        bare.dedup();
        match bare.as_slice() {
            [gear] => gear.clone(),
            _ => vec![],
        }
    }

    /// Gear of the runner whose entity definition is `runner`.
    ///
    /// A runner exists as several definitions sharing one slot-table component
    /// (the table its head and body are plugged into): the full one the
    /// investment data names, holding every skill model, and a bare one. The
    /// bare one is the definition with the least geometry of its own package,
    /// and that geometry is the gear, once anything another runner's bare
    /// definition also lists is set aside as shared content.
    fn runner_gear(cache: &TagCache, runner: TagHash) -> Vec<TagHash> {
        let package = |tag: TagHash| package_manager().package_paths.get(&tag.pkg_id()).map(|path| path.name.clone());
        let is_class = |tag: TagHash, class: u32| {
            package_manager().get_entry(tag).is_some_and(|entry| entry.reference == class)
        };
        // Bare gear list per slot table, over every runner in the package.
        let mut bare_by_slots = rustc_hash::FxHashMap::<TagHash, Vec<TagHash>>::default();
        let mut slots_of_runner = None;
        for (component, _) in package_manager().get_all_by_reference(CLASS_PATTERN_COMPONENT) {
            if package(component).as_deref() != Some(RUNNER_ENTITY_PACKAGE) {
                continue;
            }
            let Ok(data) = package_manager().read_tag(component) else { continue };
            let endian = package_manager().version.endian();
            if !scan_arrays(&data, endian).iter().any(|array| array.class == CLASS_ENTITY_SLOT_TABLE) {
                continue;
            }
            let definitions = cache
                .hashes
                .get(&component)
                .map(|scan| scan.references.clone())
                .unwrap_or_default()
                .into_iter()
                .filter(|tag| is_class(*tag, CLASS_PATTERN))
                .unique()
                .collect::<Vec<_>>();
            if definitions.contains(&runner) {
                slots_of_runner = Some(component);
            }
            let bare = definitions
                .into_iter()
                .filter_map(|definition| {
                    let mut gear = Self::resolve(cache, definition)?
                        .geometry()
                        .into_iter()
                        .filter(|geometry| package(*geometry) == package(definition))
                        .collect::<Vec<_>>();
                    gear.sort_by_key(|tag| tag.0);
                    (!gear.is_empty()).then_some(gear)
                })
                .min_by_key(Vec::len);
            if let Some(bare) = bare {
                bare_by_slots.insert(component, bare);
            }
        }
        let Some(slots) = slots_of_runner else { return vec![] };
        let Some(bare) = bare_by_slots.get(&slots) else { return vec![] };
        bare.iter()
            .copied()
            .filter(|geometry| {
                bare_by_slots.iter().all(|(other, listed)| *other == slots || !listed.contains(geometry))
            })
            .collect()
    }

    /// Geometry of the runner's own gear and spawned props, rendered with the shell.
    pub(crate) fn gear(&self, cache: &TagCache) -> Vec<TagHash> {
        let worn = self.geometry();
        self.runner.map_or_else(Vec::new, |runner| {
            Self::runner_gear(cache, runner)
                .into_iter()
                .chain(self.props.iter().copied())
                .filter(|gear| !worn.contains(gear))
                .collect()
        })
    }

    /// Channel values the skin's dye row gives every material of its runner.
    pub(crate) fn dye_channels(&self) -> Vec<(u32, [f32; 4])> {
        self.runner
            .zip(self.dye_row)
            .and_then(|(runner, row)| runner_dye_rows(runner).into_iter().nth(row as usize))
            .map_or_else(Vec::new, |(_, row)| row.into_iter().collect())
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
        let gear = model
            .wireframe
            .as_ref()
            .map_or_else(Vec::new, |wireframe| match self.runner {
                Some(runner) => Self::runner_gear(&cache, runner).into_iter().filter(|gear| !self.geometry().contains(gear)).collect(),
                None => self.default_gear(&cache, &wireframe.authored_inputs),
            });
        // Props a playing clip spawns are rendered with the shell as well.
        let extra = gear.into_iter().chain(self.props.iter().copied()).collect::<Vec<_>>();
        if !extra.is_empty() {
            let geometry = self.geometry().into_iter().chain(extra).collect();

            model = load_model_preview_from_tags(cache.clone(), self.pattern, &entry, "Runner shell", geometry, &[]);
        }
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

/// Rows of a runner's dye table, one per skin, keyed by the skin's codename.
const CLASS_RUNNER_DYE_ROWS: u32 = 0x80809627;

/// Channel values one skin gives its runner's dyed materials.
type DyeRow = rustc_hash::FxHashMap<u32, [f32; 4]>;

/// The dye table of a runner: a component of its entity definition holding,
/// per skin codename, a value for each dye channel its gear materials read.
fn runner_dye_rows(runner: TagHash) -> Vec<(u32, DyeRow)> {
    let endian = package_manager().version.endian();
    let Ok(definition) = package_manager().read_tag(runner) else {
        return vec![];
    };
    let offset_in = |data: &[u8], part: &[u8]| part.as_ptr() as usize - data.as_ptr() as usize;
    definition
        .chunks_exact(4)
        .map(|word| TagHash(u32::from_le_bytes(word.try_into().unwrap())))
        .filter(|tag| {
            package_manager().get_entry(*tag).is_some_and(|entry| entry.reference == CLASS_PATTERN_COMPONENT)
        })
        .unique()
        .filter_map(|component| package_manager().read_tag(component).ok())
        .find_map(|data| {
            let table = scan_arrays(&data, endian).into_iter().find(|array| array.class == CLASS_RUNNER_DYE_ROWS)?;
            let rows = array_records(&data, table, 0x18)
                .into_iter()
                .filter_map(|record| {
                    let record_offset = offset_in(&data, record);
                    let entries = read_array(&data, record_offset + 0x08, 0x48, endian)?;
                    let entries_offset = offset_in(&data, entries);
                    let values = (0..entries.len() / 0x48)
                        .filter_map(|index| {
                            let entry = entries_offset + index * 0x48;
                            let value = read_array(&data, entry + 0x20, 0x10, endian)?;
                            Some((read_u32_at(&data, entry, endian)?, read_vec4_f32(value, 0, endian)?))
                        })
                        .collect();
                    Some((read_u32_at(record, 0, endian)?, values))
                })
                .collect::<Vec<_>>();
            Some(rows)
        })
        .unwrap_or_default()
}

