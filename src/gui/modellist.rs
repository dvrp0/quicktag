use std::{io::Write, path::PathBuf, sync::Arc};

use eframe::egui::{
    self, Color32, RichText, Sense, Stroke,
    epaint::{Mesh, Vertex},
    pos2, vec2,
};
use eframe::egui_wgpu::Callback;
use itertools::Itertools;
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use tiger_pkg::{TagHash, manager::PackagePath, package::UEntryHeader, package_manager};

use crate::geometry::{
    GeometryPreviewKind, GeometryTagPreview, ModelTagInfo, ModelTagRole, RunnerShellAssembly,
    UvTransformPreview, WeaponModPreviewAttachment, WeaponModSocketIndex, WireframeMaterialLayer,
    WireframePreview, is_model_catalog_reference, model_has_render_geometry,
    model_info_for_reference, pattern_component_descendants,
    weapon_unoccupied_default_mod_patterns,
};
use crate::gui::common::ResponseExt;
use crate::gui::tag::format_tag_entry;
use crate::material::is_sticker_proxy_technique;
use crate::render::{channels::ModelChannels, tfx::TfxRuntimeInputs};
use crate::texture::cache::{MaterialTextureKey, TextureCache};
use crate::util::{format_file_size, ui_image_rotated};

use super::gear::{
    ModelCharmEntry, ModelMeleeSkinEntry, ModelModEntry, ModelRunnerSkinEntry, ModelWeaponCatalog,
    ModelWeaponEntry, ModelWeaponSkinEntry,
};
use super::model_renderer::{
    GpuModelPreview, LightingModel, ModelCameraFrame, ModelEnvironment, ModelExportCamera,
    ModelLightTransform, ModelPaintCallback, fixed_light_direction_to_view,
    fixed_view_direction_to_light, light_cast_direction, light_source_position,
};
use super::{TOASTS, View, ViewAction};

pub(super) const DEFAULT_MODEL_YAW: f32 = -std::f32::consts::FRAC_PI_2;
const DEFAULT_WEAPON_YAW: f32 = -24.0_f32.to_radians();
const DEFAULT_WEAPON_PITCH: f32 = 14.0_f32.to_radians();
const DEFAULT_PREVIEW_ZOOM: f32 = 2.5;
const MODEL_EXPORT_WIDTH: u32 = 4198;
const MODEL_EXPORT_HEIGHT: u32 = 2048;
const MODEL_PARAMETER_RANGE: std::ops::RangeInclusive<f32> = -10.0..=10.0;
const MODEL_OVERLAY_INSET: f32 = 8.0;
const MODEL_OVERLAY_STACK_STEP: f32 = 51.0;

fn model_overlay_frame() -> egui::Frame {
    egui::Frame::default()
        .fill(Color32::from_rgba_unmultiplied(10, 13, 18, 238))
        .stroke(Stroke::new(1.0, Color32::from_rgb(73, 82, 96)))
        .corner_radius(6)
        .inner_margin(8)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ModelExportFormat {
    #[default]
    Png,
    WebP,
}

impl ModelExportFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Png => "PNG",
            Self::WebP => "WebP",
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::WebP => "webp",
        }
    }

    fn dialog_label(self) -> &'static str {
        match self {
            Self::Png => "PNG image",
            Self::WebP => "WebP image",
        }
    }

    fn image_format(self) -> image::ImageFormat {
        match self {
            Self::Png => image::ImageFormat::Png,
            Self::WebP => image::ImageFormat::WebP,
        }
    }
}

fn model_export_filename(
    model: TagHash,
    modifications: &[(TagHash, &'static str)],
    format: ModelExportFormat,
) -> String {
    let suffix = modifications
        .iter()
        .map(|(tag, rarity)| format!("{tag}-{rarity}"))
        .join("_");
    if suffix.is_empty() {
        format!("{model}.{}", format.extension())
    } else {
        format!("{model}_{suffix}.{}", format.extension())
    }
}

fn model_modded_export_filename(model: TagHash) -> String {
    format!("{model}.zip")
}

fn high_rarity_mod_combinations(weapon: &ModelWeaponEntry) -> Vec<Vec<ModelModEntry>> {
    let pools = weapon
        .slots
        .iter()
        .filter_map(|slot| {
            let mut mods = slot
                .mods
                .iter()
                .filter(|item| matches!(item.rarity_code, "S" | "P" | "C"))
                .cloned()
                .collect_vec();
            mods.sort_by_key(|item| (item.model_tag, item.rarity_code));
            mods.dedup_by_key(|item| (item.model_tag, item.rarity_code));
            (!mods.is_empty()).then_some(mods)
        })
        .collect_vec();
    if pools.is_empty() {
        return vec![];
    }
    pools.into_iter().fold(vec![vec![]], |combinations, pool| {
        combinations
            .into_iter()
            .flat_map(|combination| {
                pool.iter().cloned().map(move |item| {
                    let mut next = combination.clone();
                    next.push(item);
                    next
                })
            })
            .collect()
    })
}

struct ModdedExportJob {
    model_tag: TagHash,
    weapon: ModelWeaponEntry,
    combinations: Vec<Vec<ModelModEntry>>,
    next: usize,
    show_default_mods: bool,
    show_stickers: bool,
    environment: ModelEnvironment,
    channels: Option<ModelChannels>,
    export_camera: Option<ModelExportCamera>,
    format: ModelExportFormat,
    path: PathBuf,
    zip: Option<zip::ZipWriter<std::fs::File>>,
}

impl ModdedExportJob {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_tag: TagHash,
        weapon: ModelWeaponEntry,
        show_default_mods: bool,
        show_stickers: bool,
        environment: ModelEnvironment,
        channels: Option<ModelChannels>,
        export_camera: Option<ModelExportCamera>,
        format: ModelExportFormat,
        path: PathBuf,
    ) -> anyhow::Result<Self> {
        let combinations = high_rarity_mod_combinations(&weapon);
        anyhow::ensure!(
            !combinations.is_empty(),
            "current gun has no Superior, Prestige, or Contraband mods"
        );
        let zip = zip::ZipWriter::new(std::fs::File::create(&path)?);
        Ok(Self {
            model_tag,
            weapon,
            combinations,
            next: 0,
            show_default_mods,
            show_stickers,
            environment,
            channels,
            export_camera,
            format,
            path,
            zip: Some(zip),
        })
    }

    fn progress(&self) -> (usize, usize) {
        (self.next, self.combinations.len())
    }

    fn step(
        &mut self,
        cache: &Arc<TagCache>,
        texture_cache: &TextureCache,
    ) -> anyhow::Result<bool> {
        if self.next == self.combinations.len() {
            self.zip
                .take()
                .expect("unfinished export has zip")
                .finish()?;
            return Ok(true);
        }
        let entry = package_manager()
            .get_entry(self.model_tag)
            .ok_or_else(|| anyhow::anyhow!("model {} is unavailable", self.model_tag))?;
        let socket_owner = self.weapon.socket_owner.unwrap_or(self.weapon.owner_tag);
        let combination = &self.combinations[self.next];
        let mut attachments = combination
            .iter()
            .map(|item| WeaponModPreviewAttachment {
                model_tag: item.model_tag,
                rarity: item.preview_rarity,
                unique_id: 0.5,
            })
            .collect_vec();
        let equipped = attachments.iter().map(|item| item.model_tag).collect_vec();
        if self.show_default_mods {
            attachments.extend(
                weapon_unoccupied_default_mod_patterns(
                    &cache,
                    self.weapon.owner_tag,
                    socket_owner,
                    &equipped,
                )
                .into_iter()
                .map(|model_tag| WeaponModPreviewAttachment {
                    model_tag,
                    rarity: None,
                    unique_id: 0.5,
                }),
            );
        }
        let preview = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            cache.clone(),
            self.model_tag,
            &entry,
            self.weapon.owner_tag,
            socket_owner,
            &attachments,
        )
        .ok_or_else(|| {
            anyhow::anyhow!("failed to assemble {} with selected mods", self.model_tag)
        })?;
        let GeometryPreviewKind::Model(model) = &preview.kind else {
            anyhow::bail!("{} did not decode as a model", self.model_tag);
        };
        let wireframe = model
            .wireframe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("wireframe unavailable for {}", self.model_tag))?;
        let fallback_color = wireframe_preview_textures(wireframe, &model.textures)
            .first()
            .copied();
        let mut runtime_inputs = TfxRuntimeInputs::for_model_preview(cache, self.weapon.owner_tag);
        if let Some(channels) = &self.channels { channels.apply_overrides(&mut runtime_inputs); }
        let gpu = Arc::new(
            GpuModelPreview::create(
                &texture_cache.render_state.device,
                wireframe,
                fallback_color,
                &runtime_inputs,
            )
            .ok_or_else(|| {
                anyhow::anyhow!("failed to create GPU preview for {}", self.model_tag)
            })?,
        );
        let export_rect = egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            vec2(MODEL_EXPORT_WIDTH as f32, MODEL_EXPORT_HEIGHT as f32),
        );
        let callback = ModelPaintCallback::new(
            gpu,
            texture_cache,
            wireframe,
            model.preview_uv_transform(),
            None,
            DEFAULT_WEAPON_YAW,
            DEFAULT_WEAPON_PITCH,
            1.0,
            egui::Vec2::ZERO,
            self.show_stickers,
            export_rect,
            1.0,
            self.environment,
        )
        .with_export_camera(self.export_camera);
        let descriptors = combination
            .iter()
            .map(|item| (item.model_tag, item.rarity_code))
            .collect_vec();
        let zip = self.zip.as_mut().expect("unfinished export has zip");
        zip.start_file(
            model_export_filename(self.model_tag, &descriptors, self.format),
            // Encoded image payloads are already compressed; ZIP deflate only wastes CPU.
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )?;
        zip.write_all(&callback.export_image_bytes(
            &texture_cache.render_state,
            [MODEL_EXPORT_WIDTH, MODEL_EXPORT_HEIGHT],
            self.format.image_format(),
        )?)?;
        self.next += 1;
        Ok(false)
    }
}

pub struct ModelsView {
    cache: Arc<TagCache>,
    texture_cache: TextureCache,
    selected_package: u16,
    packages_with_models: Vec<u16>,
    package_filter: String,
    model_filter: String,
    hide_empty_models: bool,
    models: Vec<ModelListEntry>,
    selected_model: Option<TagHash>,
    preview: Option<GeometryTagPreview>,
    gpu_model_preview: Option<Arc<GpuModelPreview>>,
    preview_channels: Option<ModelChannels>,
    preview_runtime_inputs: Option<TfxRuntimeInputs>,
    channel_render_error: Option<String>,
    preview_camera_frame: Option<ModelCameraFrame>,
    weapon_export_camera: Option<ModelExportCamera>,
    preview_yaw: f32,
    preview_pitch: f32,
    preview_zoom: f32,
    preview_pan: egui::Vec2,
    preview_show_wireframe: bool,
    preview_show_stickers: bool,
    preview_show_default_mods: bool,
    preview_environment: ModelEnvironment,
    export_all_mods: bool,
    export_format: ModelExportFormat,
    modded_export: Option<ModdedExportJob>,
    weapon_catalog: ModelWeaponCatalog,
    active_weapon: Option<usize>,
    selected_mods: Vec<Option<usize>>,
    selected_mod_unique_ids: Vec<Option<f32>>,
    mod_unique_rng: u32,
    runner_models: rustc_hash::FxHashMap<TagHash, RunnerShellAssembly>,
}

#[derive(Clone, Copy)]
struct ProjectedVertex {
    pos: egui::Pos2,
    depth: f32,
    view: [f32; 3],
}

#[derive(Clone, Copy)]
struct ProjectedTriangle {
    indices: [u32; 3],
    depth: f32,
    texture: Option<TagHash>,
    normal: Option<TagHash>,
    emissive: Option<TagHash>,
    color_tint: [u8; 4],
    emissive_strength: u8,
    light: f32,
}

impl ModelsView {
    pub fn new(cache: Arc<TagCache>, texture_cache: TextureCache) -> Self {
        Self {
            cache,
            texture_cache,
            selected_package: u16::MAX,
            packages_with_models: Self::search_models(None),
            package_filter: String::new(),
            model_filter: String::new(),
            hide_empty_models: true,
            models: vec![],
            selected_model: None,
            preview: None,
            gpu_model_preview: None,
            preview_channels: None,
            preview_runtime_inputs: None,
            channel_render_error: None,
            preview_camera_frame: None,
            weapon_export_camera: None,
            preview_yaw: DEFAULT_WEAPON_YAW,
            preview_pitch: DEFAULT_WEAPON_PITCH,
            preview_zoom: DEFAULT_PREVIEW_ZOOM,
            preview_pan: vec2(0.0, 0.0),
            preview_show_wireframe: false,
            preview_show_stickers: false,
            preview_show_default_mods: true,
            preview_environment: ModelEnvironment::default(),
            export_all_mods: false,
            export_format: ModelExportFormat::default(),
            modded_export: None,
            weapon_catalog: ModelWeaponCatalog::default(),
            active_weapon: None,
            selected_mods: vec![],
            selected_mod_unique_ids: vec![],
            mod_unique_rng: 0xA341_316C,
            runner_models: Default::default(),
        }
    }

    pub fn set_weapon_catalog(&mut self, catalog: ModelWeaponCatalog) {
        let previous_tags = self.selected_mod_tags();
        let previous_unique_ids = self.selected_mod_unique_ids.clone();
        let previous_owners = self
            .weapon_catalog
            .weapons
            .iter()
            .filter_map(|weapon| Some((weapon.owner_tag, weapon.socket_owner?)))
            .collect::<Vec<_>>();
        self.weapon_catalog = catalog;
        self.refresh_runner_models();
        for weapon in &mut self.weapon_catalog.weapons {
            weapon.socket_owner = previous_owners
                .iter()
                .find(|(owner_tag, _owner)| *owner_tag == weapon.owner_tag)
                .map(|(_owner_tag, owner)| *owner);
        }
        self.resolve_weapon_socket_owners();
        self.detect_selected_weapon();
        if let Some(weapon) = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
        {
            for (slot_index, slot) in weapon.slots.iter().enumerate() {
                self.selected_mods[slot_index] = previous_tags
                    .get(slot_index)
                    .copied()
                    .flatten()
                    .and_then(|tag| {
                        slot.mods
                            .iter()
                            .position(|modification| modification.model_tag == tag)
                    });
                if self.selected_mods[slot_index].is_some() {
                    self.selected_mod_unique_ids[slot_index] =
                        previous_unique_ids.get(slot_index).copied().flatten();
                }
            }
        }
        if self.selected_model.is_some() {
            self.rebuild_model_preview();
            self.rebuild_weapon_reference_frames();
        }
    }

    pub fn set_cache(&mut self, cache: Arc<TagCache>) {
        self.cache = cache;
        self.refresh_runner_models();
        for weapon in &mut self.weapon_catalog.weapons {
            weapon.socket_owner = None;
        }
        self.resolve_weapon_socket_owners();
        self.preview = None;
        self.gpu_model_preview = None;
        self.preview_camera_frame = None;
        self.weapon_export_camera = None;
        if let Some(tag) = self.selected_model {
            self.load_model(tag);
        }
    }

    pub fn show_model(&mut self, tag: TagHash) {
        let tag = runner_model_root(&self.runner_models, tag).unwrap_or(tag);
        self.package_filter.clear();
        self.model_filter.clear();
        self.packages_with_models = Self::search_models(None);
        self.selected_package = tag.pkg_id();
        self.load_package_models(self.selected_package);
        if self.models.iter().any(|entry| entry.tag == tag) {
            self.load_model(tag);
        }
    }

    fn resolve_weapon_socket_owners(&mut self) {
        if self.cache.hashes.is_empty()
            || !self
                .weapon_catalog
                .weapons
                .iter()
                .any(|weapon| weapon.socket_owner.is_none() && !weapon.slots.is_empty())
        {
            return;
        }
        let socket_index = WeaponModSocketIndex::new();
        for weapon in &mut self.weapon_catalog.weapons {
            if weapon.socket_owner.is_some() || weapon.slots.is_empty() {
                continue;
            }
            let modifications = weapon
                .slots
                .iter()
                .flat_map(|slot| slot.mods.iter().map(|item| item.model_tag))
                .unique()
                .collect::<Vec<_>>();
            if let Some(owner) =
                socket_index.owner_for(&self.cache, weapon.owner_tag, &modifications)
            {
                weapon.socket_owner = Some(owner);
            }
        }
    }

    fn refresh_runner_models(&mut self) {
        self.runner_models = runner_model_index(&self.cache, &self.weapon_catalog);
    }

    fn search_models(search: Option<String>) -> Vec<u16> {
        let pm = package_manager();
        let mut packages: Vec<(u16, PackagePath)> = package_manager()
            .package_paths
            .iter()
            .filter_map(|(id, path)| {
                let entries = pm.lookup.tag32_entries_by_pkg.get(id)?;
                entries
                    .iter()
                    .any(|entry| is_model_catalog_reference(entry.reference))
                    .then_some((*id, path.clone()))
            })
            .collect();

        if let Some(search) = search {
            let search = search.to_lowercase();
            packages.retain(|(_id, path)| {
                format!("{}_{}", path.name, path.id)
                    .to_lowercase()
                    .contains(&search)
            });
        }

        packages.sort_by_cached_key(|(_id, path)| format!("{}_{}", path.name, path.id));
        packages.into_iter().map(|(id, _path)| id).collect()
    }

    fn load_package_models(&mut self, id: u16) {
        let cache = self.cache.clone();
        let mut geometry_presence = rustc_hash::FxHashMap::default();
        self.models = package_manager()
            .lookup
            .tag32_entries_by_pkg
            .get(&id)
            .into_iter()
            .flat_map(|entries| entries.iter().enumerate())
            .filter_map(|(i, entry)| {
                if !is_model_catalog_reference(entry.reference) {
                    return None;
                }
                let info = model_info_for_reference(entry.reference)?;
                let tag = TagHash::new(id, i as u16);
                Some(ModelListEntry {
                    index: i,
                    tag,
                    info,
                    is_empty: !model_has_render_geometry(&cache, tag, &mut geometry_presence),
                    entry: entry.clone(),
                })
            })
            .collect();
        self.refresh_runner_models();
        self.selected_model = None;
        self.preview_channels = None;
        self.preview_runtime_inputs = None;
        self.channel_render_error = None;
        self.preview = None;
        self.gpu_model_preview = None;
        self.preview_camera_frame = None;
        self.weapon_export_camera = None;
        self.active_weapon = None;
        self.selected_mods.clear();
        self.selected_mod_unique_ids.clear();

        if let Some(tag) = self
            .models
            .iter()
            .find(|entry| !self.hide_empty_models || !entry.is_empty)
            .map(|entry| entry.tag)
        {
            self.load_model(tag);
        }
    }

    fn load_model(&mut self, tag: TagHash) {
        self.selected_model = Some(tag);
        self.preview_channels = None;
        self.preview_runtime_inputs = None;
        self.channel_render_error = None;
        self.preview_camera_frame = None;
        self.weapon_export_camera = None;
        self.detect_selected_weapon();
        if self.active_weapon.is_some() {
            self.preview_yaw = DEFAULT_WEAPON_YAW;
            self.preview_pitch = DEFAULT_WEAPON_PITCH;
        }
        self.rebuild_model_preview();
        self.rebuild_weapon_reference_frames();
    }

    fn detect_selected_weapon(&mut self) {
        self.active_weapon = self
            .selected_model
            .and_then(|selected| weapon_index_for_model(&self.weapon_catalog, selected));
        self.selected_mods = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
            .map(|weapon| vec![None; weapon.slots.len()])
            .unwrap_or_default();
        self.selected_mod_unique_ids = vec![None; self.selected_mods.len()];
    }

    fn selected_mod_tags(&self) -> Vec<Option<TagHash>> {
        let Some(weapon) = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
        else {
            return vec![];
        };
        weapon
            .slots
            .iter()
            .zip(&self.selected_mods)
            .map(|(slot, selected)| {
                selected
                    .and_then(|index| slot.mods.get(index))
                    .map(|item| item.model_tag)
            })
            .collect()
    }

    fn selected_mod_export_descriptors(&self) -> Vec<(TagHash, &'static str)> {
        let Some(weapon) = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
        else {
            return vec![];
        };
        weapon
            .slots
            .iter()
            .zip(&self.selected_mods)
            .filter_map(|(slot, selected)| {
                let item = selected.and_then(|index| slot.mods.get(index))?;
                Some((item.model_tag, item.rarity_code))
            })
            .collect()
    }

    fn selected_mod_attachments(&self) -> Vec<WeaponModPreviewAttachment> {
        let Some(weapon) = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
        else {
            return vec![];
        };
        let socket_owner = weapon.socket_owner.unwrap_or(weapon.owner_tag);
        let mut attachments = weapon
            .slots
            .iter()
            .zip(&self.selected_mods)
            .zip(&self.selected_mod_unique_ids)
            .filter_map(|((slot, selected), unique_id)| {
                let item = selected.and_then(|index| slot.mods.get(index))?;
                Some(WeaponModPreviewAttachment {
                    model_tag: item.model_tag,
                    rarity: item.preview_rarity,
                    unique_id: unique_id.unwrap_or(0.5),
                })
            })
            .collect::<Vec<_>>();
        let equipped_mods = attachments
            .iter()
            .map(|attachment| attachment.model_tag)
            .collect::<Vec<_>>();
        if self.preview_show_default_mods {
            attachments.extend(
                weapon_unoccupied_default_mod_patterns(
                    &self.cache,
                    weapon.owner_tag,
                    socket_owner,
                    &equipped_mods,
                )
                .into_iter()
                .map(|model_tag| WeaponModPreviewAttachment {
                    model_tag,
                    rarity: None,
                    unique_id: 0.5,
                }),
            );
        }
        attachments
    }

    fn rebuild_weapon_reference_frames(&mut self) {
        self.weapon_export_camera = None;
        self.preview_environment.light_model_frame = None;
        let Some(tag) = self.selected_model else {
            return;
        };
        let Some(entry) = package_manager().get_entry(tag) else {
            return;
        };
        let Some(weapon) = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
            .cloned()
        else {
            return;
        };
        let weapon_owner = weapon.socket_owner.unwrap_or(weapon.owner_tag);
        let default_attachments = weapon_unoccupied_default_mod_patterns(
            &self.cache,
            weapon.owner_tag,
            weapon_owner,
            &[],
        )
        .into_iter()
        .unique()
        .map(|model_tag| WeaponModPreviewAttachment {
            model_tag,
            rarity: None,
            unique_id: 0.5,
        })
        .collect::<Vec<_>>();
        let base = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            self.cache.clone(),
            tag,
            &entry,
            weapon.owner_tag,
            weapon_owner,
            &default_attachments,
        );
        let Some(base_wireframe) = base.as_ref().and_then(GeometryTagPreview::wireframe) else {
            return;
        };
        let vanilla = package_manager()
            .get_entry(weapon.owner_tag)
            .and_then(|entry| {
                GeometryTagPreview::load_model_with_weapon_mod_attachments(
                    self.cache.clone(),
                    weapon.owner_tag,
                    &entry,
                    weapon.owner_tag,
                    weapon_owner,
                    &default_attachments,
                )
            });
        self.preview_environment.light_model_frame = vanilla
            .as_ref()
            .and_then(GeometryTagPreview::wireframe)
            .map(ModelCameraFrame::from_wireframe);

        // Anchor to the authored default presentation, including empty-slot
        // barrel/stock/etc. meshes. Fit once against every authored attachment
        // this weapon can display; selected mod only changes pixels.
        let mut attachments = weapon
            .slots
            .iter()
            .flat_map(|slot| &slot.mods)
            .map(|item| WeaponModPreviewAttachment {
                model_tag: item.model_tag,
                rarity: item.preview_rarity,
                unique_id: 0.5,
            })
            .collect::<Vec<_>>();
        attachments.extend(default_attachments);
        attachments = attachments
            .into_iter()
            .unique_by(|attachment| attachment.model_tag)
            .collect();
        let envelope = GeometryTagPreview::load_model_with_weapon_mod_attachments(
            self.cache.clone(),
            tag,
            &entry,
            weapon.owner_tag,
            weapon_owner,
            &attachments,
        );
        let Some(envelope_wireframe) = envelope.as_ref().and_then(GeometryTagPreview::wireframe)
        else {
            return;
        };
        self.weapon_export_camera = Some(ModelExportCamera::from_wireframes(
            base_wireframe,
            envelope_wireframe,
            MODEL_EXPORT_HEIGHT as f32 / MODEL_EXPORT_WIDTH as f32,
            DEFAULT_WEAPON_YAW,
            DEFAULT_WEAPON_PITCH,
        ));
    }

    /// Same semantic as engine Pattern spawn: one random inclusive 0..1
    /// `unique_id` per attached object. Xorshift keeps this dependency-free and
    /// stable until user removes/replaces that mod instance.
    fn next_weapon_mod_unique_id(&mut self) -> f32 {
        let mut value = self.mod_unique_rng;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        if value == 0 {
            value = 0x9E37_79B9;
        }
        self.mod_unique_rng = value;
        (value as f64 / u32::MAX as f64) as f32
    }

    fn rebuild_model_preview(&mut self) {
        self.preview = None;
        self.gpu_model_preview = None;
        self.preview_runtime_inputs = None;
        self.channel_render_error = None;

        let Some(tag) = self.selected_model else {
            return;
        };
        let Some(entry) = package_manager().get_entry(tag) else {
            return;
        };
        let Ok(data) = package_manager().read_tag(tag) else {
            return;
        };

        let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
        let attachments = self.selected_mod_attachments();
        let active_weapon = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index));
        let weapon_pattern = active_weapon.map(|weapon| weapon.owner_tag).unwrap_or(tag);
        let weapon_owner = active_weapon
            .map(|weapon| weapon.socket_owner.unwrap_or(weapon.owner_tag))
            .unwrap_or(tag);
        self.preview = if let Some(composition) = self.runner_models.get(&tag) {
            composition.load(self.cache.clone())
        } else if model_info_for_reference(entry.reference).is_some() {
            GeometryTagPreview::load_model_with_weapon_mod_attachments(
                self.cache.clone(),
                tag,
                &entry,
                weapon_pattern,
                weapon_owner,
                &attachments,
            )
        } else {
            GeometryTagPreview::load(self.cache.clone(), tag, &entry, tag_type, &data)
        };
        if self.preview_camera_frame.is_none() {
            self.preview_camera_frame = self.preview.as_ref().and_then(|preview| {
                let GeometryPreviewKind::Model(model) = &preview.kind else {
                    return None;
                };
                model
                    .wireframe
                    .as_ref()
                    .map(ModelCameraFrame::from_wireframe)
            });
        }
        let runtime_inputs = TfxRuntimeInputs::for_model_preview(&self.cache, weapon_pattern);
        let previous_channels = self.preview_channels.take();
        self.preview_channels = self.preview.as_ref().and_then(|preview| {
            let GeometryPreviewKind::Model(model) = &preview.kind else { return None; };
            let mut channels = ModelChannels::discover(model.wireframe.as_ref()?, &runtime_inputs);
            if let Some(previous) = &previous_channels { channels.carry_overrides_from(previous); }
            Some(channels)
        });
        self.preview_runtime_inputs = Some(runtime_inputs);
        let mut evaluated_inputs = self.preview_runtime_inputs.clone();
        if let (Some(channels), Some(inputs)) = (&self.preview_channels, &mut evaluated_inputs) {
            channels.apply_overrides(inputs);
        }
        self.gpu_model_preview = self.preview.as_ref().and_then(|preview| {
            let GeometryPreviewKind::Model(model) = &preview.kind else {
                return None;
            };
            let wireframe = model.wireframe.as_ref()?;
            let fallback_color = wireframe_preview_textures(wireframe, &model.textures)
                .first()
                .copied();
            GpuModelPreview::create(
                &self.texture_cache.render_state.device,
                wireframe,
                fallback_color,
                evaluated_inputs.as_ref()?,
            )
            .map(Arc::new)
        });
    }

    fn apply_preview_channels(&mut self) {
        let (Some(channels), Some(base), Some(preview), Some(gpu)) = (
            &self.preview_channels, &self.preview_runtime_inputs, &self.preview, &self.gpu_model_preview,
        ) else { return; };
        let GeometryPreviewKind::Model(model) = &preview.kind else { return; };
        let Some(wireframe) = model.wireframe.as_ref() else { return; };
        let mut inputs = base.clone();
        channels.apply(&mut inputs);
        if let Some(updated) = gpu.with_channels(&self.texture_cache.render_state.device, wireframe, &inputs) {
            self.gpu_model_preview = Some(Arc::new(updated));
            self.channel_render_error = None;
        } else {
            self.channel_render_error = Some("Channel values could not be rendered. Previous frame retained.".into());
        }
    }
}

pub(super) fn weapon_index_for_model(
    catalog: &ModelWeaponCatalog,
    selected: TagHash,
) -> Option<usize> {
    let mut matches = catalog
        .weapons
        .iter()
        .enumerate()
        .filter(|(_, weapon)| weapon.model_tags.contains(&selected));
    let (index, _) = matches.next()?;
    matches.next().is_none().then_some(index)
}

fn weapon_skin_for_model(
    catalog: &ModelWeaponCatalog,
    selected: TagHash,
) -> Option<(&ModelWeaponEntry, &ModelWeaponSkinEntry)> {
    let mut matches = catalog.weapons.iter().flat_map(|weapon| {
        weapon
            .skins
            .iter()
            .filter(move |skin| skin.model_tag == selected)
            .map(move |skin| (weapon, skin))
    });
    let matched = matches.next()?;
    matches.next().is_none().then_some(matched)
}

fn melee_skins_for_model(
    catalog: &ModelWeaponCatalog,
    selected: TagHash,
) -> Vec<&ModelMeleeSkinEntry> {
    catalog
        .melee_skins
        .iter()
        .filter(|skin| skin.model_tag == selected)
        .collect()
}

fn charms_for_model(catalog: &ModelWeaponCatalog, selected: TagHash) -> Vec<&ModelCharmEntry> {
    catalog
        .charms
        .iter()
        .filter(|charm| charm.model_tag == selected)
        .collect()
}

fn runner_skin_for_model<'a>(
    catalog: &'a ModelWeaponCatalog,
    selected: TagHash,
) -> Option<&'a ModelRunnerSkinEntry> {
    let mut matches = catalog
        .runner_skins
        .iter()
        .filter(|skin| skin.model_tag == selected);
    let matched = matches.next()?;
    matches.next().is_none().then_some(matched)
}

fn runner_model_index(
    cache: &TagCache,
    catalog: &ModelWeaponCatalog,
) -> rustc_hash::FxHashMap<TagHash, RunnerShellAssembly> {
    catalog
        .runner_skins
        .iter()
        .filter_map(|skin| {
            let assembly = RunnerShellAssembly::resolve(cache, skin.model_tag)?;
            Some((assembly.pattern, assembly))
        })
        .collect()
}

fn runner_model_root(
    models: &rustc_hash::FxHashMap<TagHash, RunnerShellAssembly>,
    tag: TagHash,
) -> Option<TagHash> {
    if models.contains_key(&tag) {
        return Some(tag);
    }
    let mut owners = models.iter().filter(|(_, model)| {
        model.nested_patterns.contains(&tag) || model.parts.iter().any(|part| part.component == tag)
    });
    let root = *owners.next()?.0;
    owners.next().is_none().then_some(root)
}

impl View for ModelsView {
    fn view(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) -> Option<ViewAction> {
        let mut action = None;

        egui::SidePanel::left("models_left_panel")
            .resizable(true)
            .min_width(256.0)
            .show_inside(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);

                ui.horizontal(|ui| {
                    ui.label("Search:");
                    if ui.text_edit_singleline(&mut self.package_filter).changed() {
                        self.packages_with_models = Self::search_models(
                            (!self.package_filter.is_empty()).then(|| self.package_filter.clone()),
                        );
                    }
                });

                egui::ScrollArea::vertical()
                    .id_salt("models_package_list")
                    .max_width(f32::INFINITY)
                    .show(ui, |ui| {
                        let packages = self.packages_with_models.clone();
                        for id in packages {
                            let path = &package_manager().package_paths[&id];
                            let package_name = format!("{}_{}", path.name, path.id);
                            if ui
                                .selectable_value(
                                    &mut self.selected_package,
                                    id,
                                    format!("{id:04x}: {package_name}"),
                                )
                                .changed()
                            {
                                self.load_package_models(id);
                            }
                        }
                    });
            });

        if self.selected_package == u16::MAX {
            egui::CentralPanel::default().show_inside(ui, |ui| {
                ui.label(RichText::new("No package selected").italics());
            });
            return action;
        }

        egui::SidePanel::left("models_entry_panel")
            .resizable(true)
            .min_width(320.0)
            .default_width(420.0)
            .show_inside(ui, |ui| {
                if let Some(model_action) = self.model_list_ui(ui) {
                    action = Some(model_action);
                }
            });

        let selected_model = self.selected_model;
        let preview = self.preview.as_ref();
        let texture_cache = &self.texture_cache;
        let gpu_model_preview = self.gpu_model_preview.as_ref();
        let preview_camera_frame = self.preview_camera_frame;
        let weapon_export_camera = self.weapon_export_camera;
        let selected_mod_export_descriptors = self.selected_mod_export_descriptors();
        let preview_yaw = &mut self.preview_yaw;
        let preview_pitch = &mut self.preview_pitch;
        let preview_zoom = &mut self.preview_zoom;
        let preview_pan = &mut self.preview_pan;
        let preview_show_wireframe = &mut self.preview_show_wireframe;
        let preview_show_stickers = &mut self.preview_show_stickers;
        let default_mods_before = self.preview_show_default_mods;
        let preview_show_default_mods = &mut self.preview_show_default_mods;
        let preview_environment = &mut self.preview_environment;
        let channel_revision = self.preview_channels.as_ref().map(ModelChannels::revision);
        let preview_channels = &mut self.preview_channels;
        let channel_render_error = self.channel_render_error.as_deref();
        let export_all_mods = &mut self.export_all_mods;
        let export_format = &mut self.export_format;
        let active_weapon = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
            .cloned();
        let selected_mods = self.selected_mods.clone();
        let modded_export_active = self.modded_export.is_some();
        let mut mod_selection = None;
        let mut export_result = None;
        let mut modded_export_request = None;

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(model) = preview.and_then(|preview| match &preview.kind {
                GeometryPreviewKind::Model(model) => Some(model),
                _ => None,
            }) else {
                ui.label(RichText::new("No model selected").italics());
                return;
            };

            ui.horizontal(|ui| {
                ui.heading(model.label);
                if let Some(tag) = selected_model
                    && ui.button("Open tag").clicked()
                {
                    action = Some(ViewAction::OpenTag(tag));
                }
            });
            ui.horizontal_wrapped(|ui| {
                if let Some(class_name) = &model.class_name {
                    ui.label(format!("Class: {class_name}"));
                }
                ui.label(format!("VB: {}", model.vertex_buffers.len()));
                ui.label(format!("IB: {}", model.index_buffers.len()));
                ui.label(format!("Techniques: {}", model.techniques.len()));
                ui.label(format!("Textures: {}", model.textures.len()));
                ui.label(format!("Shaders: {}", model.shaders.len()));
                ui.label(format!("Parts: {}", model.geometry_parts.len()));
            });

            ui.separator();
            if let Some(wireframe) = &model.wireframe {
                let (viewport, texture_action) = model_wireframe_ui(
                    ui,
                    wireframe,
                    model.preview_uv_transform(),
                    texture_cache,
                    &model.textures,
                    gpu_model_preview,
                    preview_camera_frame,
                    preview_yaw,
                    preview_pitch,
                    preview_zoom,
                    preview_pan,
                    preview_show_wireframe,
                    preview_show_stickers,
                    active_weapon.is_some().then_some(preview_show_default_mods),
                    preview_environment,
                    true,
                    preview_channels.as_mut(),
                );
                if let Some(error) = channel_render_error {
                    ui.colored_label(Color32::YELLOW, error);
                }
                if texture_action.is_some() {
                    action = texture_action;
                }

                if model_export_toolbar(
                    ui.ctx(),
                    viewport,
                    export_all_mods,
                    export_format,
                    active_weapon.is_some(),
                    modded_export_active,
                ) {
                    let format = *export_format;
                    if *export_all_mods {
                        if let (Some(tag), Some(weapon)) = (selected_model, active_weapon.as_ref())
                        {
                            let filename = model_modded_export_filename(tag);
                            match native_dialog::FileDialog::new()
                                .add_filter("ZIP archive", &["zip"])
                                .set_filename(&filename)
                                .show_save_single_file()
                            {
                                Ok(Some(mut path)) => {
                                    if !path.extension().is_some_and(|extension| {
                                        extension.eq_ignore_ascii_case("zip")
                                    }) {
                                        path.set_extension("zip");
                                    }
                                    modded_export_request = Some((
                                        tag,
                                        weapon.clone(),
                                        *preview_show_default_mods,
                                        *preview_show_stickers,
                                        *preview_environment,
                                        weapon_export_camera,
                                        format,
                                        path,
                                    ));
                                }
                                Ok(None) => {}
                                Err(error) => export_result = Some(Err(error.into())),
                            }
                        }
                    } else if let Some(tag) = selected_model {
                        let filename =
                            model_export_filename(tag, &selected_mod_export_descriptors, format);
                        match native_dialog::FileDialog::new()
                            .add_filter(format.dialog_label(), &[format.extension()])
                            .set_filename(&filename)
                            .show_save_single_file()
                        {
                            Ok(Some(mut path)) => {
                                if !path.extension().is_some_and(|extension| {
                                    extension.eq_ignore_ascii_case(format.extension())
                                }) {
                                    path.set_extension(format.extension());
                                }
                                if let Some(gpu_preview) = gpu_model_preview {
                                    let export_rect = egui::Rect::from_min_size(
                                        egui::Pos2::ZERO,
                                        vec2(MODEL_EXPORT_WIDTH as f32, MODEL_EXPORT_HEIGHT as f32),
                                    );
                                    let callback = ModelPaintCallback::new(
                                        gpu_preview.clone(),
                                        texture_cache,
                                        wireframe,
                                        model.preview_uv_transform(),
                                        None,
                                        DEFAULT_WEAPON_YAW,
                                        DEFAULT_WEAPON_PITCH,
                                        1.0,
                                        egui::Vec2::ZERO,
                                        *preview_show_stickers,
                                        export_rect,
                                        1.0,
                                        *preview_environment,
                                    )
                                    .with_export_camera(weapon_export_camera);
                                    export_result = Some(
                                        callback
                                            .export_image(
                                                &texture_cache.render_state,
                                                &path,
                                                [MODEL_EXPORT_WIDTH, MODEL_EXPORT_HEIGHT],
                                                format.image_format(),
                                            )
                                            .map(|()| path),
                                    );
                                } else {
                                    export_result = Some(Err(anyhow::anyhow!(
                                        "GPU model preview is unavailable"
                                    )));
                                }
                            }
                            Ok(None) => {}
                            Err(error) => export_result = Some(Err(error.into())),
                        }
                    }
                }

                if let (Some(tag), Some(weapon)) = (selected_model, active_weapon.as_ref()) {
                    mod_selection = compact_weapon_mod_selector(
                        ui.ctx(),
                        viewport,
                        tag,
                        weapon,
                        &selected_mods,
                    );
                }
            } else {
                ui.label(RichText::new("No wireframe assembled yet").italics());
            }
        });

        let mut rebuild_preview = *preview_show_default_mods != default_mods_before;
        if let Some((slot, selection)) = mod_selection {
            let unique_id = selection.map(|_| self.next_weapon_mod_unique_id());
            if let Some(selected) = self.selected_mods.get_mut(slot) {
                *selected = selection;
                if let Some(seed) = self.selected_mod_unique_ids.get_mut(slot) {
                    *seed = unique_id;
                }
                rebuild_preview = true;
            }
        }
        if rebuild_preview {
            self.rebuild_model_preview();
        } else if self.preview_channels.as_ref().map(ModelChannels::revision) != channel_revision {
            self.apply_preview_channels();
            ui.ctx().request_repaint();
        }
        if let Some(result) = export_result {
            match result {
                Ok(path) => {
                    TOASTS
                        .lock()
                        .success(format!("Export saved to {}", path.display()));
                }
                Err(error) => {
                    TOASTS.lock().error(format!("Export failed: {error:#}"));
                }
            }
        }
        if let Some((tag, weapon, defaults, stickers, environment, camera, format, path)) =
            modded_export_request
        {
            match ModdedExportJob::new(
                tag,
                weapon,
                defaults,
                stickers,
                environment,
                self.preview_channels.clone(),
                camera,
                format,
                path,
            ) {
                Ok(job) => self.modded_export = Some(job),
                Err(error) => {
                    TOASTS.lock().error(format!("Export failed: {error:#}"));
                }
            }
        }
        if let Some(mut job) = self.modded_export.take() {
            let path = job.path.clone();
            let total = job.progress().1;
            let format = job.format;
            match job.step(&self.cache, &self.texture_cache) {
                Ok(true) => {
                    TOASTS.lock().success(format!(
                        "Saved {total} {} images to {}",
                        format.label(),
                        path.display()
                    ));
                }
                Ok(false) => {
                    self.modded_export = Some(job);
                    ui.ctx().request_repaint();
                }
                Err(error) => {
                    drop(job);
                    let _ = std::fs::remove_file(path);
                    TOASTS.lock().error(format!("Export failed: {error:#}"));
                }
            }
        }

        action
    }
}

impl ModelsView {
    fn model_list_ui(&mut self, ui: &mut egui::Ui) -> Option<ViewAction> {
        let mut action = None;

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.model_filter);
        });
        ui.checkbox(&mut self.hide_empty_models, "Hide empty");

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("models_entry_list")
            .auto_shrink([false, false])
            .max_width(f32::INFINITY)
            .show(ui, |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);

                if self.models.is_empty() {
                    ui.label(RichText::new("No model tags found in package").italics());
                    return;
                }

                let filter = self.model_filter.to_lowercase();
                let models = self.models.clone();
                let mut selected = None;
                for entry in models {
                    if self.hide_empty_models && entry.is_empty {
                        continue;
                    }
                    let detected_skin = weapon_skin_for_model(&self.weapon_catalog, entry.tag);
                    if !self.runner_models.contains_key(&entry.tag)
                        && self
                            .runner_models
                            .values()
                            .any(|model| model.nested_patterns.contains(&entry.tag))
                    {
                        continue;
                    }
                    let runner_skin = runner_skin_for_model(&self.weapon_catalog, entry.tag);
                    let melee_skins = melee_skins_for_model(&self.weapon_catalog, entry.tag);
                    let charms = charms_for_model(&self.weapon_catalog, entry.tag);
                    if !filter.is_empty()
                        && !entry
                            .search_label(detected_skin, runner_skin, &melee_skins, &charms)
                            .contains(filter.as_str())
                    {
                        continue;
                    }

                    let response = ui
                        .add(
                            egui::Button::selectable(
                                self.selected_model == Some(entry.tag),
                                entry.list_label(
                                    ui,
                                    detected_skin,
                                    runner_skin,
                                    &melee_skins,
                                    &charms,
                                ),
                            )
                            .wrap_mode(
                                if detected_skin.is_some()
                                    || runner_skin.is_some()
                                    || !melee_skins.is_empty()
                                    || !charms.is_empty()
                                {
                                    egui::TextWrapMode::Wrap
                                } else {
                                    egui::TextWrapMode::Truncate
                                },
                            ),
                        )
                        .tag_context(entry.tag);

                    if response.clicked() {
                        selected = Some(entry.tag);
                    }
                    if response.double_clicked() {
                        action = Some(ViewAction::OpenTag(entry.tag));
                    }
                }

                if let Some(tag) = selected {
                    self.load_model(tag);
                }
            });

        action
    }
}

#[derive(Clone)]
struct ModelListEntry {
    index: usize,
    tag: TagHash,
    info: ModelTagInfo,
    is_empty: bool,
    entry: UEntryHeader,
}

impl ModelListEntry {
    fn label(&self) -> String {
        format!(
            "{}: {} {} ({})",
            self.index,
            self.info.label,
            self.tag,
            format_file_size(self.entry.file_size as usize)
        )
    }

    fn search_label(
        &self,
        detected_skin: Option<(&ModelWeaponEntry, &ModelWeaponSkinEntry)>,
        runner_skin: Option<&ModelRunnerSkinEntry>,
        melee_skins: &[&ModelMeleeSkinEntry],
        charms: &[&ModelCharmEntry],
    ) -> String {
        let identity = detected_skin
            .map(|(weapon, skin)| {
                format!(
                    " {} {} {}",
                    weapon.name,
                    skin.name,
                    skin.rarity.as_deref().unwrap_or_default()
                )
            })
            .unwrap_or_default();
        let runner_identity = runner_skin
            .map(|skin| format!(" {} {}", skin.shell_name, skin.name,))
            .unwrap_or_default();
        let melee_identity = melee_skins
            .iter()
            .map(|skin| format!(" {} {}", skin.family_name, skin.name))
            .collect::<String>();
        let charm_identity = charms
            .iter()
            .map(|charm| format!(" Charm {}", charm.name))
            .collect::<String>();
        format!(
            "{} {}{identity}{runner_identity}{melee_identity}{charm_identity}",
            self.label(),
            self.info.label
        )
        .to_lowercase()
    }

    fn list_label(
        &self,
        ui: &egui::Ui,
        detected_skin: Option<(&ModelWeaponEntry, &ModelWeaponSkinEntry)>,
        runner_skin: Option<&ModelRunnerSkinEntry>,
        melee_skins: &[&ModelMeleeSkinEntry],
        charms: &[&ModelCharmEntry],
    ) -> egui::text::LayoutJob {
        let mut label = egui::text::LayoutJob::default();
        label.append(
            &self.label(),
            0.0,
            egui::TextFormat {
                font_id: egui::TextStyle::Monospace.resolve(ui.style()),
                color: self.role_color(),
                ..Default::default()
            },
        );
        if let Some((weapon, skin)) = detected_skin {
            label.append(
                "\n    └ ",
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Small.resolve(ui.style()),
                    color: Color32::GRAY,
                    ..Default::default()
                },
            );
            label.append(
                &format!("{}: {}", weapon.name, skin.name),
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Small.resolve(ui.style()),
                    color: skin.color,
                    ..Default::default()
                },
            );
        }
        if let Some(skin) = runner_skin {
            label.append(
                &format!("\n    └ {}: {}", skin.shell_name, skin.name,),
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Small.resolve(ui.style()),
                    color: skin.color,
                    ..Default::default()
                },
            );
        }
        for skin in melee_skins {
            label.append(
                &format!("\n    └ {}: {}", skin.family_name, skin.name),
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Small.resolve(ui.style()),
                    color: skin.color,
                    ..Default::default()
                },
            );
        }
        for charm in charms {
            label.append(
                &format!("\n    └ Charm: {}", charm.name),
                0.0,
                egui::TextFormat {
                    font_id: egui::TextStyle::Small.resolve(ui.style()),
                    color: charm.color,
                    ..Default::default()
                },
            );
        }
        label
    }

    fn role_color(&self) -> Color32 {
        match self.info.role {
            ModelTagRole::Mesh => Color32::LIGHT_BLUE,
            ModelTagRole::MeshData => Color32::from_rgb(130, 210, 255),
            ModelTagRole::Geometry => Color32::LIGHT_YELLOW,
            ModelTagRole::Dynamic => Color32::from_rgb(160, 210, 255),
            ModelTagRole::Container => Color32::from_rgb(220, 190, 255),
        }
    }
}

fn compact_weapon_mod_selector(
    ctx: &egui::Context,
    viewport: egui::Rect,
    selected_model: TagHash,
    weapon: &ModelWeaponEntry,
    selections: &[Option<usize>],
) -> Option<(usize, Option<usize>)> {
    if viewport.width() < 220.0 || viewport.height() < 180.0 {
        return None;
    }

    let width = (viewport.width() - MODEL_OVERLAY_INSET * 2.0).min(310.0);
    let mut changed = None;

    egui::Area::new(egui::Id::new(("weapon_mod_selector", selected_model.0)))
        .order(egui::Order::Foreground)
        .fixed_pos(viewport.right_bottom() + vec2(-MODEL_OVERLAY_INSET, -MODEL_OVERLAY_INSET))
        .pivot(egui::Align2::RIGHT_BOTTOM)
        .show(ctx, |ui| {
            model_overlay_frame().show(ui, |ui| {
                ui.set_width(width - MODEL_OVERLAY_INSET * 2.0);
                ui.horizontal(|ui| {
                    ui.strong("Mods");
                    ui.add_space(4.0);
                    ui.label(RichText::new(&weapon.name).small().color(Color32::GRAY));
                });
                ui.separator();

                for (slot_index, slot) in weapon.slots.iter().enumerate() {
                    let selected_index = selections.get(slot_index).copied().flatten();
                    let selected = selected_index.and_then(|index| slot.mods.get(index));
                    let accent = selected
                        .map(|item| item.color)
                        .unwrap_or(Color32::from_rgb(90, 98, 110));
                    ui.label(RichText::new(&slot.name).small().strong());
                    ui.horizontal(|ui| {
                        let button_width = if selected.is_some() {
                            ui.available_width() - 30.0
                        } else {
                            ui.available_width()
                        };
                        egui::Frame::default()
                            .fill(accent.gamma_multiply(0.22))
                            .stroke(Stroke::new(1.0, accent))
                            .corner_radius(4)
                            .inner_margin(2)
                            .show(ui, |ui| {
                                ui.set_width(button_width.max(120.0));
                                ui.menu_button(
                                    RichText::new(
                                        selected
                                            .map(|item| item.name.as_str())
                                            .unwrap_or("Select mod…"),
                                    )
                                    .color(
                                        if selected.is_some() {
                                            Color32::WHITE
                                        } else {
                                            Color32::GRAY
                                        },
                                    ),
                                    |ui| {
                                        ui.set_min_width((width - 36.0).max(180.0));
                                        if ui
                                            .add_sized(
                                                [ui.available_width(), 28.0],
                                                egui::Button::new("None"),
                                            )
                                            .clicked()
                                        {
                                            changed = Some((slot_index, None));
                                            ui.close();
                                        }
                                        ui.separator();
                                        for (mod_index, modification) in
                                            slot.mods.iter().enumerate()
                                        {
                                            if mod_option_button(ui, modification).clicked() {
                                                changed = Some((
                                                    slot_index,
                                                    (selected_index != Some(mod_index))
                                                        .then_some(mod_index),
                                                ));
                                                ui.close();
                                            }
                                        }
                                    },
                                );
                            });
                        if selected.is_some()
                            && ui
                                .add_sized([26.0, 26.0], egui::Button::new("×"))
                                .on_hover_text("Remove mod")
                                .clicked()
                        {
                            changed = Some((slot_index, None));
                        }
                    });
                }
            });
        });

    changed
}

fn mod_option_button(ui: &mut egui::Ui, modification: &ModelModEntry) -> egui::Response {
    ui.add_sized(
        [ui.available_width(), 30.0],
        egui::Button::new(format!("{}  ·  {}", modification.name, modification.rarity))
            .fill(modification.color.gamma_multiply(0.24))
            .stroke(Stroke::new(1.0, modification.color)),
    )
}

fn normalize_direction(direction: &mut [f32; 3], fallback: [f32; 3]) {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length <= 0.0001 {
        *direction = fallback;
    } else {
        direction.iter_mut().for_each(|value| *value /= length);
    }
}

fn render_evidence_panel(ui: &mut egui::Ui, gpu_preview: &GpuModelPreview) {
    ui.set_min_width(440.0);
    egui::ScrollArea::vertical()
        .max_height(520.0)
        .show(ui, |ui| {
            for line in gpu_preview.inspection_lines() {
                ui.monospace(line);
            }
        });
}

fn lighting_panel(ui: &mut egui::Ui, environment: &mut ModelEnvironment) {
    ui.set_min_width(440.0);
    egui::ScrollArea::vertical()
        .max_height(560.0)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Model");
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::TigerGgxApproximation,
                    "Tiger GGX",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::TigerGgxCompatibility,
                    "Compatibility",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::DebugLambert,
                    "Debug Lambert",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::SurfaceNormals,
                    "Normals",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::SurfaceProperties,
                    "M/R/AO",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::SurfaceEmissive,
                    "Emissive",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::SurfaceFlags,
                    "Flags",
                );
                ui.selectable_value(
                    &mut environment.lighting_model,
                    LightingModel::SurfaceAlbedo,
                    "Albedo MRT",
                );
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut environment.light_gizmo, "Spotlight gizmo");
                if ui.button("Reset lighting").clicked() {
                    let defaults = ModelEnvironment::default();
                    environment.light_target = defaults.light_target;
                    environment.light_orbit_position = defaults.light_orbit_position;
                    environment.light_orbit_center = defaults.light_orbit_center;
                    environment.light_orbit_radius = defaults.light_orbit_radius;
                    environment.light_range = defaults.light_range;
                    environment.light_cone_angle = defaults.light_cone_angle;
                    environment.light_size = defaults.light_size;
                    environment.light_scale_with_model = defaults.light_scale_with_model;
                    environment.shadow_strength = defaults.shadow_strength;
                    environment.sun_intensity = defaults.sun_intensity;
                    environment.ambient_intensity = defaults.ambient_intensity;
                    environment.specular_ibl_intensity = defaults.specular_ibl_intensity;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Material channel");
                let channel_name = match environment.diagnostic_pass {
                    0 => "Final",
                    1 => "Albedo",
                    2 => "Diffuse",
                    3 => "Ambient Occlusion",
                    4 => "Specular",
                    5 => "Pre-tone HDR",
                    6 => "Normal",
                    7 => "Emission",
                    8 => "Flags",
                    9 => "Dye",
                    10 => "Worn Dye",
                    11 => "Dye Detail",
                    12 => "Roughness",
                    13 => "Smoothness",
                    14 => "Emission Intensity",
                    15 => "Transparency",
                    16 => "Metalness",
                    17 => "Transmission",
                    18 => "Iridescence ID",
                    19 => "Dye Mask",
                    20 => "Wear Mask",
                    21 => "Coating Face Color",
                    22 => "Coating Grazing Color",
                    23 => "Coating Incidence",
                    24 => "Coating Coverage",
                    25 => "Coating Detail Response",
                    26 => "Coating Sharp Specular",
                    27 => "Coating Broad Specular",
                    28 => "Coating Environment",
                    29 => "Coating Premultiplied Output",
                    _ => "Unknown",
                };
                egui::ComboBox::from_id_salt("model_material_channel")
                    .selected_text(channel_name)
                    .show_ui(ui, |ui| {
                        for (value, label) in [
                            (0, "Final"),
                            (1, "Albedo"),
                            (9, "Dye"),
                            (10, "Worn Dye"),
                            (11, "Dye Detail"),
                            (19, "Dye Mask"),
                            (20, "Wear Mask"),
                            (21, "Coating Face Color"),
                            (22, "Coating Grazing Color"),
                            (23, "Coating Incidence"),
                            (24, "Coating Coverage"),
                            (25, "Coating Detail Response"),
                            (26, "Coating Sharp Specular"),
                            (27, "Coating Broad Specular"),
                            (28, "Coating Environment"),
                            (29, "Coating Premultiplied Output"),
                            (3, "Ambient Occlusion"),
                            (12, "Roughness"),
                            (13, "Smoothness"),
                            (7, "Emission"),
                            (14, "Emission Intensity"),
                            (15, "Transparency"),
                            (16, "Metalness"),
                            (17, "Transmission"),
                            (18, "Iridescence ID"),
                            (6, "Normal"),
                            (8, "Flags"),
                            (2, "Diffuse"),
                            (4, "Specular"),
                            (5, "Pre-tone HDR"),
                        ] {
                            ui.selectable_value(&mut environment.diagnostic_pass, value, label);
                        }
                    });
            });
            ui.label("Spotlight beam target (world space)");
            for axis in 0..3 {
                ui.add(
                    egui::Slider::new(
                        &mut environment.light_target[axis],
                        MODEL_PARAMETER_RANGE.clone(),
                    )
                    .text(["Target X", "Target Y", "Target Z"][axis]),
                );
            }
            ui.separator();
            ui.label("Spotlight source orbit");
            ui.label("Point on sphere");
            for axis in 0..3 {
                ui.add(
                    egui::Slider::new(&mut environment.light_orbit_position[axis], -1.0..=1.0)
                        .text(["Point X", "Point Y", "Point Z"][axis]),
                );
            }
            ui.label("Sphere center");
            for axis in 0..3 {
                ui.add(
                    egui::Slider::new(
                        &mut environment.light_orbit_center[axis],
                        MODEL_PARAMETER_RANGE.clone(),
                    )
                    .text(["Center X", "Center Y", "Center Z"][axis]),
                );
            }
            ui.add(
                egui::Slider::new(&mut environment.light_orbit_radius, 0.25..=10.0)
                    .text("Source distance"),
            );
            ui.checkbox(&mut environment.light_scale_with_model, "Scale lighting with model")
                .on_hover_text("Fit the rig to the base model with default mods. Barrel swaps keep the same lighting. Distances use model-size units; brightness stays consistent.");
            ui.add(
                egui::Slider::new(&mut environment.light_range, 0.25..=20.0)
                    .text("Beam range"),
            );
            ui.add(
                egui::Slider::new(&mut environment.light_cone_angle, 1.0..=89.0)
                    .text("Beam half-angle"),
            );
            ui.add(
                egui::Slider::new(&mut environment.light_size, 0.0..=10.0)
                    .text("Light size"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.sun_intensity,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Key light"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.shadow_strength,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Shadows"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.ambient_intensity,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Ambient light"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.specular_ibl_intensity,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Reflections"),
            );
        });
}

fn image_panel(ui: &mut egui::Ui, environment: &mut ModelEnvironment) {
    ui.set_min_width(360.0);

    ui.strong("TFX runtime");
    ui.horizontal(|ui| {
        ui.checkbox(&mut environment.tfx_paused, "Paused");
        if ui.button("Reset time").clicked() {
            environment.tfx_time_seconds = 0.0;
        }
    });
    ui.add(egui::Slider::new(&mut environment.tfx_time_seconds, 0.0..=120.0).text("Time (s)"));
    ui.add(egui::Slider::new(&mut environment.tfx_speed, 0.0..=4.0).text("Speed"));

    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("Image");
        if ui.button("Reset image").clicked() {
            let defaults = ModelEnvironment::default();
            environment.brightness = defaults.brightness;
            environment.contrast = defaults.contrast;
            environment.saturation = defaults.saturation;
            environment.gamma = defaults.gamma;
            environment.exposure = defaults.exposure;
            environment.bloom_strength = defaults.bloom_strength;
        }
    });
    ui.add(
        egui::Slider::new(&mut environment.brightness, MODEL_PARAMETER_RANGE.clone())
            .text("Brightness"),
    );
    ui.add(
        egui::Slider::new(&mut environment.contrast, MODEL_PARAMETER_RANGE.clone())
            .text("Contrast"),
    );
    ui.add(
        egui::Slider::new(&mut environment.saturation, MODEL_PARAMETER_RANGE.clone())
            .text("Saturation"),
    );
    ui.add(egui::Slider::new(&mut environment.gamma, MODEL_PARAMETER_RANGE.clone()).text("Gamma"));
    ui.add(
        egui::Slider::new(&mut environment.exposure, MODEL_PARAMETER_RANGE.clone()).text(
            if environment.auto_exposure {
                "Exposure compensation"
            } else {
                "Exposure"
            },
        ),
    );
    ui.checkbox(&mut environment.auto_exposure, "Autoexposure");
    ui.add(
        egui::Slider::new(
            &mut environment.bloom_strength,
            MODEL_PARAMETER_RANGE.clone(),
        )
        .text("Bloom"),
    );
    ui.checkbox(&mut environment.tone_mapping, "Filmic tone mapping");

    ui.separator();
    ui.strong("Environment & effects");
    ui.add(
        egui::Slider::new(
            &mut environment.vertex_ao_strength,
            MODEL_PARAMETER_RANGE.clone(),
        )
        .text("Vertex AO"),
    );
    ui.checkbox(&mut environment.hiz_culling, "HiZ culling");
    ui.checkbox(&mut environment.fxaa, "FXAA");
    ui.add(
        egui::Slider::new(
            &mut environment.ssao_strength,
            MODEL_PARAMETER_RANGE.clone(),
        )
        .text("SSAO"),
    );
}

fn model_viewport_toolbar(
    ctx: &egui::Context,
    rect: egui::Rect,
    texture_cache: &TextureCache,
    textures: &[(TagHash, UEntryHeader)],
    gpu_preview: Option<&Arc<GpuModelPreview>>,
    environment: &mut ModelEnvironment,
    show_textures: bool,
    channels: Option<&mut ModelChannels>,
) -> Option<ViewAction> {
    let mut action = None;
    egui::Area::new(egui::Id::new("model_viewport_toolbar"))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.right_top() + vec2(-MODEL_OVERLAY_INSET, MODEL_OVERLAY_INSET))
        .pivot(egui::Align2::RIGHT_TOP)
        .show(ctx, |ui| {
            model_overlay_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    let lighting = ui.button("Lighting");
                    egui::Popup::from_toggle_button_response(&lighting)
                        .align(egui::RectAlign::BOTTOM_END)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| lighting_panel(ui, environment));

                    let channel = ui.add_enabled(channels.is_some(), egui::Button::new("Channel"));
                    egui::Popup::from_toggle_button_response(&channel)
                        .align(egui::RectAlign::BOTTOM_END)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                            if let Some(channels) = channels {
                                model_channel_panel(ui, channels);
                            }
                        });

                    let image = ui.button("Image");
                    egui::Popup::from_toggle_button_response(&image)
                        .align(egui::RectAlign::BOTTOM_END)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| image_panel(ui, environment));

                    if show_textures {
                        let textures_button = ui.button("Textures");
                        egui::Popup::from_toggle_button_response(&textures_button)
                            .align(egui::RectAlign::BOTTOM_END)
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                            .show(|ui| {
                                ui.set_min_width(500.0);
                                egui::ScrollArea::vertical()
                                    .max_height(560.0)
                                    .show(ui, |ui| {
                                        if let Some(texture_action) =
                                            model_textures_ui(ui, texture_cache, textures)
                                        {
                                            action = Some(texture_action);
                                        }
                                    });
                            });
                    }

                    let evidence =
                        ui.add_enabled(gpu_preview.is_some(), egui::Button::new("Render evidence"));
                    egui::Popup::from_toggle_button_response(&evidence)
                        .align(egui::RectAlign::BOTTOM_END)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| {
                            if let Some(gpu_preview) = gpu_preview {
                                render_evidence_panel(ui, gpu_preview);
                            }
                        });
                });
            });
        });
    action
}

fn model_channel_panel(ui: &mut egui::Ui, channels: &mut ModelChannels) {
    ui.set_min_width(490.0);
    ui.label("Used channels · edits apply live");
    ui.horizontal(|ui| {
        if ui.button("Reset channels").clicked() { channels.reset(); }
    });
    if channels.rows().is_empty() {
        ui.label("This model reads no channels.");
        return;
    }
    let mut changed = false;
    egui::ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
        for row in channels.rows_mut() {
            ui.push_id(row.key(), |ui| {
                let name = row.hash.and_then(super::get_string_for_hash)
                    .unwrap_or_else(|| row.hash.map_or_else(|| format!("Global {}", row.id), |hash| format!("unk_{hash:08X}")));
                let label = row.hash.map_or(name.clone(), |hash| format!("{name} (0x{hash:08X})"));
                ui.label(label).on_hover_text(format!("{:?} channel · binding {}", row.domain, row.id));
                ui.horizontal(|ui| {
                    for (lane, label) in ["X", "Y", "Z", "W"].iter().enumerate() {
                        ui.label(*label);
                        let response = ui.add(egui::TextEdit::singleline(&mut row.buffers[lane])
                            .desired_width(75.0).hint_text("Unknown"));
                        changed |= response.changed();
                        if let Some(error) = &row.errors[lane] { response.on_hover_text(error); }
                    }
                });
                if row.errors.iter().any(Option::is_some) {
                    ui.colored_label(Color32::YELLOW, "Enter four finite numbers.");
                } else if row.default_value.is_none() {
                    ui.weak("No single default value in this model's scopes.");
                }
                ui.separator();
            });
        }
    });
    if changed { let _ = channels.commit(); }
}

fn model_view_options_toolbar(
    ctx: &egui::Context,
    rect: egui::Rect,
    show_wireframe: &mut bool,
    show_stickers: &mut bool,
    show_default_mods: Option<&mut bool>,
    yaw: &mut f32,
    pitch: &mut f32,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
) {
    let is_weapon = show_default_mods.is_some();
    egui::Area::new(egui::Id::new("model_view_options_toolbar"))
        .order(egui::Order::Foreground)
        .fixed_pos(
            rect.right_top()
                + vec2(
                    -MODEL_OVERLAY_INSET,
                    MODEL_OVERLAY_INSET + MODEL_OVERLAY_STACK_STEP,
                ),
        )
        .pivot(egui::Align2::RIGHT_TOP)
        .show(ctx, |ui| {
            model_overlay_frame().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(show_wireframe, "Wireframe");
                    ui.checkbox(show_stickers, "Stickers");
                    if let Some(show_default_mods) = show_default_mods {
                        ui.checkbox(show_default_mods, "Default mods")
                            .on_hover_text("Show authored empty-slot weapon meshes");
                    }
                    if ui.button("Reset view").clicked() {
                        *yaw = if is_weapon {
                            DEFAULT_WEAPON_YAW
                        } else {
                            DEFAULT_MODEL_YAW
                        };
                        *pitch = if is_weapon {
                            DEFAULT_WEAPON_PITCH
                        } else {
                            0.05
                        };
                        *zoom = DEFAULT_PREVIEW_ZOOM;
                        *pan = vec2(0.0, 0.0);
                    }
                });
            });
        });
}

fn model_view_info_overlay(
    ctx: &egui::Context,
    rect: egui::Rect,
    wireframe: &WireframePreview,
    yaw: f32,
    pitch: f32,
) {
    egui::Area::new(egui::Id::new("model_view_info"))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.left_top() + vec2(MODEL_OVERLAY_INSET, MODEL_OVERLAY_INSET))
        .show(ctx, |ui| {
            model_overlay_frame().show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "Yaw {:+.1}°  Pitch {:+.1}°  Roll {:+.1}°",
                        yaw.to_degrees(),
                        pitch.to_degrees(),
                        0.0_f32,
                    ))
                    .monospace(),
                );
                ui.label(
                    RichText::new(format!(
                        "{} vertices, {} indices ({})",
                        wireframe.vertex_count_total,
                        wireframe.index_count_total,
                        wireframe.position_format
                    ))
                    .small()
                    .color(Color32::GRAY),
                );
            });
        });
}

fn model_export_toolbar(
    ctx: &egui::Context,
    rect: egui::Rect,
    all_mods: &mut bool,
    format: &mut ModelExportFormat,
    all_mods_available: bool,
    exporting: bool,
) -> bool {
    if !all_mods_available {
        *all_mods = false;
    }

    let mut clicked = false;
    egui::Area::new(egui::Id::new("model_export_toolbar"))
        .order(egui::Order::Foreground)
        .fixed_pos(
            rect.right_top()
                + vec2(
                    -MODEL_OVERLAY_INSET,
                    MODEL_OVERLAY_INSET + MODEL_OVERLAY_STACK_STEP * 2.0,
                ),
        )
        .pivot(egui::Align2::RIGHT_TOP)
        .show(ctx, |ui| {
            model_overlay_frame().show(ui, |ui| {
                ui.add_enabled_ui(!exporting, |ui| {
                    ui.horizontal(|ui| {
                        ui.add_enabled(
                            all_mods_available,
                            egui::Checkbox::new(all_mods, "All Mods"),
                        );
                        egui::ComboBox::from_id_salt("model_export_format")
                            .selected_text(format.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(format, ModelExportFormat::Png, "PNG");
                                ui.selectable_value(format, ModelExportFormat::WebP, "WebP");
                            });
                        clicked = ui.button("Export").clicked();
                    });
                });
            });
        });
    clicked
}

pub(super) fn model_wireframe_ui(
    ui: &mut egui::Ui,
    wireframe: &WireframePreview,
    uv_transform: Option<UvTransformPreview>,
    texture_cache: &TextureCache,
    textures: &[(TagHash, UEntryHeader)],
    gpu_preview: Option<&Arc<GpuModelPreview>>,
    camera_frame: Option<ModelCameraFrame>,
    yaw: &mut f32,
    pitch: &mut f32,
    zoom: &mut f32,
    pan: &mut egui::Vec2,
    show_wireframe: &mut bool,
    show_stickers: &mut bool,
    show_default_mods: Option<&mut bool>,
    environment: &mut ModelEnvironment,
    show_textures: bool,
    channels: Option<&mut ModelChannels>,
) -> (egui::Rect, Option<ViewAction>) {
    if !environment.tfx_paused {
        environment.tfx_time_seconds += ui.input(|input| input.stable_dt) * environment.tfx_speed;
        ui.ctx().request_repaint();
    }
    let preview_textures = wireframe_preview_textures(wireframe, textures);

    let available = ui.available_size();
    let size = vec2(available.x.max(320.0), available.y.max(320.0));
    let (rect, response) = ui.allocate_exact_size(size, Sense::drag());
    let toolbar_action = model_viewport_toolbar(
        ui.ctx(),
        rect,
        texture_cache,
        textures,
        gpu_preview,
        environment,
        show_textures,
        channels,
    );
    model_view_options_toolbar(
        ui.ctx(),
        rect,
        show_wireframe,
        show_stickers,
        show_default_mods,
        yaw,
        pitch,
        zoom,
        pan,
    );
    model_view_info_overlay(ui.ctx(), rect, wireframe, *yaw, *pitch);

    let defaults = ModelEnvironment::default();
    let mut orbit_position = environment.light_orbit_position;
    normalize_direction(&mut orbit_position, defaults.light_orbit_position);
    let orbit_view = fixed_light_direction_to_view(orbit_position);
    let light_transform = ModelLightTransform::new(environment, wireframe);
    let light_world_position = light_transform.to_model(light_source_position(environment));
    let orbit_center_world = light_transform.to_model(environment.light_orbit_center);
    let mut cast_direction = light_cast_direction(environment);
    let cast_length = cast_direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    normalize_direction(&mut cast_direction, [0.276, -0.627, -0.728]);
    let cast_view = fixed_light_direction_to_view(cast_direction);
    let camera_frame = camera_frame.unwrap_or_else(|| ModelCameraFrame::from_wireframe(wireframe));
    let viewport_center = rect.center() + *pan;
    let pixels_per_world_unit =
        0.84 * *zoom / camera_frame.radius.max(0.0001) * rect.height() * 0.5;
    let project_model_point = |position: [f32; 3]| {
        let relative = std::array::from_fn(|axis| position[axis] - camera_frame.center[axis]);
        let view = fixed_light_direction_to_view(relative);
        viewport_center
            + vec2(
                -view[0] * pixels_per_world_unit,
                -view[1] * pixels_per_world_unit,
            )
    };
    let gizmo_center = project_model_point(orbit_center_world);
    let light_handle = project_model_point(light_world_position);
    let center_relative =
        std::array::from_fn(|axis| orbit_center_world[axis] - camera_frame.center[axis]);
    let center_view = fixed_light_direction_to_view(center_relative);
    let orbit_radius_world = environment.light_orbit_radius.max(0.0) * light_transform.scale;
    let gizmo_radius = (0..=64)
        .map(|step| {
            let angle = step as f32 / 64.0 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            let point = [
                orbit_center_world[0] + cos * orbit_radius_world,
                orbit_center_world[1],
                orbit_center_world[2] + sin * orbit_radius_world,
            ];
            project_model_point(point).distance(gizmo_center)
        })
        .fold(24.0_f32, f32::max);
    let direction_length = 64.0_f32;
    let direction_world_length = direction_length / pixels_per_world_unit.max(0.0001);
    let direction_handle = project_model_point(std::array::from_fn(|axis| {
        light_world_position[axis] + cast_direction[axis] * direction_world_length
    }));
    let light_handle_radius = (6.0 * environment.light_size.sqrt()).clamp(5.0, 12.0);
    let orbit_gizmo_response = environment.light_gizmo.then(|| {
        ui.interact(
            egui::Rect::from_center_size(light_handle, vec2(32.0, 32.0)),
            ui.id().with("model_light_orbit_point"),
            Sense::drag(),
        )
        .on_hover_text("Move spotlight source on orbit sphere")
    });
    let direction_gizmo_response = environment.light_gizmo.then(|| {
        ui.interact(
            egui::Rect::from_center_size(direction_handle, vec2(28.0, 28.0)),
            ui.id().with("model_light_direction"),
            Sense::drag(),
        )
        .on_hover_text("Move spotlight beam target")
    });
    let center_gizmo_response = environment.light_gizmo.then(|| {
        ui.interact(
            egui::Rect::from_center_size(gizmo_center, vec2(26.0, 26.0)),
            ui.id().with("model_light_orbit_center"),
            Sense::drag(),
        )
        .on_hover_text("Move spotlight orbit center")
    });
    if let Some(gizmo) = &orbit_gizmo_response
        && gizmo.dragged()
        && let Some(pointer) = gizmo.interact_pointer_pos()
    {
        let offset = pointer - gizmo_center;
        let x = (-offset.x / gizmo_radius).clamp(-1.0, 1.0);
        let y = (-offset.y / gizmo_radius).clamp(-1.0, 1.0);
        let planar_length = (x * x + y * y).sqrt();
        let (x, y) = if planar_length > 1.0 {
            (x / planar_length, y / planar_length)
        } else {
            (x, y)
        };
        let z_sign = if orbit_view[2] < 0.0 { -1.0 } else { 1.0 };
        environment.light_orbit_position =
            fixed_view_direction_to_light([x, y, (1.0 - x * x - y * y).max(0.0).sqrt() * z_sign]);
        ui.ctx().request_repaint();
    }
    if let Some(gizmo) = &direction_gizmo_response
        && gizmo.dragged()
        && let Some(pointer) = gizmo.interact_pointer_pos()
    {
        let offset = pointer - light_handle;
        let x = (-offset.x / direction_length).clamp(-1.0, 1.0);
        let y = (-offset.y / direction_length).clamp(-1.0, 1.0);
        let planar_length = (x * x + y * y).sqrt();
        let (x, y) = if planar_length > 1.0 {
            (x / planar_length, y / planar_length)
        } else {
            (x, y)
        };
        let z_sign = if cast_view[2] < 0.0 { -1.0 } else { 1.0 };
        let cast_direction =
            fixed_view_direction_to_light([x, y, (1.0 - x * x - y * y).max(0.0).sqrt() * z_sign]);
        let target_distance = cast_length.max(0.25) * light_transform.scale;
        environment.light_target = light_transform.to_rig(std::array::from_fn(|axis| {
            light_world_position[axis] + cast_direction[axis] * target_distance
        }));
        ui.ctx().request_repaint();
    }
    if let Some(gizmo) = &center_gizmo_response
        && gizmo.dragged()
        && let Some(pointer) = gizmo.interact_pointer_pos()
    {
        let offset = pointer - gizmo_center;
        let center_relative = fixed_view_direction_to_light([
            center_view[0] - offset.x / pixels_per_world_unit.max(0.0001),
            center_view[1] - offset.y / pixels_per_world_unit.max(0.0001),
            center_view[2],
        ]);
        environment.light_orbit_center = light_transform.to_rig(std::array::from_fn(|axis| {
            camera_frame.center[axis] + center_relative[axis]
        }));
        ui.ctx().request_repaint();
    }

    let pointer_delta = ui.input(|i| i.pointer.delta());
    let dragging_light = [
        orbit_gizmo_response.as_ref(),
        direction_gizmo_response.as_ref(),
        center_gizmo_response.as_ref(),
    ]
    .into_iter()
    .flatten()
    .any(|gizmo| gizmo.dragged() || gizmo.hovered());

    if response.dragged_by(egui::PointerButton::Primary) && !dragging_light {
        // Left mouse button: horizontal rotation only
        *yaw += pointer_delta.x * 0.01;
        ui.ctx().request_repaint();
    }

    if response.dragged_by(egui::PointerButton::Secondary) {
        // Right mouse button: vertical rotation only
        *pitch = (*pitch + pointer_delta.y * 0.01).clamp(-1.45, 1.45);
        ui.ctx().request_repaint();
    }

    if response.dragged_by(egui::PointerButton::Middle) {
        // Middle mouse button / wheel button: pan only
        *pan += pointer_delta;
        ui.ctx().request_repaint();
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            *zoom = (*zoom + scroll * 0.01).clamp(0.05, 50.0);
            // The wheel belongs to model zoom while the pointer is over the
            // viewport; do not also pass it to the surrounding texture scroll.
            ui.input_mut(|input| input.smooth_scroll_delta.y = 0.0);
            ui.ctx().request_repaint();
        }
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_rgb(109, 143, 176));
    painter.rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0, Color32::from_rgb(60, 70, 80)),
        egui::StrokeKind::Middle,
    );

    let gpu_rendered = if let Some(gpu_preview) = gpu_preview
        && wireframe.uvs.is_some()
        && !preview_textures.is_empty()
    {
        let callback = ModelPaintCallback::new(
            gpu_preview.clone(),
            texture_cache,
            wireframe,
            uv_transform,
            Some(camera_frame),
            *yaw,
            *pitch,
            *zoom,
            *pan,
            *show_stickers,
            rect,
            ui.ctx().pixels_per_point(),
            *environment,
        );
        callback.show_loading_status(ui, rect);
        ui.painter()
            .add(Callback::new_paint_callback(rect, callback));
        true
    } else {
        false
    };

    if environment.light_gizmo {
        let sphere_stroke = Stroke::new(1.0, Color32::from_white_alpha(90));
        for (axis, color) in [
            (0, Color32::from_rgba_unmultiplied(245, 90, 90, 125)),
            (1, Color32::from_rgba_unmultiplied(90, 220, 120, 125)),
        ] {
            let points = (0..=64)
                .map(|step| {
                    let angle = step as f32 / 64.0 * std::f32::consts::TAU;
                    let (sin, cos) = angle.sin_cos();
                    let offset = if axis == 0 {
                        [cos * orbit_radius_world, 0.0, sin * orbit_radius_world]
                    } else {
                        [0.0, cos * orbit_radius_world, sin * orbit_radius_world]
                    };
                    project_model_point(std::array::from_fn(|component| {
                        orbit_center_world[component] + offset[component]
                    }))
                })
                .collect::<Vec<_>>();
            painter.add(egui::Shape::line(points, Stroke::new(1.0, color)));
        }
        painter.circle_stroke(gizmo_center, gizmo_radius, sphere_stroke);
        painter.line_segment(
            [gizmo_center, light_handle],
            Stroke::new(1.2, Color32::from_rgba_unmultiplied(255, 220, 105, 170)),
        );
        painter.line_segment(
            [gizmo_center - vec2(7.0, 0.0), gizmo_center + vec2(7.0, 0.0)],
            Stroke::new(1.5, Color32::from_rgb(115, 210, 255)),
        );
        painter.line_segment(
            [gizmo_center - vec2(0.0, 7.0), gizmo_center + vec2(0.0, 7.0)],
            Stroke::new(1.5, Color32::from_rgb(115, 210, 255)),
        );
        painter.circle_filled(
            light_handle,
            light_handle_radius,
            Color32::from_rgb(255, 218, 90),
        );
        painter.circle_stroke(
            light_handle,
            light_handle_radius,
            Stroke::new(1.5, Color32::WHITE),
        );
        painter.arrow(
            light_handle,
            direction_handle - light_handle,
            Stroke::new(2.0, Color32::from_rgb(255, 126, 82)),
        );
        painter.circle_filled(direction_handle, 4.5, Color32::from_rgb(255, 126, 82));
    }

    let Some(projected) = project_vertices(
        wireframe,
        Some(camera_frame),
        *yaw,
        *pitch,
        *zoom,
        *pan,
        rect,
    ) else {
        return (rect, toolbar_action);
    };

    if wireframe.indices.len() >= 3 {
        if !gpu_rendered && wireframe.uvs.is_some() && !preview_textures.is_empty() {
            draw_textured_model_mesh(
                &painter,
                wireframe,
                &projected,
                texture_cache,
                textures,
                uv_transform,
                *show_stickers,
            );
        }

        if *show_wireframe {
            let stroke = Stroke::new(0.7, Color32::from_rgb(140, 210, 255));
            for tri in wireframe.indices.chunks_exact(3) {
                let Some(a) = projected.get(tri[0] as usize).map(|v| v.pos) else {
                    continue;
                };
                let Some(b) = projected.get(tri[1] as usize).map(|v| v.pos) else {
                    continue;
                };
                let Some(c) = projected.get(tri[2] as usize).map(|v| v.pos) else {
                    continue;
                };
                painter.line_segment([a, b], stroke);
                painter.line_segment([b, c], stroke);
                painter.line_segment([c, a], stroke);
            }
        }
    } else {
        for point in projected.iter().take(50_000) {
            painter.circle_filled(point.pos, 1.0, Color32::from_rgb(140, 210, 255));
        }
    }

    (rect, toolbar_action)
}

fn draw_textured_model_mesh(
    painter: &egui::Painter,
    wireframe: &WireframePreview,
    projected: &[ProjectedVertex],
    texture_cache: &TextureCache,
    fallback_textures: &[(TagHash, UEntryHeader)],
    uv_transform: Option<UvTransformPreview>,
    show_stickers: bool,
) {
    let mut base_triangles = projected_triangles(wireframe, projected, false, show_stickers);
    base_triangles.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    let mut triangles = projected_triangles(wireframe, projected, false, show_stickers);
    triangles.sort_by(|a, b| a.depth.total_cmp(&b.depth));

    let mut base_mesh = Mesh::default();
    add_projected_triangles_to_mesh(
        &mut base_mesh,
        wireframe,
        projected,
        &base_triangles,
        uv_transform,
        Color32::from_rgb(135, 100, 92),
    );
    if !base_mesh.indices.is_empty() {
        painter.add(egui::Shape::mesh(base_mesh));
    }

    let fallback_texture = (!wireframe
        .material_ranges
        .iter()
        .any(|range| range.texture.is_some()))
    .then(|| fallback_textures.first().map(|(tag, _entry)| *tag))
    .flatten();
    let material_keys = triangles
        .iter()
        .filter_map(|triangle| triangle_material_key(triangle, fallback_texture))
        .unique()
        .collect_vec();
    for key in material_keys {
        let texture_id = texture_cache
            .get_material_or_load(key)
            .map(|(_texture, texture_id)| texture_id)
            .unwrap_or_else(|| texture_cache.get_or_default(key.color.expect("textured triangle")).1);
        let texture_triangles = triangles
            .iter()
            .copied()
            .filter(|triangle| triangle_material_key(triangle, fallback_texture) == Some(key))
            .collect_vec();
        let mut mesh = Mesh::with_texture(texture_id);
        add_projected_triangles_to_mesh(
            &mut mesh,
            wireframe,
            projected,
            &texture_triangles,
            uv_transform,
            Color32::WHITE,
        );

        if !mesh.indices.is_empty() {
            painter.add(egui::Shape::mesh(mesh));
        }
    }
}

fn triangle_material_key(
    triangle: &ProjectedTriangle,
    fallback_texture: Option<TagHash>,
) -> Option<MaterialTextureKey> {
    Some(MaterialTextureKey {
        color: Some(triangle.texture.or(fallback_texture)?),
        normal: triangle.normal,
        emissive: triangle.emissive,
        color_tint: triangle.color_tint,
        emissive_strength: triangle.emissive_strength,
    })
}

fn projected_triangles(
    wireframe: &WireframePreview,
    projected: &[ProjectedVertex],
    expand_layers: bool,
    show_stickers: bool,
) -> Vec<ProjectedTriangle> {
    let mut material_iter = wireframe.material_ranges.iter().peekable();
    wireframe
        .indices
        .chunks_exact(3)
        .enumerate()
        .flat_map(|(triangle_index, tri)| {
            let index_start = triangle_index * 3;
            while material_iter
                .peek()
                .is_some_and(|range| index_start >= range.index_start + range.index_count)
            {
                material_iter.next();
            }
            let material_range = material_iter.peek().filter(|range| {
                index_start >= range.index_start
                    && index_start < range.index_start + range.index_count
            });
            if material_range.is_some_and(|range| {
                !show_stickers && range.technique.is_some_and(is_sticker_proxy_technique)
            }) {
                return Vec::new();
            }
            let Some(a) = projected.get(tri[0] as usize) else {
                return Vec::new();
            };
            let Some(b) = projected.get(tri[1] as usize) else {
                return Vec::new();
            };
            let Some(c) = projected.get(tri[2] as usize) else {
                return Vec::new();
            };

            let depth = (a.depth + b.depth + c.depth) / 3.0;
            let light = face_light(*a, *b, *c);
            let layers = projected_material_layers(material_range.copied(), expand_layers);
            layers
                .into_iter()
                .map(|layer| ProjectedTriangle {
                    indices: [tri[0], tri[1], tri[2]],
                    depth,
                    texture: layer.color,
                    normal: layer.normal,
                    emissive: layer.emissive,
                    color_tint: layer.color_tint,
                    emissive_strength: layer.emissive_strength,
                    light,
                })
                .collect_vec()
        })
        .collect()
}

#[derive(Clone, Copy)]
struct ProjectedMaterialLayer {
    color: Option<TagHash>,
    normal: Option<TagHash>,
    emissive: Option<TagHash>,
    color_tint: [u8; 4],
    emissive_strength: u8,
}

fn projected_material_layers(
    range: Option<&crate::geometry::WireframeMaterialRange>,
    expand_layers: bool,
) -> Vec<ProjectedMaterialLayer> {
    let Some(range) = range else {
        return vec![ProjectedMaterialLayer {
            color: None,
            normal: None,
            emissive: None,
            color_tint: [255, 255, 255, 255],
            emissive_strength: 0,
        }];
    };

    let source_layers = if expand_layers && !range.textures.layers.is_empty() {
        range.textures.layers.clone()
    } else {
        vec![WireframeMaterialLayer {
            color: range.textures.color.or(range.texture),
            normal: range.textures.normal,
            emissive: range.textures.emissive,
        }]
    };

    source_layers
        .into_iter()
        .map(|layer| ProjectedMaterialLayer {
            color: layer.color.or(range.texture),
            normal: layer.normal,
            emissive: layer.emissive,
            color_tint: range.textures.color_tint,
            emissive_strength: range.textures.emissive_strength,
        })
        .collect()
}

fn face_light(a: ProjectedVertex, b: ProjectedVertex, c: ProjectedVertex) -> f32 {
    let ab = [
        b.view[0] - a.view[0],
        b.view[1] - a.view[1],
        b.view[2] - a.view[2],
    ];
    let ac = [
        c.view[0] - a.view[0],
        c.view[1] - a.view[1],
        c.view[2] - a.view[2],
    ];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if len <= f32::EPSILON {
        return 0.82;
    }

    let nx = normal[0] / len;
    let ny = normal[1] / len;
    let nz = normal[2] / len;
    let light = [0.35_f32, -0.45, 0.82];
    let light_len = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt();
    let lambert =
        (nx * light[0] / light_len + ny * light[1] / light_len + nz * light[2] / light_len).abs();
    (0.85 + lambert * 0.15).clamp(0.75, 1.0)
}

fn wireframe_preview_textures(
    wireframe: &WireframePreview,
    textures: &[(TagHash, UEntryHeader)],
) -> Vec<TagHash> {
    let assigned = wireframe
        .material_ranges
        .iter()
        .filter_map(|range| range.texture)
        .unique()
        .collect_vec();
    if assigned.is_empty() {
        textures
            .first()
            .map(|(tag, _entry)| *tag)
            .into_iter()
            .collect()
    } else {
        assigned
    }
}

fn add_projected_triangles_to_mesh(
    mesh: &mut Mesh,
    wireframe: &WireframePreview,
    projected: &[ProjectedVertex],
    triangles: &[ProjectedTriangle],
    uv_transform: Option<UvTransformPreview>,
    color: Color32,
) {
    for tri in triangles {
        let base = mesh.vertices.len() as u32;
        let color = shaded_color(color, tri.light);
        for index in tri.indices {
            let vertex_index = index as usize;
            let Some(position) = wireframe.vertices.get(vertex_index).copied() else {
                continue;
            };
            let Some(screen_pos) = projected.get(vertex_index).map(|v| v.pos) else {
                continue;
            };
            mesh.vertices.push(Vertex {
                pos: screen_pos,
                uv: preview_uv(vertex_index, position, wireframe, uv_transform),
                color,
            });
        }

        if mesh.vertices.len() as u32 == base + 3 {
            mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        } else {
            mesh.vertices.truncate(base as usize);
        }
    }
}

fn shaded_color(color: Color32, light: f32) -> Color32 {
    let rgba = color.to_array();
    Color32::from_rgba_unmultiplied(
        (rgba[0] as f32 * light).clamp(0.0, 255.0) as u8,
        (rgba[1] as f32 * light).clamp(0.0, 255.0) as u8,
        (rgba[2] as f32 * light).clamp(0.0, 255.0) as u8,
        rgba[3],
    )
}

fn preview_uv(
    vertex_index: usize,
    position: [f32; 3],
    wireframe: &WireframePreview,
    uv_transform: Option<UvTransformPreview>,
) -> egui::Pos2 {
    if let Some(uvs) = &wireframe.uvs
        && let Some(uv) = uvs.get(vertex_index)
    {
        return transformed_uv(*uv, uv_transform);
    }

    let _ = (position, wireframe, uv_transform);
    pos2(0.0, 0.0)
}

fn transformed_uv(uv: [f32; 2], uv_transform: Option<UvTransformPreview>) -> egui::Pos2 {
    let mut u = uv[0];
    let mut v = uv[1];
    if let Some(transform) = uv_transform {
        u = u * transform.scale[0] + transform.offset[0];
        v = v * transform.scale[1] + transform.offset[1];
    }

    pos2(u.fract().abs(), v.fract().abs())
}

fn project_vertices(
    wireframe: &WireframePreview,
    camera_frame: Option<ModelCameraFrame>,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    pan: egui::Vec2,
    rect: egui::Rect,
) -> Option<Vec<ProjectedVertex>> {
    if wireframe.vertices.is_empty() {
        return None;
    }

    let camera_frame = camera_frame.unwrap_or_else(|| ModelCameraFrame::from_wireframe(wireframe));
    let center = camera_frame.center;
    let radius = camera_frame.radius;
    let scale = rect.width().min(rect.height()) * 0.42 * zoom / radius;
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let screen_center = rect.center() + pan;

    Some(
        wireframe
            .vertices
            .iter()
            .map(|position| {
                let x = position[0] - center[0];
                let y = position[2] - center[2];
                let z = position[1] - center[1];
                let xz = x * cy + z * sy;
                let zz = -x * sy + z * cy;
                let yz = y * cp - zz * sp;
                let depth = y * sp + zz * cp;

                ProjectedVertex {
                    pos: pos2(screen_center.x - xz * scale, screen_center.y - yz * scale),
                    depth,
                    view: [xz, yz, depth],
                }
            })
            .collect(),
    )
}

fn model_textures_ui(
    ui: &mut egui::Ui,
    texture_cache: &TextureCache,
    textures: &[(TagHash, UEntryHeader)],
) -> Option<ViewAction> {
    let mut action = None;

    ui.heading("Textures");
    if textures.is_empty() {
        ui.label(RichText::new("No textures found in traversal cache").italics());
        return None;
    }

    ui.horizontal_wrapped(|ui| {
        for (tag, entry) in textures {
            let response = ui.allocate_response(vec2(112.0, 138.0), Sense::click());
            let rect = response.rect;
            let painter = ui.painter_at(rect);

            painter.rect_filled(rect, 4.0, Color32::from_rgb(20, 22, 24));

            let image_rect = egui::Rect::from_min_size(rect.min + vec2(8.0, 8.0), vec2(96.0, 96.0));
            if ui.is_rect_visible(image_rect) {
                let (tex, tid) = texture_cache.get_or_default(*tag);
                painter.rect_filled(image_rect, 3.0, Color32::BLACK);
                ui_image_rotated(
                    &painter,
                    tid,
                    image_rect,
                    if tex.desc.array_size == 6 { 90.0 } else { 0.0 },
                    tex.desc.array_size == 6,
                );
            }

            let text = format_tag_entry(*tag, Some(entry));
            painter.text(
                pos2(rect.left() + 8.0, rect.bottom() - 26.0),
                egui::Align2::LEFT_TOP,
                text,
                egui::TextStyle::Small.resolve(ui.style()),
                Color32::WHITE,
            );

            if response.hovered() {
                painter.rect_stroke(
                    rect,
                    4.0,
                    Stroke::new(1.0, Color32::WHITE),
                    egui::StrokeKind::Middle,
                );
            }

            if response
                .tag_context_with_preview(*tag, Some(texture_cache), true)
                .on_hover_text(format!("Show {tag} in Textures tab"))
                .clicked()
            {
                action = Some(ViewAction::ShowTexture(*tag));
            }
        }
    });

    action
}

#[cfg(test)]
mod tests {
    include!("model_lighting_catalog_test.rs");
    #[test]
    #[ignore = "requires installed Marathon packages and GPU"]
    fn preserves_open_and_skin_switch_inputs() {
        use super::*;
        use std::time::Instant;
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into());
        let manager = Arc::new(
            PackageManager::new(
                packages,
                GameVersion::Marathon(MarathonVersion::Marathon),
                None,
            )
            .expect("packages"),
        );
        tiger_pkg::initialize_package_manager(&manager);
        quicktag_core::classes::initialize_reference_names();
        let graph = Arc::new(quicktag_scanner::load_tag_cache());
        let strings = Arc::new(quicktag_strings::localized::create_stringmap().expect("strings"));
        let mut gear = super::super::gear::GearView::new(strings);
        gear.reconcile_weapon_skin_models(&graph);
        let catalog = gear.model_weapon_catalog();
        let target = TagHash(0x80B7CE0A);
        let owner = weapon_index_for_model(&catalog, target).expect("target catalog owner");
        let weapon = &catalog.weapons[owner];
        let other = weapon
            .skins
            .iter()
            .find(|skin| skin.model_tag != target)
            .expect("another skin for switch test")
            .model_tag;
        eprintln!(
            "open benchmark: {} target={target} alternate={other} owner={} slots={}",
            weapon.name,
            weapon.owner_tag,
            weapon.slots.len()
        );
        let render_state = crate::create_headless_render_state().expect("GPU");
        let mut view = ModelsView::new(graph, TextureCache::new(render_state));
        view.set_weapon_catalog(catalog);

        let start = Instant::now();
        crate::asset_cache::without_asset_cache(|| view.load_model(target));
        let uncached_ms = start.elapsed().as_secs_f64() * 1000.0;
        let expected = format!("{:?}", view.preview.as_ref().expect("preview"));
        let expected_frame = view.preview_camera_frame;
        let expected_export = view.weapon_export_camera;
        let expected_gpu = view
            .gpu_model_preview
            .as_ref()
            .expect("GPU preview")
            .inspection_lines();
        let expected_vertices = view
            .gpu_model_preview
            .as_ref()
            .unwrap()
            .vertex_input_bytes();
        eprintln!("open uncached_ms={uncached_ms:.3}");
        for (iteration, selected) in [target, other, target, other, target]
            .into_iter()
            .enumerate()
        {
            let start = Instant::now();
            view.load_model(selected);
            eprintln!(
                "open {iteration} tag={selected} elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            if selected == target {
                assert!(
                    format!("{:?}", view.preview.as_ref().unwrap()) == expected,
                    "skin switch changed decoded model"
                );
                assert_eq!(view.preview_camera_frame, expected_frame);
                assert_eq!(view.weapon_export_camera, expected_export);
                assert_eq!(
                    view.gpu_model_preview.as_ref().unwrap().inspection_lines(),
                    expected_gpu
                );
                assert!(
                    view.gpu_model_preview
                        .as_ref()
                        .unwrap()
                        .vertex_input_bytes()
                        == expected_vertices,
                    "skin switch changed GPU vertex bytes"
                );
            }
        }
        // Driver teardown is unrelated to load timing and can block on Windows.
        std::mem::forget(view);
    }

    use super::*;

    #[test]
    fn export_filename_distinguishes_mod_rarity_concisely() {
        let model = TagHash(0x80B7CAE9);
        let mod_tag = TagHash(0x80A6071A);
        assert_eq!(
            model_export_filename(model, &[], ModelExportFormat::Png),
            "80B7CAE9.png"
        );
        assert_eq!(
            model_export_filename(model, &[], ModelExportFormat::WebP),
            "80B7CAE9.webp"
        );
        assert_eq!(model_modded_export_filename(model), "80B7CAE9.zip");
        assert_eq!(
            model_export_filename(model, &[(mod_tag, "E")], ModelExportFormat::Png),
            "80B7CAE9_80A6071A-E.png"
        );
        assert_eq!(
            model_export_filename(model, &[(mod_tag, "D")], ModelExportFormat::Png),
            "80B7CAE9_80A6071A-D.png"
        );
        assert_eq!(
            model_export_filename(model, &[(mod_tag, "S")], ModelExportFormat::Png),
            "80B7CAE9_80A6071A-S.png"
        );
        assert_eq!(
            model_export_filename(model, &[(mod_tag, "P")], ModelExportFormat::Png),
            "80B7CAE9_80A6071A-P.png"
        );
        assert_eq!(
            model_export_filename(model, &[(mod_tag, "C")], ModelExportFormat::Png),
            "80B7CAE9_80A6071A-C.png"
        );
        assert_eq!(
            model_export_filename(
                model,
                &[(mod_tag, "E"), (TagHash(0x80A61008), "D")],
                ModelExportFormat::Png,
            ),
            "80B7CAE9_80A6071A-E_80A61008-D.png"
        );
    }

    #[test]
    fn all_modded_uses_cartesian_product_of_superior_and_above() {
        fn modification(tag: u32, rarity_code: &'static str) -> ModelModEntry {
            ModelModEntry {
                name: format!("{tag:08X}"),
                rarity: String::new(),
                rarity_code,
                color: Color32::WHITE,
                model_tag: TagHash(tag),
                preview_rarity: None,
            }
        }
        let weapon = ModelWeaponEntry {
            name: "Test".to_owned(),
            owner_tag: TagHash(1),
            socket_owner: None,
            model_tags: vec![],
            skins: vec![],
            slots: vec![
                super::super::gear::ModelModSlot {
                    name: "A".to_owned(),
                    mods: vec![
                        modification(10, "D"),
                        modification(11, "S"),
                        modification(12, "P"),
                    ],
                },
                super::super::gear::ModelModSlot {
                    name: "B".to_owned(),
                    mods: vec![modification(20, "C"), modification(21, "E")],
                },
            ],
        };
        let combinations = high_rarity_mod_combinations(&weapon);
        assert_eq!(combinations.len(), 2);
        assert_eq!(
            combinations
                .iter()
                .map(|items| items.iter().map(|item| item.model_tag.0).collect_vec())
                .collect_vec(),
            [vec![11, 20], vec![12, 20]]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires installed Marathon packages and GPU"]
    async fn exports_modded_pngs_into_named_zip() {
        use std::io::Read;
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};
        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into());
        let manager = Arc::new(
            PackageManager::new(
                packages,
                GameVersion::Marathon(MarathonVersion::Marathon),
                None,
            )
            .unwrap(),
        );
        tiger_pkg::initialize_package_manager(&manager);
        quicktag_core::classes::initialize_reference_names();
        let graph = Arc::new(quicktag_scanner::load_tag_cache());
        let strings = Arc::new(quicktag_strings::localized::create_stringmap().unwrap());
        let mut gear = super::super::gear::GearView::new(strings);
        gear.reconcile_weapon_skin_models(&graph);
        let weapon = gear
            .model_weapon_catalog()
            .weapons
            .into_iter()
            .find(|weapon| weapon.name == "Firestorm")
            .unwrap();
        assert_eq!(high_rarity_mod_combinations(&weapon).len(), 1);
        let model = weapon.model_tags[0];
        let path = std::env::temp_dir().join(format!(
            "quicktag-{model}-{}-modded-export.zip",
            std::process::id()
        ));
        let texture_cache = TextureCache::new(crate::create_headless_render_state().unwrap());
        let mut job = ModdedExportJob::new(
            model,
            weapon,
            true,
            false,
            ModelEnvironment::default(),
            None,
            None,
            ModelExportFormat::Png,
            path.clone(),
        )
        .unwrap();
        while !job.step(&graph, &texture_cache).unwrap() {}
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        assert_eq!(archive.len(), 1);
        let mut png = vec![];
        archive.by_index(0).unwrap().read_to_end(&mut png).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        drop(archive);
        std::fs::remove_file(&path).unwrap();
        std::mem::forget(texture_cache);
    }

    #[test]
    #[ignore = "requires current installed Marathon packages"]
    fn audits_known_model_catalog_regressions() {
        use std::path::PathBuf;
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let pm = PackageManager::new(
            packages.to_string_lossy().to_string(),
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .expect("package manager");
        tiger_pkg::initialize_package_manager(&Arc::new(pm));
        quicktag_core::classes::initialize_reference_names();
        let cache = quicktag_scanner::load_tag_cache();
        let strings =
            Arc::new(quicktag_strings::localized::create_stringmap().expect("localized strings"));
        let mut gear = super::super::gear::GearView::new(strings);
        gear.reconcile_weapon_skin_models(&cache);
        let catalog = gear.model_weapon_catalog();

        let conquest = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "Conquest LMG")
            .expect("Conquest catalog entry");
        assert!(conquest.model_tags.contains(&TagHash(0x80B7C031)));
        let components = pattern_component_descendants(&cache, conquest.model_tags.iter().copied());
        assert!(components.contains(&TagHash(0x80B7C02F)));
        assert!(components.contains(&TagHash(0x80B7C030)));
        assert!(!components.contains(&TagHash(0x80B7C031)));

        let lookout = catalog
            .weapons
            .iter()
            .find(|weapon| weapon.name == "V66 Lookout")
            .expect("V66 catalog entry");
        let components = pattern_component_descendants(&cache, lookout.model_tags.iter().copied());
        assert!(components.contains(&TagHash(0x80B7C1D3)));

        let ikari = catalog
            .runner_skins
            .iter()
            .find(|skin| skin.name.to_lowercase().contains("ikari"))
            .expect("Ikari runner skin catalog entry");
        let entry = package_manager()
            .get_entry(ikari.model_tag)
            .expect("Ikari model tag");
        assert_eq!(ikari.name, "IKARI YŌKAI");
        assert_eq!(ikari.shell_name, "Destroyer");
        assert_eq!(ikari.model_tag, TagHash(0x80A9F6BD));
        assert_eq!(entry.reference, 0x8080BAAD);
        let compositions = runner_model_index(&cache, &catalog);
        let composed = compositions
            .get(&ikari.model_tag)
            .expect("Ikari authored shell Pattern");
        assert_eq!(composed.pattern, ikari.model_tag);
        assert_eq!(composed.geometry().len(), 2);
        assert_eq!(
            runner_skin_for_model(&catalog, ikari.model_tag).map(|skin| skin.name.as_str()),
            Some("IKARI YŌKAI")
        );
    }

    #[test]
    fn detects_only_exact_authored_weapon_models_and_skins() {
        let catalog = ModelWeaponCatalog {
            weapons: vec![ModelWeaponEntry {
                name: "Misriah 2442".to_owned(),
                owner_tag: TagHash(0x80a7acec),
                socket_owner: None,
                model_tags: vec![TagHash(0x80aa0e95), TagHash(0x80b6cc5d)],
                skins: vec![ModelWeaponSkinEntry {
                    name: "Yōkai's Claw".to_owned(),
                    model_tag: TagHash(0x80b6cc5d),
                    rarity: Some("Prestige".to_owned()),
                    color: Color32::from_rgb(232, 184, 72),
                }],
                slots: vec![],
            }],
            runner_skins: vec![],
            melee_skins: vec![],
            charms: vec![],
        };

        assert_eq!(
            weapon_index_for_model(&catalog, TagHash(0x80b6cc5d)),
            Some(0)
        );
        assert_eq!(
            weapon_index_for_model(&catalog, TagHash(0x80aa0e95)),
            Some(0)
        );
        assert_eq!(weapon_index_for_model(&catalog, TagHash(0x80a7aced)), None);
        let (weapon, skin) =
            weapon_skin_for_model(&catalog, TagHash(0x80b6cc5d)).expect("exact skin tag");
        assert_eq!(weapon.name, "Misriah 2442");
        assert_eq!(skin.name, "Yōkai's Claw");
        assert_eq!(skin.rarity.as_deref(), Some("Prestige"));
        assert!(weapon_skin_for_model(&catalog, TagHash(0x80b6cc5e)).is_none());
    }

    #[test]
    fn rejects_model_tags_claimed_by_multiple_weapons() {
        let shared = TagHash(0x80b6cc5d);
        let catalog = ModelWeaponCatalog {
            weapons: ["ARES RG", "V00 ZEUS RG"]
                .into_iter()
                .map(|name| ModelWeaponEntry {
                    name: name.to_owned(),
                    owner_tag: shared,
                    socket_owner: None,
                    model_tags: vec![shared],
                    skins: vec![ModelWeaponSkinEntry {
                        name: format!("{name} skin"),
                        model_tag: shared,
                        rarity: None,
                        color: Color32::WHITE,
                    }],
                    slots: vec![],
                })
                .collect(),
            runner_skins: vec![],
            melee_skins: vec![],
            charms: vec![],
        };

        assert_eq!(weapon_index_for_model(&catalog, shared), None);
        assert!(weapon_skin_for_model(&catalog, shared).is_none());
    }

    #[test]
    fn finds_runner_skin_on_authored_pattern() {
        let skin = ModelRunnerSkinEntry {
            name: "Arata Vectus".to_owned(),
            shell_name: "Assassin".to_owned(),
            model_tag: TagHash(0x80B140CE),
            color: Color32::from_rgb(232, 184, 72),
        };
        let catalog = ModelWeaponCatalog {
            weapons: vec![],
            runner_skins: vec![skin],
            melee_skins: vec![],
            charms: vec![],
        };
        let found = runner_skin_for_model(&catalog, TagHash(0x80B140CE)).expect("shell identity");
        assert_eq!(found.shell_name, "Assassin");
        assert_eq!(found.name, "Arata Vectus");
        assert_eq!(found.color, Color32::from_rgb(232, 184, 72));
        assert_eq!(
            runner_skin_for_model(&catalog, TagHash(0x80B140CE)).map(|skin| skin.name.as_str()),
            Some("Arata Vectus")
        );
    }

    #[test]
    fn finds_all_melee_skins_sharing_an_authored_pattern() {
        let model = TagHash(0x80B6E6D2);
        let catalog = ModelWeaponCatalog {
            weapons: vec![],
            runner_skins: vec![],
            melee_skins: vec![
                ModelMeleeSkinEntry {
                    name: "Alpha Cutter".to_owned(),
                    family_name: "Knives".to_owned(),
                    model_tag: model,
                    color: Color32::from_rgb(166, 92, 214),
                },
                ModelMeleeSkinEntry {
                    name: "Monoclast".to_owned(),
                    family_name: "Knives".to_owned(),
                    model_tag: model,
                    color: Color32::from_rgb(232, 184, 72),
                },
            ],
            charms: vec![],
        };
        let found = melee_skins_for_model(&catalog, model);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, "Alpha Cutter");
        assert_eq!(found[1].name, "Monoclast");
    }

    #[test]
    fn finds_all_charms_sharing_an_authored_pattern() {
        let model = TagHash(0x80B6E701);
        let catalog = ModelWeaponCatalog {
            weapons: vec![],
            runner_skins: vec![],
            melee_skins: vec![],
            charms: vec![
                ModelCharmEntry {
                    name: "Lucky Cat".to_owned(),
                    model_tag: model,
                    color: Color32::from_rgb(166, 92, 214),
                },
                ModelCharmEntry {
                    name: "Second Charm".to_owned(),
                    model_tag: model,
                    color: Color32::from_rgb(232, 184, 72),
                },
            ],
        };
        let found = charms_for_model(&catalog, model);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].name, "Lucky Cat");
        assert_eq!(found[1].name, "Second Charm");
    }

    #[test]
    fn identified_skin_rows_override_single_line_truncation() {
        let ctx = egui::Context::default();
        let mut single_line_height = 0.0;
        let mut skin_row_height = 0.0;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.set_width(420.0);
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                single_line_height = ui
                    .add(egui::Button::selectable(
                        false,
                        "1610: Model container 80B6E64A (3.33 KB)",
                    ))
                    .rect
                    .height();

                let mut label = egui::text::LayoutJob::default();
                label.append(
                    "1610: Model container 80B6E64A (3.33 KB)",
                    0.0,
                    egui::TextFormat::default(),
                );
                label.append(
                    "\n    └ KKV-9SD: TAC Standard",
                    0.0,
                    egui::TextFormat::default(),
                );
                skin_row_height = ui
                    .add(egui::Button::selectable(false, label).wrap_mode(egui::TextWrapMode::Wrap))
                    .rect
                    .height();
            });
        });

        assert!(
            skin_row_height > single_line_height * 1.5,
            "skin identity was clipped: single={single_line_height}, skin={skin_row_height}"
        );
    }
}
