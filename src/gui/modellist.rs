use std::sync::Arc;

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
    GeometryPreviewKind, GeometryTagPreview, ModelTagInfo, ModelTagRole, UvTransformPreview,
    WeaponModPreviewAttachment, WeaponModSocketIndex, WireframeMaterialLayer, WireframePreview,
    is_model_catalog_reference, model_info_for_reference, weapon_unoccupied_default_mod_patterns,
};
use crate::gui::common::ResponseExt;
use crate::gui::tag::format_tag_entry;
use crate::material::is_sticker_proxy_technique;
use crate::texture::cache::{MaterialTextureKey, TextureCache};
use crate::util::{format_file_size, ui_image_rotated};

use super::gear::{ModelModEntry, ModelWeaponCatalog, ModelWeaponEntry, ModelWeaponSkinEntry};
use super::model_renderer::{
    GpuModelPreview, ModelCameraFrame, ModelEnvironment, ModelPaintCallback,
};
use super::{View, ViewAction};

pub(super) const DEFAULT_MODEL_YAW: f32 = -std::f32::consts::FRAC_PI_2;
const MODEL_PARAMETER_RANGE: std::ops::RangeInclusive<f32> = -10.0..=10.0;

pub struct ModelsView {
    cache: Arc<TagCache>,
    texture_cache: TextureCache,
    selected_package: u16,
    packages_with_models: Vec<u16>,
    package_filter: String,
    model_filter: String,
    models: Vec<ModelListEntry>,
    selected_model: Option<TagHash>,
    preview: Option<GeometryTagPreview>,
    gpu_model_preview: Option<Arc<GpuModelPreview>>,
    preview_camera_frame: Option<ModelCameraFrame>,
    preview_yaw: f32,
    preview_pitch: f32,
    preview_zoom: f32,
    preview_pan: egui::Vec2,
    preview_show_wireframe: bool,
    preview_show_stickers: bool,
    preview_show_default_mods: bool,
    preview_environment: ModelEnvironment,
    weapon_catalog: ModelWeaponCatalog,
    active_weapon: Option<usize>,
    selected_mods: Vec<Option<usize>>,
    selected_mod_unique_ids: Vec<Option<f32>>,
    mod_unique_rng: u32,
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
            models: vec![],
            selected_model: None,
            preview: None,
            gpu_model_preview: None,
            preview_camera_frame: None,
            preview_yaw: DEFAULT_MODEL_YAW,
            preview_pitch: 0.05,
            preview_zoom: 1.0,
            preview_pan: vec2(0.0, 0.0),
            preview_show_wireframe: false,
            preview_show_stickers: false,
            preview_show_default_mods: true,
            preview_environment: ModelEnvironment::default(),
            weapon_catalog: ModelWeaponCatalog::default(),
            active_weapon: None,
            selected_mods: vec![],
            selected_mod_unique_ids: vec![],
            mod_unique_rng: 0xA341_316C,
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
        }
    }

    pub fn set_cache(&mut self, cache: Arc<TagCache>) {
        self.cache = cache;
        for weapon in &mut self.weapon_catalog.weapons {
            weapon.socket_owner = None;
        }
        self.resolve_weapon_socket_owners();
        self.preview = None;
        self.gpu_model_preview = None;
        self.preview_camera_frame = None;
        if let Some(tag) = self.selected_model {
            self.load_model(tag);
        }
    }

    pub fn show_model(&mut self, tag: TagHash) {
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
                Some(ModelListEntry {
                    index: i,
                    tag: TagHash::new(id, i as u16),
                    info,
                    entry: entry.clone(),
                })
            })
            .collect();
        self.selected_model = None;
        self.preview = None;
        self.gpu_model_preview = None;
        self.preview_camera_frame = None;
        self.active_weapon = None;
        self.selected_mods.clear();
        self.selected_mod_unique_ids.clear();

        if let Some(tag) = self.models.first().map(|entry| entry.tag) {
            self.load_model(tag);
        }
    }

    fn load_model(&mut self, tag: TagHash) {
        self.selected_model = Some(tag);
        self.preview_camera_frame = None;
        self.detect_selected_weapon();
        self.rebuild_model_preview();
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
        self.preview = if model_info_for_reference(entry.reference).is_some() {
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
            )
            .map(Arc::new)
        });
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
        let preview_yaw = &mut self.preview_yaw;
        let preview_pitch = &mut self.preview_pitch;
        let preview_zoom = &mut self.preview_zoom;
        let preview_pan = &mut self.preview_pan;
        let preview_show_wireframe = &mut self.preview_show_wireframe;
        let preview_show_stickers = &mut self.preview_show_stickers;
        let default_mods_before = self.preview_show_default_mods;
        let preview_show_default_mods = &mut self.preview_show_default_mods;
        let preview_environment = &mut self.preview_environment;
        let active_weapon = self
            .active_weapon
            .and_then(|index| self.weapon_catalog.weapons.get(index))
            .cloned();
        let selected_mods = self.selected_mods.clone();
        let mut mod_selection = None;

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
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if let Some(wireframe) = &model.wireframe {
                        let viewport = model_wireframe_ui(
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
                        );
                        if let (Some(tag), Some(weapon)) = (selected_model, active_weapon.as_ref())
                        {
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

                    ui.separator();
                    if let Some(texture_action) =
                        model_textures_ui(ui, texture_cache, &model.textures)
                    {
                        action = Some(texture_action);
                    }
                });
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
        }

        action
    }
}

impl ModelsView {
    #[cfg(any())]
    fn model_grid_ui(&mut self, ui: &mut egui::Ui) -> Option<ViewAction> {
        const GAP: f32 = 10.0;
        const TARGET_CARD_WIDTH: f32 = 280.0;
        const IMAGE_ASPECT: f32 = 0.82;
        const FOOTER_HEIGHT: f32 = 38.0;

        let filter = self.model_filter.to_lowercase();
        let models = self
            .models
            .iter()
            .filter(|entry| {
                filter.is_empty()
                    || entry
                        .search_label(weapon_skin_for_model(&self.weapon_catalog, entry.tag))
                        .contains(&filter)
            })
            .cloned()
            .collect_vec();
        if models.is_empty() {
            ui.label(RichText::new("No matching model entities").italics());
            return None;
        }

        let available_width = ui.available_width().max(TARGET_CARD_WIDTH);
        let columns = ((available_width + GAP) / (TARGET_CARD_WIDTH + GAP))
            .floor()
            .max(1.0) as usize;
        let card_width = ((available_width - GAP * (columns.saturating_sub(1)) as f32)
            / columns as f32)
            .max(160.0);
        let image_height = card_width * IMAGE_ASPECT;
        let card_height = image_height + FOOTER_HEIGHT;
        let row_height = card_height + GAP;
        let row_count = models.len().div_ceil(columns);
        let mut action = None;
        let mut clicked = None;

        egui::ScrollArea::vertical()
            .id_salt("models_grid")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, row_count, |ui, rows| {
                for row in rows {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = GAP;
                        for entry in models.iter().skip(row * columns).take(columns) {
                            if !self.thumbnails.contains_key(&entry.tag) {
                                let thumbnail = ModelThumbnail::load(
                                    self.cache.clone(),
                                    &self.texture_cache,
                                    entry.tag,
                                );
                                self.thumbnails.insert(entry.tag, thumbnail);
                            }

                            let (rect, response) = ui
                                .allocate_exact_size(vec2(card_width, card_height), Sense::click());
                            let response = response.tag_context(entry.tag);
                            let selected = self.selected_model == Some(entry.tag);
                            let footer = egui::Rect::from_min_max(
                                pos2(rect.left(), rect.bottom() - FOOTER_HEIGHT),
                                rect.max,
                            );
                            let image_rect = egui::Rect::from_min_max(
                                rect.min,
                                pos2(rect.right(), footer.top()),
                            );
                            let painter = ui.painter_at(rect);
                            painter.rect_filled(image_rect, 3.0, Color32::from_rgb(89, 108, 150));
                            painter.rect_filled(footer, 0.0, Color32::from_rgb(7, 8, 10));

                            if let Some(Some(thumbnail)) = self.thumbnails.get(&entry.tag)
                                && let GeometryPreviewKind::Model(model) = &thumbnail.preview.kind
                                && let Some(wireframe) = &model.wireframe
                            {
                                let callback = ModelPaintCallback::new(
                                    thumbnail.gpu.clone(),
                                    &self.texture_cache,
                                    wireframe,
                                    model.preview_uv_transform(),
                                    None,
                                    DEFAULT_MODEL_YAW,
                                    0.05,
                                    0.9,
                                    egui::Vec2::ZERO,
                                    false,
                                    image_rect,
                                    ui.ctx().pixels_per_point(),
                                    ModelEnvironment::default(),
                                );
                                ui.painter()
                                    .add(Callback::new_paint_callback(image_rect, callback));
                            } else {
                                painter.text(
                                    image_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "Preview unavailable",
                                    egui::TextStyle::Small.resolve(ui.style()),
                                    Color32::GRAY,
                                );
                            }

                            painter.text(
                                pos2(footer.left() + 10.0, footer.center().y),
                                egui::Align2::LEFT_CENTER,
                                entry.tag.to_string(),
                                egui::TextStyle::Monospace.resolve(ui.style()),
                                Color32::WHITE,
                            );
                            painter.rect_stroke(
                                rect,
                                3.0,
                                Stroke::new(
                                    if selected || response.hovered() {
                                        2.0
                                    } else {
                                        1.0
                                    },
                                    if selected {
                                        Color32::WHITE
                                    } else {
                                        Color32::DARK_GRAY
                                    },
                                ),
                                egui::StrokeKind::Inside,
                            );

                            if response.clicked() {
                                clicked = Some(entry.tag);
                            }
                            if response.double_clicked() {
                                action = Some(ViewAction::OpenTag(entry.tag));
                            }
                        }
                    });
                }
            });

        if let Some(tag) = clicked {
            self.selected_model = Some(tag);
        }
        action
    }

    fn model_list_ui(&mut self, ui: &mut egui::Ui) -> Option<ViewAction> {
        let mut action = None;

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.model_filter);
        });

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
                    let detected_skin = weapon_skin_for_model(&self.weapon_catalog, entry.tag);
                    if !filter.is_empty()
                        && !entry.search_label(detected_skin).contains(filter.as_str())
                    {
                        continue;
                    }

                    let response = ui
                        .add(
                            egui::Button::selectable(
                                self.selected_model == Some(entry.tag),
                                entry.list_label(ui, detected_skin),
                            )
                            .wrap_mode(if detected_skin.is_some() {
                                egui::TextWrapMode::Wrap
                            } else {
                                egui::TextWrapMode::Truncate
                            }),
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
        format!("{} {}{identity}", self.label(), self.info.label).to_lowercase()
    }

    fn list_label(
        &self,
        ui: &egui::Ui,
        detected_skin: Option<(&ModelWeaponEntry, &ModelWeaponSkinEntry)>,
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

    let width = viewport.width().min(310.0) - 20.0;
    let height = (52.0 + weapon.slots.len() as f32 * 62.0).min(viewport.height() - 20.0);
    let position = pos2(
        viewport.right() - width - 10.0,
        viewport.bottom() - height - 10.0,
    );
    let mut changed = None;

    egui::Area::new(egui::Id::new(("weapon_mod_selector", selected_model.0)))
        .order(egui::Order::Foreground)
        .fixed_pos(position)
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(Color32::from_rgba_unmultiplied(10, 13, 18, 238))
                .stroke(Stroke::new(1.0, Color32::from_rgb(73, 82, 96)))
                .corner_radius(6)
                .inner_margin(10)
                .show(ui, |ui| {
                    ui.set_width(width - 20.0);
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

fn normalize_light_position(position: &mut [f32; 3]) {
    let length = position
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length <= 0.0001 {
        *position = ModelEnvironment::default().light_position;
    } else {
        position.iter_mut().for_each(|value| *value /= length);
    }
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
) -> egui::Rect {
    ui.horizontal(|ui| {
        ui.label(format!(
            "{} vertices, {} indices ({})",
            wireframe.vertex_count_total, wireframe.index_count_total, wireframe.position_format
        ));
        ui.checkbox(show_wireframe, "Wireframe");
        ui.checkbox(show_stickers, "Stickers");
        if let Some(show_default_mods) = show_default_mods {
            ui.checkbox(show_default_mods, "Default mods")
                .on_hover_text("Show authored empty-slot weapon meshes");
        }
        if ui.button("Reset view").clicked() {
            *yaw = DEFAULT_MODEL_YAW;
            *pitch = 0.05;
            *zoom = 1.0;
            *pan = vec2(0.0, 0.0);
        }
    });
    egui::CollapsingHeader::new("Lighting")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut environment.light_gizmo, "Orbit gizmo");
                if ui.button("Reset lighting").clicked() {
                    let defaults = ModelEnvironment::default();
                    environment.light_position = defaults.light_position;
                    environment.light_orbit_radius = defaults.light_orbit_radius;
                    environment.light_size = defaults.light_size;
                    environment.shadow_strength = defaults.shadow_strength;
                    environment.sun_intensity = defaults.sun_intensity;
                    environment.ambient_intensity = defaults.ambient_intensity;
                    environment.specular_ibl_intensity = defaults.specular_ibl_intensity;
                }
            });
            ui.label("Drag the yellow light on the orbit sphere, or edit its axes.");
            ui.add(
                egui::Slider::new(
                    &mut environment.light_position[0],
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Orbit X"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.light_position[1],
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Orbit Y"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.light_position[2],
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Orbit Z"),
            );
            ui.add(
                egui::Slider::new(
                    &mut environment.light_orbit_radius,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Orbit radius"),
            );
            ui.add(
                egui::Slider::new(&mut environment.light_size, MODEL_PARAMETER_RANGE.clone())
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
                    &mut environment.shadow_softness,
                    MODEL_PARAMETER_RANGE.clone(),
                )
                .text("Shadow softness"),
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
    egui::CollapsingHeader::new("Image")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
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
            ui.add(
                egui::Slider::new(&mut environment.gamma, MODEL_PARAMETER_RANGE.clone())
                    .text("Gamma"),
            );
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
        });
    ui.collapsing("Environment & effects", |ui| {
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
    });
    let preview_textures = wireframe_preview_textures(wireframe, textures);

    let available = ui.available_size();
    let size = vec2(available.x.max(320.0), available.y.clamp(320.0, 620.0));
    let (rect, response) = ui.allocate_exact_size(size, Sense::drag());

    let mut gizmo_light_position = environment.light_position;
    normalize_light_position(&mut gizmo_light_position);
    let gizmo_center = rect.center() + *pan;
    let gizmo_radius = rect
        .width()
        .min(rect.height())
        .mul_add(0.32 * environment.light_orbit_radius.abs(), 0.0)
        .max(24.0);
    let light_handle = gizmo_center
        + vec2(
            -gizmo_light_position[0] * gizmo_radius,
            -gizmo_light_position[1] * gizmo_radius,
        );
    let light_handle_radius = (6.0 * environment.light_size.abs().sqrt()).clamp(5.0, 12.0);
    let light_gizmo_response = environment.light_gizmo.then(|| {
        ui.interact(
            egui::Rect::from_center_size(light_handle, vec2(32.0, 32.0)),
            ui.id().with("model_light_orbit_gizmo"),
            Sense::drag(),
        )
        .on_hover_text("Drag light around mesh")
    });
    if let Some(gizmo) = &light_gizmo_response
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
        let z_sign = if environment.light_position[2] < 0.0 {
            -1.0
        } else {
            1.0
        };
        environment.light_position = [x, y, (1.0 - x * x - y * y).max(0.0).sqrt() * z_sign];
        ui.ctx().request_repaint();
    }

    let pointer_delta = ui.input(|i| i.pointer.delta());
    let dragging_light = light_gizmo_response
        .as_ref()
        .is_some_and(|gizmo| gizmo.dragged() || gizmo.hovered());

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
            camera_frame,
            *yaw,
            *pitch,
            *zoom,
            *pan,
            *show_stickers,
            rect,
            ui.ctx().pixels_per_point(),
            *environment,
        );
        ui.painter()
            .add(Callback::new_paint_callback(rect, callback));
        true
    } else {
        false
    };

    if environment.light_gizmo {
        let sphere_stroke = Stroke::new(1.0, Color32::from_white_alpha(90));
        painter.circle_stroke(gizmo_center, gizmo_radius, sphere_stroke);
        for (axis, color) in [
            (0, Color32::from_rgba_unmultiplied(245, 90, 90, 125)),
            (1, Color32::from_rgba_unmultiplied(90, 220, 120, 125)),
        ] {
            let points = (0..=64)
                .map(|step| {
                    let angle = step as f32 / 64.0 * std::f32::consts::TAU;
                    let (sin, cos) = angle.sin_cos();
                    if axis == 0 {
                        gizmo_center + vec2(cos * gizmo_radius, sin * gizmo_radius * 0.28)
                    } else {
                        gizmo_center + vec2(cos * gizmo_radius * 0.28, sin * gizmo_radius)
                    }
                })
                .collect::<Vec<_>>();
            painter.add(egui::Shape::line(points, Stroke::new(1.0, color)));
        }
        painter.line_segment(
            [gizmo_center, light_handle],
            Stroke::new(1.2, Color32::from_rgba_unmultiplied(255, 220, 105, 170)),
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
    }

    let Some(projected) =
        project_vertices(wireframe, camera_frame, *yaw, *pitch, *zoom, *pan, rect)
    else {
        return rect;
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

    let camera_text = format!(
        "Yaw {:+.1}°  Pitch {:+.1}°  Roll {:+.1}°",
        yaw.to_degrees(),
        pitch.to_degrees(),
        0.0_f32,
    );
    let text_position = rect.left_top() + vec2(10.0, 10.0);
    let text_galley = painter.layout_no_wrap(
        camera_text,
        egui::FontId::monospace(12.0),
        Color32::from_gray(225),
    );
    painter.rect_filled(
        egui::Rect::from_min_size(
            text_position - vec2(5.0, 4.0),
            text_galley.size() + vec2(10.0, 8.0),
        ),
        3.0,
        Color32::from_black_alpha(165),
    );
    painter.galley(text_position, text_galley, Color32::WHITE);
    rect
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
            .unwrap_or_else(|| texture_cache.get_or_default(key.color).1);
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
        color: triangle.texture.or(fallback_texture)?,
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
    use super::*;

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
        };

        assert_eq!(weapon_index_for_model(&catalog, shared), None);
        assert!(weapon_skin_for_model(&catalog, shared).is_none());
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
