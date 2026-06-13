use std::sync::Arc;

use eframe::egui::{
    self, Color32, RichText, Sense, Stroke,
    epaint::{Mesh, Vertex},
    pos2, vec2,
};
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::TagCache;
use tiger_pkg::{TagHash, manager::PackagePath, package::UEntryHeader, package_manager};

use crate::geometry::{
    GeometryPreviewKind, GeometryTagPreview, ModelTagInfo, ModelTagRole, UvTransformPreview,
    WireframePreview, model_info_for_reference,
};
use crate::gui::common::ResponseExt;
use crate::gui::tag::format_tag_entry;
use crate::texture::cache::TextureCache;
use crate::util::{format_file_size, ui_image_rotated};

use super::{View, ViewAction};

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
    preview_yaw: f32,
    preview_pitch: f32,
    preview_zoom: f32,
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
            preview_yaw: 0.4,
            preview_pitch: 0.25,
            preview_zoom: 1.0,
        }
    }

    pub fn set_cache(&mut self, cache: Arc<TagCache>) {
        self.cache = cache;
        self.preview = None;
        if let Some(tag) = self.selected_model {
            self.load_model(tag);
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
                    .any(|entry| model_info_for_reference(entry.reference).is_some())
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

        if let Some(tag) = self.models.first().map(|entry| entry.tag) {
            self.load_model(tag);
        }
    }

    fn load_model(&mut self, tag: TagHash) {
        self.selected_model = Some(tag);
        self.preview = None;

        let Some(entry) = package_manager().get_entry(tag) else {
            return;
        };
        let Ok(data) = package_manager().read_tag(tag) else {
            return;
        };

        let tag_type = TagType::from_type_subtype(entry.file_type, entry.file_subtype);
        self.preview = GeometryTagPreview::load(self.cache.clone(), tag, &entry, tag_type, &data);
    }
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
        let preview_yaw = &mut self.preview_yaw;
        let preview_pitch = &mut self.preview_pitch;
        let preview_zoom = &mut self.preview_zoom;

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
            });

            if let Some(mesh) = &model.mesh_source {
                ui.monospace(format!(
                    "{} mesh: ib={} vb0={} vb1={} index_start={} index_count={} prim={} lod={}",
                    mesh.kind,
                    mesh.index_buffer,
                    mesh.vertex0_buffer,
                    mesh.vertex1_buffer,
                    mesh.index_start,
                    mesh.index_count,
                    mesh.primitive_type,
                    mesh.lod_category
                ));

                if let Some(uv) = mesh.uv_transform {
                    ui.monospace(format!(
                        "uv: scale=[{:.6}, {:.6}] offset=[{:.6}, {:.6}]",
                        uv.scale[0], uv.scale[1], uv.offset[0], uv.offset[1]
                    ));
                } else {
                    ui.monospace("uv: default scale/offset");
                }

                for constant in &mesh.shader_constants {
                    ui.monospace(format!(
                        "shader constant {} = [{:.6}, {:.6}, {:.6}, {:.6}] ({})",
                        constant.name,
                        constant.value[0],
                        constant.value[1],
                        constant.value[2],
                        constant.value[3],
                        constant.source
                    ));
                }
            }

            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if let Some(wireframe) = &model.wireframe {
                        model_wireframe_ui(
                            ui,
                            wireframe,
                            model
                                .mesh_source
                                .as_ref()
                                .and_then(|mesh| mesh.uv_transform),
                            texture_cache,
                            &model.textures,
                            preview_yaw,
                            preview_pitch,
                            preview_zoom,
                        );
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
                    if !filter.is_empty()
                        && !entry
                            .search_label()
                            .to_lowercase()
                            .contains(filter.as_str())
                    {
                        continue;
                    }

                    let response = ui
                        .selectable_label(
                            self.selected_model == Some(entry.tag),
                            RichText::new(entry.label())
                                .monospace()
                                .color(entry.role_color()),
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

    fn search_label(&self) -> String {
        format!("{} {}", self.label(), self.info.label)
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

pub(super) fn model_wireframe_ui(
    ui: &mut egui::Ui,
    wireframe: &WireframePreview,
    uv_transform: Option<UvTransformPreview>,
    texture_cache: &TextureCache,
    textures: &[(TagHash, UEntryHeader)],
    yaw: &mut f32,
    pitch: &mut f32,
    zoom: &mut f32,
) {
    ui.horizontal(|ui| {
        ui.label(format!(
            "{} vertices, {} indices ({})",
            wireframe.vertex_count_total, wireframe.index_count_total, wireframe.position_format
        ));
        if ui.button("Reset view").clicked() {
            *yaw = 0.4;
            *pitch = 0.25;
            *zoom = 1.0;
        }
    });
    let preview_texture = textures
        .first()
        .map(|(tag, _entry)| (*tag, texture_cache.get_or_default(*tag)));
    if let Some((tag, (texture, _tid))) = &preview_texture {
        let uv_source = wireframe
            .uv_format
            .as_deref()
            .unwrap_or("fallback planar UVs");
        ui.monospace(format!(
            "textured preview: {} using {} ({}x{}, {:?})",
            tag, uv_source, texture.desc.width, texture.desc.height, texture.desc.format
        ));
    } else {
        ui.monospace("textured preview: no related texture resolved");
    }
    ui.monospace(format!(
        "source={} bounds [{:.3}, {:.3}, {:.3}] .. [{:.3}, {:.3}, {:.3}]",
        wireframe.source,
        wireframe.min[0],
        wireframe.min[1],
        wireframe.min[2],
        wireframe.max[0],
        wireframe.max[1],
        wireframe.max[2]
    ));

    let available = ui.available_size();
    let size = vec2(available.x.max(320.0), available.y.clamp(320.0, 620.0));
    let (rect, response) = ui.allocate_exact_size(size, Sense::drag());

    if response.dragged() {
        let delta = response.drag_delta();
        *yaw += delta.x * 0.01;
        *pitch = (*pitch + delta.y * 0.01).clamp(-1.45, 1.45);
        ui.ctx().request_repaint();
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            *zoom = (*zoom * (1.0 + scroll * 0.001)).clamp(0.05, 50.0);
            ui.ctx().request_repaint();
        }
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_rgb(12, 16, 20));
    painter.rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0, Color32::from_rgb(60, 70, 80)),
        egui::StrokeKind::Middle,
    );

    let Some(projected) = project_vertices(wireframe, *yaw, *pitch, *zoom, rect) else {
        return;
    };

    if wireframe.indices.len() >= 3 {
        if let Some((_tag, (_texture, texture_id))) = preview_texture {
            draw_textured_model_mesh(&painter, wireframe, &projected, texture_id, uv_transform);
        }

        let stroke = Stroke::new(0.7, Color32::from_rgb(140, 210, 255));
        for tri in wireframe.indices.chunks_exact(3).take(20_000) {
            let Some(a) = projected.get(tri[0] as usize).copied() else {
                continue;
            };
            let Some(b) = projected.get(tri[1] as usize).copied() else {
                continue;
            };
            let Some(c) = projected.get(tri[2] as usize).copied() else {
                continue;
            };
            painter.line_segment([a, b], stroke);
            painter.line_segment([b, c], stroke);
            painter.line_segment([c, a], stroke);
        }
    } else {
        for point in projected.iter().take(50_000) {
            painter.circle_filled(*point, 1.0, Color32::from_rgb(140, 210, 255));
        }
    }
}

fn draw_textured_model_mesh(
    painter: &egui::Painter,
    wireframe: &WireframePreview,
    projected: &[egui::Pos2],
    texture_id: egui::TextureId,
    uv_transform: Option<UvTransformPreview>,
) {
    let mut mesh = Mesh::with_texture(texture_id);
    let tint = Color32::from_rgba_premultiplied(255, 255, 255, 220);

    for tri in wireframe.indices.chunks_exact(3).take(20_000) {
        let base = mesh.vertices.len() as u32;
        for index in tri {
            let vertex_index = *index as usize;
            let Some(position) = wireframe.vertices.get(vertex_index).copied() else {
                continue;
            };
            let Some(screen_pos) = projected.get(vertex_index).copied() else {
                continue;
            };
            mesh.vertices.push(Vertex {
                pos: screen_pos,
                uv: preview_uv(vertex_index, position, wireframe, uv_transform),
                color: tint,
            });
        }

        if mesh.vertices.len() as u32 == base + 3 {
            mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        } else {
            mesh.vertices.truncate(base as usize);
        }
    }

    if !mesh.indices.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
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

    planar_preview_uv(position, wireframe, uv_transform)
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

fn planar_preview_uv(
    position: [f32; 3],
    wireframe: &WireframePreview,
    uv_transform: Option<UvTransformPreview>,
) -> egui::Pos2 {
    let extent_x = (wireframe.max[0] - wireframe.min[0]).abs().max(1.0);
    let extent_z = (wireframe.max[2] - wireframe.min[2]).abs();
    let extent_y = (wireframe.max[1] - wireframe.min[1]).abs().max(1.0);
    let mut u = (position[0] - wireframe.min[0]) / extent_x;
    let mut v = if extent_z > 0.0001 {
        (position[2] - wireframe.min[2]) / extent_z.max(1.0)
    } else {
        (position[1] - wireframe.min[1]) / extent_y
    };

    if let Some(uv) = uv_transform {
        u = u * uv.scale[0] + uv.offset[0];
        v = v * uv.scale[1] + uv.offset[1];
    }

    pos2(u.fract().abs(), v.fract().abs())
}

fn project_vertices(
    wireframe: &WireframePreview,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    rect: egui::Rect,
) -> Option<Vec<egui::Pos2>> {
    if wireframe.vertices.is_empty() {
        return None;
    }

    let center = [
        (wireframe.min[0] + wireframe.max[0]) * 0.5,
        (wireframe.min[1] + wireframe.max[1]) * 0.5,
        (wireframe.min[2] + wireframe.max[2]) * 0.5,
    ];
    let extent = [
        wireframe.max[0] - wireframe.min[0],
        wireframe.max[1] - wireframe.min[1],
        wireframe.max[2] - wireframe.min[2],
    ];
    let radius = extent.into_iter().fold(0.0_f32, f32::max).max(1.0);
    let scale = rect.width().min(rect.height()) * 0.42 * zoom / radius;
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let screen_center = rect.center();

    Some(
        wireframe
            .vertices
            .iter()
            .map(|position| {
                let x = position[0] - center[0];
                let y = position[1] - center[1];
                let z = position[2] - center[2];
                let xz = x * cy + z * sy;
                let zz = -x * sy + z * cy;
                let yz = y * cp - zz * sp;

                pos2(screen_center.x + xz * scale, screen_center.y - yz * scale)
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
