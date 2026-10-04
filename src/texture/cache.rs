use crate::{
    texture::{Texture, TextureType},
    util::{UiExt, ui_image_rotated},
};
use eframe::{
    egui::{Color32, Sense, TextureId, vec2},
    egui_wgpu::RenderState,
    wgpu,
};
use either::Either::{self, Left};
use image::RgbaImage;
use linked_hash_map::LinkedHashMap;
use parking_lot::RwLock;
use poll_promise::Promise;
use rustc_hash::FxHasher;
use std::{hash::BuildHasherDefault, rc::Rc, sync::Arc};
use tiger_pkg::TagHash;

pub type LoadedTexture = (Arc<Texture>, TextureId);

pub(crate) type TextureCacheKey = (TagHash, bool);

pub(crate) type TextureCacheMap = LinkedHashMap<
    TextureCacheKey,
    Either<Option<LoadedTexture>, Promise<Option<LoadedTexture>>>,
    BuildHasherDefault<FxHasher>,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialTextureKey {
    pub color: Option<TagHash>,
    pub normal: Option<TagHash>,
    pub emissive: Option<TagHash>,
    pub color_tint: [u8; 4],
    pub emissive_strength: u8,
}

impl MaterialTextureKey {
    pub fn has_composite_layers(self) -> bool {
        self.normal.is_some() || self.emissive.is_some()
    }
}

type MaterialTextureCacheMap =
    LinkedHashMap<MaterialTextureKey, Option<LoadedTexture>, BuildHasherDefault<FxHasher>>;

#[derive(Clone)]
pub struct TextureCache {
    pub render_state: RenderState,
    pub(crate) cache: Rc<RwLock<TextureCacheMap>>,
    pub(crate) material_cache: Rc<RwLock<MaterialTextureCacheMap>>,
    pub(crate) loading_placeholder: LoadedTexture,
}

impl TextureCache {
    pub fn new(render_state: RenderState) -> Self {
        let loading_placeholder =
            Texture::load_png(&render_state, include_bytes!("../../loading.png")).unwrap();

        let loading_placeholder_id = render_state.renderer.write().register_native_texture(
            &render_state.device,
            &loading_placeholder.view,
            wgpu::FilterMode::Linear,
        );

        Self {
            render_state,
            cache: Rc::new(RwLock::new(TextureCacheMap::default())),
            material_cache: Rc::new(RwLock::new(MaterialTextureCacheMap::default())),
            loading_placeholder: (Arc::new(loading_placeholder), loading_placeholder_id),
        }
    }

    pub fn is_loading_textures(&self) -> bool {
        self.cache
            .read()
            .iter()
            .any(|(_, v)| matches!(v, Either::Right(_)))
    }

    pub fn get_or_default(&self, hash: TagHash) -> LoadedTexture {
        self.get_or_load(hash)
            .unwrap_or_else(|| self.loading_placeholder.clone())
    }

    pub fn get_or_load(&self, hash: TagHash) -> Option<LoadedTexture> {
        self.get_or_load_with_alpha_mode(hash, true)
    }

    pub(crate) fn get_or_default_material(&self, hash: TagHash) -> LoadedTexture {
        self.get_or_load_material(hash)
            .unwrap_or_else(|| self.loading_placeholder.clone())
    }

    pub(crate) fn get_or_load_material(&self, hash: TagHash) -> Option<LoadedTexture> {
        self.get_or_load_with_alpha_mode(hash, false)
    }

    pub(crate) fn material_texture_failed(&self, hash: TagHash) -> bool {
        matches!(self.cache.read().get(&(hash, false)), Some(Either::Left(None)))
    }

    fn get_or_load_with_alpha_mode(
        &self,
        hash: TagHash,
        premultiply_alpha: bool,
    ) -> Option<LoadedTexture> {
        let key = (hash, premultiply_alpha);
        let mut cache = self.cache.write();

        let c = cache.remove(&key);

        let texture = if let Some(Either::Left(r)) = c {
            cache.insert(key, Left(r.clone()));
            r.clone()
        } else if let Some(Either::Right(p)) = c {
            if let std::task::Poll::Ready(r) = p.poll() {
                cache.insert(key, Left(r.clone()));
                r.clone()
            } else {
                cache.insert(key, Either::Right(p));
                None
            }
        } else if c.is_none() {
            let pending = cache
                .values()
                .filter(|value| matches!(value, Either::Right(_)))
                .count();
            if pending < Self::MAX_PENDING_TEXTURE_LOADS {
                cache.insert(
                    key,
                    Either::Right(Promise::spawn_async(Self::load_texture_task(
                        self.render_state.clone(),
                        hash,
                        premultiply_alpha,
                    ))),
                );
            }

            None
        } else {
            None
        };

        drop(cache);
        self.truncate();

        texture
    }

    pub(crate) async fn load_texture_task(
        render_state: RenderState,
        hash: TagHash,
        premultiply_alpha: bool,
    ) -> Option<LoadedTexture> {
        let texture = match Texture::load(&render_state, hash, premultiply_alpha) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Failed to load texture {hash}: {e}");
                return None;
            }
        };

        let raw_view = texture.raw_view();
        let id = render_state.renderer.write().register_native_texture(
            &render_state.device,
            &raw_view,
            wgpu::FilterMode::Linear,
        );
        Some((Arc::new(texture), id))
    }

    pub fn get_material_or_load(&self, key: MaterialTextureKey) -> Option<LoadedTexture> {
        if !key.has_composite_layers() {
            return self.get_or_load(key.color?);
        }

        {
            let mut cache = self.material_cache.write();
            if let Some(existing) = cache.remove(&key) {
                cache.insert(key, existing.clone());
                return existing;
            }
        }

        let color = self.get_or_load(key.color?)?;
        let normal = match key.normal {
            Some(tag) => self.get_or_load(tag),
            None => None,
        };
        if key.normal.is_some() && normal.is_none() {
            return None;
        }
        let emissive = match key.emissive {
            Some(tag) => self.get_or_load(tag),
            None => None,
        };
        if key.emissive.is_some() && emissive.is_none() {
            return None;
        }

        let material = self.compose_material_texture(key, color, normal, emissive)?;
        self.material_cache
            .write()
            .insert(key, Some(material.clone()));
        self.truncate_materials();
        Some(material)
    }

    fn compose_material_texture(
        &self,
        key: MaterialTextureKey,
        color: LoadedTexture,
        normal: Option<LoadedTexture>,
        emissive: Option<LoadedTexture>,
    ) -> Option<LoadedTexture> {
        let color_image = color.0.to_image(&self.render_state, 0).ok()?.to_rgba8();
        let normal_image = normal
            .as_ref()
            .and_then(|(texture, _id)| texture.to_image(&self.render_state, 0).ok())
            .map(|image| image.to_rgba8());
        let emissive_image = emissive
            .as_ref()
            .and_then(|(texture, _id)| texture.to_image(&self.render_state, 0).ok())
            .map(|image| image.to_rgba8());

        let (width, height) = color_image.dimensions();
        let mut out = vec![0u8; (width * height * 4) as usize];

        for y in 0..height {
            for x in 0..width {
                let base = color_image.get_pixel(x, y).0;
                let shade = normal_image
                    .as_ref()
                    .map(|image| normal_shade(sample_rgba(image, x, y, width, height)))
                    .unwrap_or(1.0);
                let emissive = emissive_image
                    .as_ref()
                    .map(|image| sample_rgba(image, x, y, width, height))
                    .unwrap_or([0, 0, 0, 255]);

                let offset = ((y * width + x) * 4) as usize;
                let emissive_strength = key.emissive_strength as f32 / 255.0;
                out[offset] = ((base[0] as f32 * shade * key.color_tint[0] as f32 / 255.0)
                    + emissive[0] as f32 * emissive_strength)
                    .clamp(0.0, 255.0) as u8;
                out[offset + 1] = ((base[1] as f32 * shade * key.color_tint[1] as f32 / 255.0)
                    + emissive[1] as f32 * emissive_strength)
                    .clamp(0.0, 255.0) as u8;
                out[offset + 2] = ((base[2] as f32 * shade * key.color_tint[2] as f32 / 255.0)
                    + emissive[2] as f32 * emissive_strength)
                    .clamp(0.0, 255.0) as u8;
                out[offset + 3] = 255;
            }
        }

        let texture = Texture::from_rgba8(
            &self.render_state,
            width,
            height,
            out,
            Some(format!(
                "composited material color={:?} normal={:?} emissive={:?} tint={:?} emissive_strength={}",
                key.color, key.normal, key.emissive, key.color_tint, key.emissive_strength
            )),
        )
        .ok()?;
        let raw_view = texture.raw_view();
        let id = self.render_state.renderer.write().register_native_texture(
            &self.render_state.device,
            &raw_view,
            wgpu::FilterMode::Linear,
        );

        Some((Arc::new(texture), id))
    }

    pub fn texture_preview(&self, hash: TagHash, ui: &mut eframe::egui::Ui) {
        if let Some((tex, egui_tex)) = self.get_or_load(hash) {
            let screen_size = ui.ctx().content_rect().size();
            let screen_aspect_ratio = screen_size.x / screen_size.y;
            let texture_aspect_ratio = tex.aspect_ratio;

            let max_size = if ui.input(|i| i.modifiers.ctrl) {
                screen_size * 0.70
            } else {
                ui.label("ℹ Hold ctrl to enlarge");
                screen_size * 0.30
            };

            let tex_size = if texture_aspect_ratio > screen_aspect_ratio {
                vec2(max_size.x, max_size.x / texture_aspect_ratio)
            } else {
                vec2(max_size.y * texture_aspect_ratio, max_size.y)
            };

            let (response, painter) = ui.allocate_painter(tex_size, Sense::hover());
            ui_image_rotated(
                &painter,
                egui_tex,
                response.rect,
                // Rotate the image if it's a cubemap
                if tex.desc.kind() == TextureType::TextureCube {
                    90.
                } else {
                    0.
                },
                tex.desc.kind() == TextureType::TextureCube,
            );

            ui.horizontal(|ui| {
                match tex.desc.kind() {
                    TextureType::Texture2D => ui.chip("2D", Color32::YELLOW, Color32::BLACK),
                    TextureType::TextureCube => ui.chip("Cube", Color32::BLUE, Color32::WHITE),
                    TextureType::Texture3D => ui.chip("3D", Color32::GREEN, Color32::BLACK),
                };

                ui.label(tex.desc.info());
            });
        }
    }

    // Count-only limits are unsafe for modern Tiger assets: a handful of 4K
    // RGBA textures can outweigh hundreds of small BC maps. Keep both a hard
    // entry ceiling and a conservative GPU-byte budget so browsing models
    // cannot accumulate several gigabytes of resident textures.
    pub(crate) const MAX_TEXTURES: usize = 256;
    pub(crate) const MAX_PENDING_TEXTURE_LOADS: usize = 8;
    pub(crate) const MAX_TEXTURE_GPU_BYTES: u64 = 384 * 1024 * 1024;

    pub(crate) fn truncate(&self) {
        let mut cache = self.cache.write();

        // A spawned Promise can already own a completed GPU texture even when
        // its key is never requested again (common while rapidly browsing
        // models). Promote completed promises before accounting so those
        // allocations cannot sit outside the byte budget indefinitely.
        for (_key, value) in cache.iter_mut() {
            let completed = match value {
                Either::Right(promise) => match promise.poll() {
                    std::task::Poll::Ready(result) => Some(result.clone()),
                    std::task::Poll::Pending => None,
                },
                Either::Left(_) => None,
            };
            if let Some(result) = completed {
                *value = Either::Left(result);
            }
        }

        let mut estimated_bytes = cache
            .values()
            .filter_map(|value| match value {
                Either::Left(Some((texture, _))) => Some(texture.estimated_gpu_bytes()),
                _ => None,
            })
            .fold(0u64, u64::saturating_add);

        while cache.len() > Self::MAX_TEXTURES || estimated_bytes > Self::MAX_TEXTURE_GPU_BYTES {
            let Some((_, value)) = cache.pop_front() else {
                break;
            };
            if let Either::Left(Some((texture, tid))) = value {
                estimated_bytes = estimated_bytes.saturating_sub(texture.estimated_gpu_bytes());
                self.render_state.renderer.write().free_texture(&tid);
            }
        }
    }

    pub(crate) const MAX_MATERIAL_TEXTURES: usize = 64;
    pub(crate) const MAX_MATERIAL_TEXTURE_GPU_BYTES: u64 = 128 * 1024 * 1024;

    pub(crate) fn truncate_materials(&self) {
        let mut cache = self.material_cache.write();
        let mut estimated_bytes = cache
            .values()
            .filter_map(|value| {
                value
                    .as_ref()
                    .map(|(texture, _)| texture.estimated_gpu_bytes())
            })
            .fold(0u64, u64::saturating_add);

        while cache.len() > Self::MAX_MATERIAL_TEXTURES
            || estimated_bytes > Self::MAX_MATERIAL_TEXTURE_GPU_BYTES
        {
            let Some((_, value)) = cache.pop_front() else {
                break;
            };
            if let Some((texture, tid)) = value {
                estimated_bytes = estimated_bytes.saturating_sub(texture.estimated_gpu_bytes());
                self.render_state.renderer.write().free_texture(&tid);
            }
        }
    }
}

fn sample_rgba(image: &RgbaImage, x: u32, y: u32, width: u32, height: u32) -> [u8; 4] {
    let sx = x
        .saturating_mul(image.width())
        .checked_div(width)
        .unwrap_or(0);
    let sy = y
        .saturating_mul(image.height())
        .checked_div(height)
        .unwrap_or(0);
    image
        .get_pixel(sx.min(image.width() - 1), sy.min(image.height() - 1))
        .0
}

fn normal_shade(normal: [u8; 4]) -> f32 {
    let nx = normal[0] as f32 / 127.5 - 1.0;
    let ny = normal[1] as f32 / 127.5 - 1.0;
    let nz = normal[2] as f32 / 127.5 - 1.0;
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(0.001);
    let dot = (nx / len * 0.35 + ny / len * -0.45 + nz / len * 0.82).clamp(-1.0, 1.0);
    (0.62 + dot.max(0.0) * 0.38).clamp(0.45, 1.15)
}
