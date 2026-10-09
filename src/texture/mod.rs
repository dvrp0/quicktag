pub mod cache;
mod capture;
mod dxgi;
mod headers_pc;
mod headers_ps;
mod headers_xbox;
mod metadata;
mod swizzle;
pub use capture::capture_texture;
pub use metadata::TextureExpressionMetadata;

use anyhow::Context;
use binrw::BinReaderExt;
use dxgi::{GcmSurfaceFormat, GcnSurfaceFormat};
use eframe::egui_wgpu::RenderState;
use eframe::wgpu;
use eframe::wgpu::TextureDimension;
use eframe::wgpu::util::DeviceExt;
use headers_pc::TextureHeaderPC;
use headers_ps::{TextureHeaderD2Ps4, TextureHeaderPs3, TextureHeaderRoiPs4};
use headers_xbox::{TextureHeaderDevAlphaX360, TextureHeaderRoiXbox};
use image::{DynamicImage, GenericImageView};
use swizzle::Deswizzler;
use swizzle::swizzle_ps::{GcmDeswizzler, GcnDeswizzler};
use swizzle::swizzle_xbox::XenosDetiler;
use tiger_pkg::version::EngineVersion;
use tiger_pkg::{DestinyVersion, MarathonVersion, package_manager};
use tiger_pkg::{GameVersion, TagHash, package::PackagePlatform};

#[derive(Debug, Clone)]
pub struct TextureHeaderGeneric {
    pub data_size: u32,
    pub format: wgpu::TextureFormat,
    pub width: u16,
    pub height: u16,
    pub depth: u16,
    pub array_size: u16,
    pub large_buffer: Option<TagHash>,

    pub deswizzle: bool,
    pub psformat: Option<GcnSurfaceFormat>,
    pub expression_metadata: Option<TextureExpressionMetadata>,
}

impl TryFrom<TextureHeaderD2Ps4> for TextureHeaderGeneric {
    type Error = anyhow::Error;

    fn try_from(v: TextureHeaderD2Ps4) -> Result<Self, Self::Error> {
        Ok(TextureHeaderGeneric {
            data_size: v.data_size,
            format: v.format.to_wgpu()?,
            width: v.width,
            height: v.height,
            depth: v.depth,
            array_size: v.array_size,
            large_buffer: v.large_buffer,

            deswizzle: (v.flags1 & 0xc00) != 0x400,
            psformat: Some(v.format),
            expression_metadata: None,
        })
    }
}

impl TryFrom<TextureHeaderPC> for TextureHeaderGeneric {
    type Error = anyhow::Error;

    fn try_from(v: TextureHeaderPC) -> Result<Self, Self::Error> {
        Ok(TextureHeaderGeneric {
            data_size: v.data_size,
            format: v.format.to_wgpu()?,
            width: v.width,
            height: v.height,
            depth: v.depth,
            array_size: v.array_size,
            large_buffer: v.large_buffer,

            deswizzle: false,
            psformat: None,
            expression_metadata: v.has_tiling_params.then_some(TextureExpressionMetadata {
                tiling_params: v.tiling_params,
                tile_count: v.tile_count,
            }),
        })
    }
}

pub struct Texture {
    pub view: wgpu::TextureView,
    pub handle: wgpu::Texture,
    preview_2d_texture: Option<wgpu::Texture>,
    pub full_cubemap_texture: Option<wgpu::Texture>,
    pub aspect_ratio: f32,
    pub desc: TextureDesc,

    pub comment: Option<String>,
}

pub struct TextureDesc {
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    /// Should the alpha channel be pre-multiplied on creation?
    pub premultiply_alpha: bool,
}

/// Return the colour-space view compatible with an 8-bit colour texture.
///
/// Tiger resources do not consistently encode material intent in the DXGI
/// format: the same BC/RGBA storage family can be albedo in one shader and a
/// packed control map in another. Keep both compatible views available so the
/// material binding, rather than the container header, decides whether the GPU
/// performs sRGB decoding.
pub(crate) fn srgb_texture_format(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            wgpu::TextureFormat::Rgba8UnormSrgb
        }
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            wgpu::TextureFormat::Bgra8UnormSrgb
        }
        wgpu::TextureFormat::Bc1RgbaUnorm | wgpu::TextureFormat::Bc1RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc1RgbaUnormSrgb
        }
        wgpu::TextureFormat::Bc2RgbaUnorm | wgpu::TextureFormat::Bc2RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc2RgbaUnormSrgb
        }
        wgpu::TextureFormat::Bc3RgbaUnorm | wgpu::TextureFormat::Bc3RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc3RgbaUnormSrgb
        }
        wgpu::TextureFormat::Bc7RgbaUnorm | wgpu::TextureFormat::Bc7RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc7RgbaUnormSrgb
        }
        _ => format,
    }
}

pub(crate) fn linear_texture_format(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
            wgpu::TextureFormat::Rgba8Unorm
        }
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            wgpu::TextureFormat::Bgra8Unorm
        }
        wgpu::TextureFormat::Bc1RgbaUnorm | wgpu::TextureFormat::Bc1RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc1RgbaUnorm
        }
        wgpu::TextureFormat::Bc2RgbaUnorm | wgpu::TextureFormat::Bc2RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc2RgbaUnorm
        }
        wgpu::TextureFormat::Bc3RgbaUnorm | wgpu::TextureFormat::Bc3RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc3RgbaUnorm
        }
        wgpu::TextureFormat::Bc7RgbaUnorm | wgpu::TextureFormat::Bc7RgbaUnormSrgb => {
            wgpu::TextureFormat::Bc7RgbaUnorm
        }
        _ => format,
    }
}

fn compatible_view_formats(format: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
    let mut formats = vec![format];
    for candidate in [linear_texture_format(format), srgb_texture_format(format)] {
        if !formats.contains(&candidate) {
            formats.push(candidate);
        }
    }
    formats
}

fn compressed_format_with_alpha(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bc1RgbaUnorm
            | wgpu::TextureFormat::Bc1RgbaUnormSrgb
            | wgpu::TextureFormat::Bc2RgbaUnorm
            | wgpu::TextureFormat::Bc2RgbaUnormSrgb
            | wgpu::TextureFormat::Bc3RgbaUnorm
            | wgpu::TextureFormat::Bc3RgbaUnormSrgb
            | wgpu::TextureFormat::Bc7RgbaUnorm
            | wgpu::TextureFormat::Bc7RgbaUnormSrgb
    )
}

fn mip_level_byte_size(format: wgpu::TextureFormat, width: u32, height: u32, depth: u32) -> usize {
    let extent = wgpu::Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    }
    .physical_size(format);
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap_or(4);
    ((extent.width / block_width) * (extent.height / block_height) * depth.max(1) * block_size)
        as usize
}

/// Infer how much of the tightly packed Tiger mip chain is present.
///
/// Large-buffer textures append the small-buffer tail in `load_data_d2`.
/// Previously that payload was fetched and then silently discarded by a
/// one-level GPU descriptor. Keep a conservative prefix only: incomplete next
/// levels never become a declared GPU mip.
fn available_mip_level_count(desc: &TextureDesc, data_len: usize) -> u32 {
    let layers = desc.array_size.max(1) as usize;
    let max_dimension = desc.width.max(desc.height).max(1);
    let theoretical_levels = u32::BITS - max_dimension.leading_zeros();
    let mut per_layer_bytes = 0usize;
    let mut available = 0u32;
    for level in 0..theoretical_levels {
        per_layer_bytes = per_layer_bytes.saturating_add(mip_level_byte_size(
            desc.format,
            desc.width.checked_shr(level).unwrap_or(0).max(1),
            desc.height.checked_shr(level).unwrap_or(0).max(1),
            desc.depth.checked_shr(level).unwrap_or(0).max(1),
        ));
        if per_layer_bytes.saturating_mul(layers) > data_len {
            break;
        }
        available = level + 1;
    }
    available.max(1)
}

impl TextureDesc {
    pub fn info(&self) -> String {
        let cubemap = if self.array_size == 6 {
            " (cubemap)"
        } else {
            ""
        };
        format!(
            "{}x{}x{} {:?}{cubemap}",
            self.width, self.height, self.depth, self.format
        )
    }

    pub fn kind(&self) -> TextureType {
        if self.array_size == 6 {
            TextureType::TextureCube
        } else if self.depth > 1 {
            TextureType::Texture3D
        } else {
            TextureType::Texture2D
        }
    }
}

impl Texture {
    /// Conservative GPU-residency estimate used by the shared texture cache.
    ///
    /// TextureDesc does not retain the exact uploaded mip count, so budget the
    /// complete theoretical mip chain. This intentionally overestimates sparse
    /// Tiger mip tails a little rather than allowing the cache to grow until
    /// the OS/GPU driver starts paging or OOMs.
    pub(crate) fn estimated_gpu_bytes(&self) -> u64 {
        let mut width = self.desc.width.max(1);
        let mut height = self.desc.height.max(1);
        let mut depth = self.desc.depth.max(1);
        let mut per_layer = 0u64;
        loop {
            per_layer =
                per_layer.saturating_add(
                    mip_level_byte_size(self.desc.format, width, height, depth) as u64,
                );
            if width == 1 && height == 1 && depth == 1 {
                break;
            }
            width = (width / 2).max(1);
            height = (height / 2).max(1);
            depth = (depth / 2).max(1);
        }

        // create_texture() retains one ordinary 2D or 3D allocation. Array
        // textures additionally retain the full array/cubemap allocation.
        let retained_layers = 1u64
            + self
                .full_cubemap_texture
                .as_ref()
                .map_or(0, |_| u64::from(self.desc.array_size.max(1)));
        let preview_bytes = self.preview_2d_texture.as_ref().map_or(0, |_| {
            mip_level_byte_size(self.desc.format, self.desc.width, self.desc.height, 1) as u64
        });
        per_layer
            .saturating_mul(retained_layers)
            .saturating_add(preview_bytes)
    }

    /// Raw texel view for UI presentation and extraction.
    ///
    /// Quicktag's Windows UI target is non-sRGB. Sampling an sRGB view there
    /// decodes the texels without a matching display encode, making previews
    /// darker and changing saturation. A format-compatible linear view keeps
    /// the stored channel values identical to a direct package decode.
    pub(crate) fn raw_view(&self) -> wgpu::TextureView {
        self.preview_2d_texture
            .as_ref()
            .unwrap_or(&self.handle)
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(linear_texture_format(self.desc.format)),
                ..Default::default()
            })
    }

    /// Reuse a successfully validated D2/Marathon descriptor without retaining pixels.
    pub(crate) fn validated_descriptor_d2(hash: TagHash) -> anyhow::Result<TextureHeaderGeneric> {
        metadata::descriptor(hash)
    }

    pub fn load_data_d2(
        hash: TagHash,
        load_full_mip: bool,
    ) -> anyhow::Result<(TextureHeaderGeneric, Vec<u8>, String)> {
        let texture_header_ref = package_manager()
            .get_entry(hash)
            .context("Texture header entry not found")?
            .reference;

        let header_data = package_manager()
            .read_tag(hash)
            .context("Failed to read texture header")?;

        let is_prebl = matches!(package_manager().version, GameVersion::Destiny(v) if v.is_prebl());

        let mut cur = std::io::Cursor::new(header_data);
        let texture: TextureHeaderGeneric = match package_manager().platform {
            PackagePlatform::PS4 => {
                let texheader: TextureHeaderD2Ps4 = cur.read_le_args((is_prebl,))?;
                TextureHeaderGeneric::try_from(texheader)?
            }
            PackagePlatform::Win64 => {
                let texheader: TextureHeaderPC = cur.read_le_args((is_prebl,))?;
                TextureHeaderGeneric::try_from(texheader)?
            }
            _ => unreachable!("Unsupported platform for D2 textures"),
        };
        let mut texture_data = if let Some(t) = texture.large_buffer {
            package_manager()
                .read_tag(t)
                .context("Failed to read texture data")?
        } else {
            package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read texture data")?
                .to_vec()
        };

        if load_full_mip && texture.large_buffer.is_some() {
            let ab = package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read large texture buffer")?
                .to_vec();

            texture_data.extend(ab);
        }

        let comment = format!("{texture:#X?}");

        match package_manager().platform {
            PackagePlatform::PS4 => {
                if texture.psformat.is_none() {
                    anyhow::bail!("Texture data not found: psformat: {:?}", texture.psformat);
                }
                let psformat = texture.psformat.unwrap();
                let expected_size =
                    (texture.width as usize * texture.height as usize * psformat.bpp()) / 8;

                if texture_data.len() < expected_size {
                    anyhow::bail!(
                        "Texture data size mismatch for {hash} ({}x{}x{} {:?}): expected {expected_size}, got {}",
                        texture.width,
                        texture.height,
                        texture.depth,
                        texture.format,
                        texture_data.len()
                    );
                }

                if texture.deswizzle {
                    let unswizzled = GcnDeswizzler
                        .deswizzle(
                            &texture_data,
                            texture.width as usize,
                            texture.height as usize,
                            if texture.array_size > 1 {
                                texture.array_size as usize
                            } else {
                                texture.depth as usize
                            },
                            texture.psformat.unwrap(),
                            false,
                        )
                        .context("Failed to deswizzle texture")?;
                    Ok((texture, unswizzled, comment))
                } else {
                    Ok((texture, texture_data, comment))
                }
            }
            _ => Ok((texture, texture_data, comment)),
        }
    }

    pub fn load_data_roi_ps4(
        hash: TagHash,
        _load_full_mip: bool,
    ) -> anyhow::Result<(TextureHeaderRoiPs4, Vec<u8>, String)> {
        let texture_header_ref = package_manager()
            .get_entry(hash)
            .context("Texture header entry not found")?
            .reference;

        let texture: TextureHeaderRoiPs4 = package_manager().read_tag_binrw(hash)?;

        let large_buffer = package_manager()
            .get_entry(texture_header_ref)
            .map(|v| TagHash(v.reference))
            .unwrap_or_default();

        let texture_data = if large_buffer.is_some() {
            package_manager()
                .read_tag(large_buffer)
                .context("Failed to read texture data")?
        } else {
            package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read texture data")?
                .to_vec()
        };

        let expected_size =
            (texture.width as usize * texture.height as usize * texture.format.bpp()) / 8;

        if texture_data.len() < expected_size {
            anyhow::bail!(
                "Texture data size mismatch for {hash} ({}x{}x{} {:?}): expected {expected_size}, got {}",
                texture.width,
                texture.height,
                texture.depth,
                texture.format,
                texture_data.len()
            );
        }

        let comment = format!("{texture:#X?}");
        if (texture.flags1 & 0xF00) != 0x500 {
            let unswizzled = GcnDeswizzler
                .deswizzle(
                    &texture_data,
                    texture.width as usize,
                    texture.height as usize,
                    if texture.array_size > 1 {
                        texture.array_size as usize
                    } else {
                        texture.depth as usize
                    },
                    texture.format,
                    true,
                )
                .context("Failed to deswizzle texture")?;

            Ok((texture, unswizzled, comment))
        } else {
            Ok((texture, texture_data, comment))
        }
    }

    pub fn load_data_devalpha_x360(
        hash: TagHash,
        _load_full_mip: bool,
    ) -> anyhow::Result<(TextureHeaderDevAlphaX360, Vec<u8>, String)> {
        let texture_header_ref = package_manager()
            .get_entry(hash)
            .context("Texture header entry not found")?
            .reference;

        let texture: TextureHeaderDevAlphaX360 = package_manager().read_tag_binrw(hash)?;

        let large_buffer = package_manager()
            .get_entry(texture_header_ref)
            .map(|v| TagHash(v.reference))
            .unwrap_or_default();

        let texture_data = if large_buffer.is_some() {
            package_manager()
                .read_tag(large_buffer)
                .context("Failed to read texture data")?
        } else {
            package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read texture data")?
                .to_vec()
        };

        let expected_size =
            (texture.width as usize * texture.height as usize * texture.format.bpp() as usize) / 8;

        if texture_data.len() < expected_size {
            anyhow::bail!(
                "Texture data size mismatch for {hash} ({}x{}x{} {:?}): expected {expected_size}, got {}",
                texture.width,
                texture.height,
                texture.depth,
                texture.format,
                texture_data.len()
            );
        }

        let comment = format!("{texture:#X?}");

        let untiled = XenosDetiler
            .deswizzle(
                &texture_data,
                texture.width as usize,
                texture.height as usize,
                if texture.array_size > 1 {
                    texture.array_size as usize
                } else {
                    texture.depth as usize
                },
                texture.format,
                false,
            )
            .context("Failed to deswizzle texture")?;

        Ok((texture, untiled, comment))
    }

    pub fn load_data_roi_xone(
        hash: TagHash,
        _load_full_mip: bool,
    ) -> anyhow::Result<(TextureHeaderRoiXbox, Vec<u8>, String)> {
        let texture_header_ref = package_manager()
            .get_entry(hash)
            .context("Texture header entry not found")?
            .reference;

        let texture: TextureHeaderRoiXbox = package_manager().read_tag_binrw(hash)?;

        let large_buffer = package_manager()
            .get_entry(texture_header_ref)
            .map(|v| TagHash(v.reference))
            .unwrap_or_default();

        let texture_data = if large_buffer.is_some() {
            package_manager()
                .read_tag(large_buffer)
                .context("Failed to read texture data")?
        } else {
            package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read texture data")?
                .to_vec()
        };

        let expected_size =
            (texture.width as usize * texture.height as usize * texture.format.bpp()) / 8;

        if texture_data.len() < expected_size {
            anyhow::bail!(
                "Texture data size mismatch for {hash} ({}x{}x{} {:?}): expected {expected_size}, got {}",
                texture.width,
                texture.height,
                texture.depth,
                texture.format,
                texture_data.len()
            );
        }

        let comment = format!("{texture:#X?}");
        if texture.tile_mode != 8 {
            let unswizzled = swizzle::swizzle_xbox::DurangoDeswizzler
                .deswizzle(
                    &texture_data,
                    texture.width as usize,
                    texture.height as usize,
                    texture.depth as usize,
                    (texture.format, texture.tile_mode),
                    true,
                )
                .context("Failed to deswizzle texture")?;
            Ok((texture, unswizzled, comment))
        } else {
            Ok((texture, texture_data, comment))
        }

        // let mut untiled = vec![];
        // swizzle::xbox::untile(
        //     &texture_data,
        //     &mut untiled,
        //     texture.width as usize,
        //     texture.height as usize,
        //     texture.format,
        // );
        // Ok((texture, texture_data, comment))
    }

    pub fn load_data_ps3_ttk(
        hash: TagHash,
        _load_full_mip: bool,
    ) -> anyhow::Result<(TextureHeaderPs3, Vec<u8>, String)> {
        let texture_header_ref = package_manager()
            .get_entry(hash)
            .context("Texture header entry not found")?
            .reference;

        let texture: TextureHeaderPs3 = package_manager().read_tag_binrw(hash)?;

        let large_buffer = package_manager()
            .get_entry(texture_header_ref)
            .map(|v| TagHash(v.reference))
            .unwrap_or_default();

        let texture_data = if large_buffer.is_some() {
            package_manager()
                .read_tag(large_buffer)
                .context("Failed to read texture data")?
        } else {
            package_manager()
                .read_tag(texture_header_ref)
                .context("Failed to read texture data")?
                .to_vec()
        };

        let expected_size =
            (texture.width as usize * texture.height as usize * texture.format.bpp()) / 8;

        if texture_data.len() < expected_size {
            anyhow::bail!(
                "Texture data size mismatch for {hash} ({}x{}x{} {:?}): expected {expected_size}, got {}",
                texture.width,
                texture.height,
                texture.depth,
                texture.format,
                texture_data.len()
            );
        }

        let mut data = texture_data.clone();
        let comment = format!("{texture:#X?}");

        if (texture.format as u8 & 0x20) != 0
            || (texture.format == GcmSurfaceFormat::A8R8G8B8 && texture.flags1 == 0)
            || texture.format == GcmSurfaceFormat::B8
        {
            data = GcmDeswizzler
                .deswizzle(
                    &texture_data,
                    texture.width as usize,
                    texture.height as usize,
                    if texture.array_size > 1 {
                        texture.array_size as usize
                    } else {
                        texture.depth as usize
                    },
                    texture.format,
                    false,
                )
                .context("Failed to deswizzle texture")?;
        }

        let unswizzled = GcmDeswizzler::color_deswizzle(&data, texture.format);
        Ok((texture, unswizzled, comment))
    }

    pub fn load_desc(hash: TagHash) -> anyhow::Result<TextureDesc> {
        match package_manager().version.engine_version() {
            EngineVersion::TigerD1Alpha
            | EngineVersion::TigerD1Indev
            | EngineVersion::TigerD1v1
            | EngineVersion::TigerD1v2 => match package_manager().platform {
                PackagePlatform::X360 => {
                    let texture: TextureHeaderDevAlphaX360 =
                        package_manager().read_tag_binrw(hash)?;
                    Ok(TextureDesc {
                        format: texture.format.to_wgpu()?,
                        width: texture.width as u32,
                        height: texture.height as u32,
                        array_size: texture.array_size as u32,
                        depth: texture.depth as u32,
                        premultiply_alpha: false,
                    })
                }
                PackagePlatform::PS3 => {
                    let texture: TextureHeaderPs3 = package_manager().read_tag_binrw(hash)?;
                    Ok(TextureDesc {
                        format: texture.format.to_wgpu()?,
                        width: texture.width as u32,
                        height: texture.height as u32,
                        array_size: texture.array_size as u32,
                        depth: texture.depth as u32,
                        premultiply_alpha: false,
                    })
                }
                PackagePlatform::PS4 => {
                    let texture: TextureHeaderRoiPs4 = package_manager().read_tag_binrw(hash)?;
                    Ok(TextureDesc {
                        format: texture.format.to_wgpu()?,
                        width: texture.width as u32,
                        height: texture.height as u32,
                        array_size: texture.array_size as u32,
                        depth: texture.depth as u32,
                        premultiply_alpha: false,
                    })
                }
                PackagePlatform::XboxOne => {
                    let texture: TextureHeaderRoiXbox = package_manager().read_tag_binrw(hash)?;
                    Ok(TextureDesc {
                        format: texture.format.to_wgpu()?,
                        width: texture.width as u32,
                        height: texture.height as u32,
                        array_size: texture.array_size as u32,
                        depth: texture.depth as u32,
                        premultiply_alpha: false,
                    })
                }
                _ => unreachable!("Unsupported platform for D1 textures"),
            },
            EngineVersion::TigerD2v1 | EngineVersion::TigerD2v2 | EngineVersion::TigerGoliath => {
                let is_prebl =
                    matches!(package_manager().version, GameVersion::Destiny(v) if v.is_prebl());
                match package_manager().platform {
                    PackagePlatform::PS4 => {
                        let header_data = package_manager()
                            .read_tag(hash)
                            .context("Failed to read texture header")?;

                        let mut cur = std::io::Cursor::new(header_data);
                        let texture: TextureHeaderD2Ps4 = cur.read_le_args((is_prebl,))?;

                        Ok(TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha: false,
                        })
                    }
                    PackagePlatform::Win64 => {
                        let header_data = package_manager()
                            .read_tag(hash)
                            .context("Failed to read texture header")?;

                        let mut cur = std::io::Cursor::new(header_data);
                        let texture: TextureHeaderPC = cur.read_le_args((is_prebl,))?;

                        Ok(TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha: false,
                        })
                    }
                    _ => unreachable!("Unsupported platform for D2 textures"),
                }
            }
        }
    }

    pub fn load(
        rs: &RenderState,
        hash: TagHash,
        premultiply_alpha: bool,
    ) -> anyhow::Result<Texture> {
        match package_manager().version.engine_version() {
            EngineVersion::TigerD1Alpha
            | EngineVersion::TigerD1Indev
            | EngineVersion::TigerD1v1
            | EngineVersion::TigerD1v2 => match package_manager().platform {
                PackagePlatform::X360 => {
                    let (texture, texture_data, comment) =
                        Self::load_data_devalpha_x360(hash, true)?;
                    Self::create_texture(
                        rs,
                        hash,
                        TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha,
                        },
                        texture_data,
                        Some(comment),
                    )
                }
                PackagePlatform::PS3 => {
                    let (texture, texture_data, comment) = Self::load_data_ps3_ttk(hash, true)?;
                    Self::create_texture(
                        rs,
                        hash,
                        TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha,
                        },
                        texture_data,
                        Some(comment),
                    )
                }
                PackagePlatform::PS4 => {
                    let (texture, texture_data, comment) = Self::load_data_roi_ps4(hash, true)?;
                    Self::create_texture(
                        rs,
                        hash,
                        TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha,
                        },
                        texture_data,
                        Some(comment),
                    )
                }
                PackagePlatform::XboxOne => {
                    // anyhow::bail!("Xbox One textures are not supported yet");
                    let (texture, texture_data, comment) = Self::load_data_roi_xone(hash, true)?;
                    Self::create_texture(
                        rs,
                        hash,
                        TextureDesc {
                            format: texture.format.to_wgpu()?,
                            width: texture.width as u32,
                            height: texture.height as u32,
                            depth: texture.depth as u32,
                            array_size: texture.array_size as u32,
                            premultiply_alpha,
                        },
                        texture_data,
                        Some(comment),
                    )
                }
                _ => unreachable!("Unsupported platform for RoI textures"),
            },
            EngineVersion::TigerD2v1 | EngineVersion::TigerD2v2 | EngineVersion::TigerGoliath => {
                let (texture, texture_data, comment) = Self::load_data_d2(hash, true)?;
                Self::create_texture(
                    rs,
                    hash,
                    TextureDesc {
                        format: texture.format,
                        width: texture.width as u32,
                        height: texture.height as u32,
                        depth: texture.depth as u32,
                        array_size: texture.array_size as u32,
                        premultiply_alpha,
                    },
                    texture_data,
                    Some(comment),
                )
            }
        }
    }

    /// Create a wgpu texture from unswizzled texture data
    fn create_texture(
        rs: &RenderState,
        hash: TagHash,
        desc: TextureDesc,
        // cohae: Take ownership of the data so we don't have to clone it for premultiplication
        mut data: Vec<u8>,
        comment: Option<String>,
    ) -> anyhow::Result<Texture> {
        // Pre-multiply alpha where possible
        if desc.premultiply_alpha
            && matches!(
                desc.format,
                wgpu::TextureFormat::Rgba8Unorm
                    | wgpu::TextureFormat::Rgba8UnormSrgb
                    | wgpu::TextureFormat::Bgra8Unorm
                    | wgpu::TextureFormat::Bgra8UnormSrgb
            )
        {
            for c in data.chunks_exact_mut(4) {
                c[0] = (c[0] as f32 * c[3] as f32 / 255.) as u8;
                c[1] = (c[1] as f32 * c[3] as f32 / 255.) as u8;
                c[2] = (c[2] as f32 * c[3] as f32 / 255.) as u8;
                // c[3] = c[3];
            }
        }

        let image_size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth.max(1),
        };

        {
            let block_size = desc.format.block_copy_size(None).unwrap_or(4);
            let (block_width, block_height) = desc.format.block_dimensions();
            let physical_size = image_size.physical_size(desc.format);
            let width_blocks = physical_size.width / block_width;
            let height_blocks = physical_size.height / block_height;

            let bytes_per_row = width_blocks * block_size;
            let expected_data_size =
                bytes_per_row * height_blocks * image_size.depth_or_array_layers;

            anyhow::ensure!(
                data.len() >= expected_data_size as usize,
                "Not enough data for texture {hash} ({}): expected 0x{:X}, got 0x{:X}",
                desc.info(),
                expected_data_size,
                data.len()
            );
        }

        let view_formats = compatible_view_formats(desc.format);
        let mip_level_count = available_mip_level_count(&desc, data.len());
        let handle = rs.device.create_texture_with_data(
            &rs.queue,
            &wgpu::TextureDescriptor {
                label: Some(&*format!("Texture {hash}")),
                size: wgpu::Extent3d { ..image_size },
                mip_level_count,
                sample_count: 1,
                dimension: if desc.depth > 1 {
                    TextureDimension::D3
                } else {
                    TextureDimension::D2
                },
                format: desc.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &view_formats,
            },
            wgpu::util::TextureDataOrder::default(),
            &data,
        );

        let view = handle.create_view(&wgpu::TextureViewDescriptor {
            ..Default::default()
        });

        // egui accepts only D2 user textures. Preserve its historical
        // first-slice preview without collapsing the shader-facing resource.
        let preview_2d_texture = (desc.depth > 1).then(|| {
            let byte_count = mip_level_byte_size(desc.format, desc.width, desc.height, 1);
            rs.device.create_texture_with_data(
                &rs.queue,
                &wgpu::TextureDescriptor {
                    label: Some(&format!("Texture {hash} (D3 preview slice)")),
                    size: wgpu::Extent3d {
                        width: desc.width,
                        height: desc.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: desc.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &view_formats,
                },
                wgpu::util::TextureDataOrder::default(),
                &data[..byte_count],
            )
        });

        if desc.premultiply_alpha
            && desc.array_size == 1
            && desc.depth == 1
            && compressed_format_with_alpha(desc.format)
        {
            let raw_view = handle.create_view(&wgpu::TextureViewDescriptor {
                format: Some(linear_texture_format(desc.format)),
                ..Default::default()
            });
            return Self::premultiply_compressed_texture(rs, hash, &raw_view, desc, comment);
        }

        let full_texture = if desc.array_size > 1 && desc.depth <= 1 {
            let handle = rs.device.create_texture_with_data(
                &rs.queue,
                &wgpu::TextureDescriptor {
                    label: Some(&*format!("Texture {hash} (full)")),
                    size: wgpu::Extent3d {
                        depth_or_array_layers: desc.array_size,
                        ..image_size
                    },
                    mip_level_count,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: desc.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &view_formats,
                },
                wgpu::util::TextureDataOrder::default(),
                &data,
            );

            Some(handle)
        } else {
            None
        };

        Ok(Texture {
            view,
            handle,
            preview_2d_texture,
            full_cubemap_texture: full_texture,
            aspect_ratio: desc.width as f32 / desc.height as f32,
            desc,
            comment,
        })
    }

    fn premultiply_compressed_texture(
        rs: &RenderState,
        hash: TagHash,
        source_view: &wgpu::TextureView,
        mut desc: TextureDesc,
        comment: Option<String>,
    ) -> anyhow::Result<Texture> {
        // UI blending happens in its encoded, non-sRGB target. Decode the BC
        // storage through a linear view and premultiply those stored values,
        // not gamma-decoded linear-light values.
        let output_format = wgpu::TextureFormat::Rgba8Unorm;
        let handle = rs.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Premultiplied texture {hash}")),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: output_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = handle.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = rs
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Texture alpha premultiplication"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    var output: VertexOutput;
    output.uv = vec2<f32>(select(0.0, 2.0, vertex == 1u), select(0.0, 2.0, vertex == 2u));
    output.position = vec4<f32>(output.uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source_texture, source_sampler, input.uv);
    return vec4<f32>(color.rgb * color.a, color.a);
}
"#
                    .into(),
                ),
            });
        let bind_group_layout =
            rs.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Texture alpha premultiplication bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let pipeline_layout = rs
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Texture alpha premultiplication pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = rs
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Texture alpha premultiplication pipeline"),
                layout: Some(&pipeline_layout),
                cache: None,
                multiview: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
            });
        let sampler = rs.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture alpha premultiplication sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind_group = rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture alpha premultiplication bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let mut encoder = rs
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Texture alpha premultiplication encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Texture alpha premultiplication pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        rs.queue.submit(Some(encoder.finish()));

        desc.format = output_format;
        Ok(Texture {
            view,
            handle,
            preview_2d_texture: None,
            full_cubemap_texture: None,
            aspect_ratio: desc.width as f32 / desc.height as f32,
            desc,
            comment: Some(format!(
                "{}\nGPU-premultiplied compressed alpha for UI",
                comment.unwrap_or_default()
            )),
        })
    }

    fn load_png(render_state: &RenderState, bytes: &[u8]) -> anyhow::Result<Texture> {
        let img = image::load_from_memory(bytes)?;
        let rgba = img.to_rgba8();
        let (width, height) = img.dimensions();
        Self::create_texture(
            render_state,
            TagHash::NONE,
            TextureDesc {
                format: wgpu::TextureFormat::Rgba8Unorm,
                width,
                height,
                array_size: 1,
                depth: 1,
                premultiply_alpha: true,
            },
            rgba.into_raw(),
            None,
        )
    }

    pub fn from_rgba8(
        render_state: &RenderState,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        comment: Option<String>,
    ) -> anyhow::Result<Texture> {
        Self::create_texture(
            render_state,
            TagHash::NONE,
            TextureDesc {
                format: wgpu::TextureFormat::Rgba8Unorm,
                width,
                height,
                array_size: 1,
                depth: 1,
                premultiply_alpha: false,
            },
            rgba,
            comment,
        )
    }

    pub fn to_image(&self, rs: &RenderState, layer: u32) -> anyhow::Result<DynamicImage> {
        let (rgba_data, padded_width, padded_height) = capture_texture(rs, self, layer)?;
        let image = image::RgbaImage::from_raw(padded_width, padded_height, rgba_data)
            .context("Failed to create image")?;

        Ok(DynamicImage::from(image).crop(0, 0, self.desc.width, self.desc.height))
    }
}

#[derive(PartialEq)]
pub enum TextureType {
    Texture2D,
    Texture3D,
    TextureCube,
}
