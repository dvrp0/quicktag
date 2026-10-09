//! Dense descriptors for independently audited material programs.
//! Program identity selects the contract; resource counts never select a shader.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SurfacePixelAbi {
    Audited { rows: u64, textures: u32, volume: Option<u32>, cube: Option<u32>, samplers: u32 },
    Body,
    Layered,
    Hands,
    Hardware,
}

pub(crate) fn create_layout(device: &wgpu::Device, abi: SurfacePixelAbi) -> wgpu::BindGroupLayout {
    let cube_slot = match abi {
        SurfacePixelAbi::Audited { cube, .. } => cube,
        _ => None,
    };
    let (rows, texture_count, volume_slot, sampler_count, label) = match abi {
        SurfacePixelAbi::Audited { rows, textures, volume, samplers, .. } => (rows, textures, volume, samplers, "audited opaque PS resource ABI"),
        SurfacePixelAbi::Body => (125, 8, Some(7), 3, "authored body PS dense ABI"),
        SurfacePixelAbi::Layered => (129, 9, Some(8), 3, "authored chest/sleeve nine-texture PS ABI"),
        SurfacePixelAbi::Hands => (138, 9, Some(7), 3, "authored hands PS dense ABI"),
        SurfacePixelAbi::Hardware => (117, 7, Some(6), 3, "authored hardware PS dense ABI"),
    };
    let entries: Vec<_> = (0..3 + texture_count + sampler_count)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: if binding < 3 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new([rows, 31, 29][binding as usize] * 16),
                }
            } else if binding < 3 + texture_count {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: if Some(binding - 3) == volume_slot {
                        wgpu::TextureViewDimension::D3
                    } else if Some(binding - 3) == cube_slot {
                        wgpu::TextureViewDimension::Cube
                    } else {
                        wgpu::TextureViewDimension::D2
                    },
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub(crate) fn bind<'a>(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 3],
    textures: &[&wgpu::TextureView],
    samplers: impl AsRef<[&'a wgpu::Sampler]>,
) -> wgpu::BindGroup {
    let mut entries = Vec::with_capacity(6 + textures.len());
    entries.extend(
        buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            }),
    );
    entries.extend(
        textures
            .iter()
            .enumerate()
            .map(|(slot, view)| wgpu::BindGroupEntry {
                binding: 3 + slot as u32,
                resource: wgpu::BindingResource::TextureView(view),
            }),
    );
    entries.extend(
        samplers.as_ref()
            .iter()
            .enumerate()
            .map(|(slot, sampler)| wgpu::BindGroupEntry {
                binding: 3 + textures.len() as u32 + slot as u32,
                resource: wgpu::BindingResource::Sampler(sampler),
            }),
    );
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("authored dense PS explicit runtime inputs"),
        layout,
        entries: &entries,
    })
}
