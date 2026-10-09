//! C107's six-buffer source ABI. RGBA8-UNORM is the explicit static viewer
//! color policy; shader arithmetic and geometry ownership remain authored.
pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..6).map(|binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: if binding < 4 { wgpu::BufferBindingType::Storage { read_only: true } }
                else { wgpu::BufferBindingType::Uniform },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(match binding {
                0 => 16, 4 => 27 * 16, 5 => 31 * 16, _ => 4,
            }),
        },
        count: None,
    }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("authored C107 vertex-color ABI"), entries: &entries,
    })
}

pub(crate) fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 6]) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers.iter().enumerate().map(|(binding, buffer)|
        wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("authored static vertex-color inputs"), layout, entries: &entries,
    })
}

/// Decode exactly the declared records, including any authored final sentinel.
/// The package header does not encode a DXGI format; this is the approved
/// RGBA8-UNORM static preview policy, not a claim about the game's view writer.
pub(crate) fn decode(bytes: &[u8], count: u32) -> Option<Vec<[f32; 4]>> {
    if count == 0 || bytes.len() != (count as usize).checked_mul(4)? { return None; }
    Some(bytes.chunks_exact(4).map(|rgba| std::array::from_fn(|i| rgba[i] as f32 / 255.0)).collect())
}
