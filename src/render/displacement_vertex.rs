//! EC03 scalar static streams plus its exact authored displacement cbuffer.
//! No auxiliary/color stream exists in this source. CB0's unused tail is ABI
//! padding; the single loaded row is supplied by package-authored constants.

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..6).map(|binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: if matches!(binding, 0 | 1 | 5) {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only: true }
            },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(match binding {
                0 => 31 * 16,
                1 => 27 * 16,
                5 => 30 * 16,
                _ => 4,
            }),
        },
        count: None,
    }).collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("authored EC03 vertex ABI"), entries: &entries,
    })
}

pub(crate) fn bind(
    device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
    inputs: super::body_vertex::BodyVertexInputs<'_>, cb0: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffers = [inputs.skinning, inputs.view, inputs.positions,
                   inputs.frames, inputs.previous_positions, cb0];
    let entries: Vec<_> = buffers.iter().enumerate().map(|(binding, buffer)|
        wgpu::BindGroupEntry { binding: binding as u32,
            resource: buffer.as_entire_binding() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("retained EC03 authored static inputs"), layout, entries: &entries,
    })
}
