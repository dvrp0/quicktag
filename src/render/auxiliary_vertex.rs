//! Shared seven-buffer ABI for source programs with a float4 auxiliary stream.
//! Callers supply authored data, or a bounded inert image only when the paired
//! PS proves TEXCOORD8 unused. cb0 retains its distinct source row contract.
//! Previous positions are explicit for the viewer-owned static pose.

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..7)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: if binding < 4 {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Uniform
                },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(match binding {
                    0 => 16,
                    // Smallest declared image of these exact source ABIs.
                    // Admission validates/pads the full 61/80/97/149-row image.
                    4 => 61 * 16,
                    5 => 27 * 16,
                    6 => 31 * 16,
                    _ => 4,
                }),
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("authored auxiliary vertex ABI"),
        entries: &entries,
    })
}

pub(crate) fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 7],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("retained static auxiliary vertex inputs"),
        layout,
        entries: &entries,
    })
}
