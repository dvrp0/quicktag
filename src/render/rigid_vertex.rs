//! Exact direct-IA contract for VS80A9B7D7. No generated storage input.

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("authored rigid direct-IA VS uniforms"),
        entries: &std::array::from_fn::<_, 2, _>(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(if binding == 0 {240} else {432}),
            },
            count: None,
        }),
    })
}

pub(crate) fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
                  object: &wgpu::Buffer, view: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("authored rigid direct-IA VS bindings"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {binding:0,resource:object.as_entire_binding()},
            wgpu::BindGroupEntry {binding:1,resource:view.as_entire_binding()},
        ],
    })
}

pub(crate) fn vertex_state(module: &wgpu::ShaderModule) -> wgpu::VertexState<'_> {
    const GEOMETRY: [wgpu::VertexAttribute;3] = wgpu::vertex_attr_array![
        0 => Float32x4, 1 => Float32x4, 2 => Float32x4];
    const UV: [wgpu::VertexAttribute;1] = wgpu::vertex_attr_array![3 => Snorm16x2];
    wgpu::VertexState {
        module, entry_point:Some("main"), compilation_options:Default::default(),
        buffers:&[
            wgpu::VertexBufferLayout {array_stride:48,step_mode:wgpu::VertexStepMode::Vertex,attributes:&GEOMETRY},
            wgpu::VertexBufferLayout {array_stride:4,step_mode:wgpu::VertexStepMode::Vertex,attributes:&UV},
        ],
    }
}
