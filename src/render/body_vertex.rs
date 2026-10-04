//! Descriptor and raw IA contract for the audited body VS80A60DAD.

pub(crate) struct BodyVertexInputs<'a> {
    pub skinning: &'a wgpu::Buffer,
    pub view: &'a wgpu::Buffer,
    pub positions: &'a wgpu::Buffer,
    pub frames: &'a wgpu::Buffer,
    pub previous_positions: &'a wgpu::Buffer,
}

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..5)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: if binding < 2 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: true }
                },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(match binding {
                    0 => 31 * 16,
                    1 => 27 * 16,
                    _ => 4,
                }),
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("authored body VS dense ABI"),
        entries: &entries,
    })
}

/// Buffer identities stay fixed across frames. Current/previous ownership and
/// word bases are explicit runtime inputs, not inferred from buffer aliases.
pub(crate) fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    inputs: BodyVertexInputs<'_>,
) -> wgpu::BindGroup {
    let buffers = [
        inputs.skinning,
        inputs.view,
        inputs.positions,
        inputs.frames,
        inputs.previous_positions,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("authored body VS explicit runtime inputs"),
        layout,
        entries: &entries,
    })
}

/// Raw layout7 IA shared by the audited body/head vertex programs. Program
/// selection still belongs to the payload registry, never to this layout.
pub(crate) fn layout7_vertex_state(module: &wgpu::ShaderModule) -> wgpu::VertexState<'_> {
    const STREAM0: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Snorm16x4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Snorm16x4,
            offset: 8,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Snorm16x4,
            offset: 16,
            shader_location: 2,
        },
    ];
    const STREAM1: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Snorm16x2,
        offset: 0,
        shader_location: 3,
    }];
    wgpu::VertexState {
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        buffers: &[
            wgpu::VertexBufferLayout {
                array_stride: 24,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &STREAM0,
            },
            wgpu::VertexBufferLayout {
                array_stride: 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &STREAM1,
            },
        ],
    }
}
