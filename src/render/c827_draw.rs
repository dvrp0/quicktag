//! Exact textureless C827 receiver multiplier. Viewer color is stored in both
//! HDR and albedo surfaces, so the same native RT0 operation is projected into
//! each color surface. Normal, properties and motion are never reinterpreted.
use std::ops::Range;

pub(crate) struct C827DrawPipeline {
    pub pixel_layout: wgpu::BindGroupLayout,
    pub vertex_layout: wgpu::BindGroupLayout,
    color: wgpu::RenderPipeline,
    albedo: Option<wgpu::RenderPipeline>,
    index_format: wgpu::IndexFormat,
}

pub(crate) struct C827DrawBindings {
    pub pixel: wgpu::BindGroup,
    pub vertex: wgpu::BindGroup,
    pub source: wgpu::Buffer,
    pub uv: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub range: Range<u32>,
}

impl C827DrawPipeline {
    pub fn new(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        color_formats: [wgpu::TextureFormat; 4],
        depth: Option<wgpu::DepthStencilState>,
        cull: Option<wgpu::Face>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        Self::build(device, vs, ps, color_formats, depth, cull, true, index_format)
    }

    pub fn new_material(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        formats: [wgpu::TextureFormat; 4],
        depth: Option<wgpu::DepthStencilState>,
        cull: Option<wgpu::Face>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        Self::build(device, vs, ps, formats, depth, cull, false, index_format)
    }

    fn build(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        color_formats: [wgpu::TextureFormat; 4],
        depth: Option<wgpu::DepthStencilState>,
        cull: Option<wgpu::Face>,
        viewer_projection: bool,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        let pixel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("C827 pixel ABI"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(9 * 16),
                },
                count: None,
            }],
        });
        let vertex_layout = super::body_vertex::create_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("C827 paired program ABI"),
            bind_group_layouts: &[&pixel_layout, &vertex_layout],
            push_constant_ranges: &[],
        });
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let create = |label, formats: [Option<wgpu::TextureFormat>; 4]| {
            let targets = std::array::from_fn::<_, 4, _>(|i| {
                formats[i].map(|format| wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask: if i == 0 {
                        wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE
                    } else {
                        wgpu::ColorWrites::empty()
                    },
                })
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: super::body_vertex::layout7_vertex_state(vs),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(index_format),
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: depth.clone(),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: ps,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview: None,
                cache: None,
            })
        };
        let color = create("C827 native RT0 to viewer HDR", color_formats.map(Some));
        let albedo = viewer_projection.then(|| {
            create(
                "C827 native RT0 to viewer albedo",
                [Some(color_formats[3]), None, None, None],
            )
        });
        Self {
            pixel_layout,
            vertex_layout,
            color,
            albedo,
            index_format,
        }
    }

    pub fn encode(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        bindings: &C827DrawBindings,
        albedo: bool,
    ) {
        let width = match self.index_format {
            wgpu::IndexFormat::Uint16 => 2,
            wgpu::IndexFormat::Uint32 => 4,
        };
        assert!(bindings.range.start < bindings.range.end
            && u64::from(bindings.range.end) * width <= bindings.indices.size(),
            "C827 strip exceeds authored index image");
        pass.set_pipeline(if albedo {
            self.albedo.as_ref().expect("viewer albedo projection")
        } else {
            &self.color
        });
        pass.set_bind_group(0, &bindings.pixel, &[]);
        pass.set_bind_group(1, &bindings.vertex, &[]);
        pass.set_vertex_buffer(0, bindings.source.slice(..));
        pass.set_vertex_buffer(1, bindings.uv.slice(..));
        pass.set_index_buffer(bindings.indices.slice(..), self.index_format);
        pass.draw_indexed(bindings.range.clone(), 0, 0..1);
    }
}
