//! Exact D0F2 pixel ABI and retained paired layout7 submission.
//! Runtime scopes, source normal, attachment formats and inherited depth/cull
//! states are explicit inputs; no compatibility shader enters this component.

pub(crate) struct DecalPixelInputs<'a> {
    pub globals: &'a wgpu::Buffer,
    pub view: &'a wgpu::Buffer,
    pub scene_normal: &'a wgpu::TextureView,
    pub textures: &'a [&'a wgpu::TextureView],
    pub samplers: &'a [&'a wgpu::Sampler],
}

#[derive(Clone, Copy)]
pub(crate) struct DecalPixelContract {
    pub constant_rows: usize,
    pub uses_view: bool,
    pub scene_normal: bool,
    pub texture_count: u32,
    pub sampler_count: u32,
    pub blend: u8,
}

impl DecalPixelContract {
    pub const D0F2: Self = Self {
        constant_rows: 9,
        uses_view: true,
        scene_normal: true,
        texture_count: 2,
        sampler_count: 1,
        blend: 26,
    };
}

pub(crate) struct DecalDrawPipeline {
    pixel_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    contract: DecalPixelContract,
    index_format: wgpu::IndexFormat,
}

impl DecalDrawPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        vertex_layout: &wgpu::BindGroupLayout,
        contract: DecalPixelContract,
        formats: [wgpu::TextureFormat; 4],
        front_face: wgpu::FrontFace,
        cull_mode: Option<wgpu::Face>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        assert!(matches!(contract.blend, 26 | 27));
        let buffer_count = 1 + u32::from(contract.uses_view);
        let texture_count = contract.texture_count + u32::from(contract.scene_normal);
        let entries: Vec<_> = (0..buffer_count + texture_count + contract.sampler_count)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: if binding < buffer_count {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(if binding == 0 {
                            contract.constant_rows as u64 * 16
                        } else {
                            29 * 16
                        }),
                    }
                } else if binding < buffer_count + texture_count {
                    wgpu::BindingType::Texture {
                        // t2 is Texture.Load only; FP32 native-MRT probes do not
                        // require FLOAT32_FILTERABLE or a sampling approximation.
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: !(contract.scene_normal && binding == buffer_count),
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }
                } else {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                },
                count: None,
            })
            .collect();
        let pixel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("authored D0F2 dense PS ABI"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("authored paired D0F2 pipeline ABI"),
            bind_group_layouts: &[&pixel_layout, vertex_layout],
            push_constant_ranges: &[],
        });
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::SrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let rgb = wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE;
        let masks = [
            rgb,
            rgb,
            if contract.blend == 26 {
                wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE
            } else {
                rgb
            },
            wgpu::ColorWrites::empty(),
        ];
        let targets: [_; 4] = std::array::from_fn(|i| {
            Some(wgpu::ColorTargetState {
                format: formats[i],
                blend: Some(blend),
                write_mask: masks[i],
            })
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("authored D0F2 VS/PS state26 independent MRT masks"),
            layout: Some(&layout),
            vertex: super::body_vertex::layout7_vertex_state(vs),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(index_format),
                front_face,
                cull_mode,
                ..Default::default()
            },
            depth_stencil,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: ps,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview: None,
            cache: None,
        });
        Self {
            pixel_layout,
            pipeline,
            contract,
            index_format,
        }
    }

    pub fn bind(&self, device: &wgpu::Device, inputs: DecalPixelInputs<'_>) -> wgpu::BindGroup {
        assert_eq!(inputs.textures.len(), self.contract.texture_count as usize);
        assert_eq!(inputs.samplers.len(), self.contract.sampler_count as usize);
        let mut resources = vec![inputs.globals.as_entire_binding()];
        if self.contract.uses_view {
            resources.push(inputs.view.as_entire_binding());
        }
        if self.contract.scene_normal {
            resources.push(wgpu::BindingResource::TextureView(inputs.scene_normal));
        }
        resources.extend(
            inputs
                .textures
                .iter()
                .map(|t| wgpu::BindingResource::TextureView(t)),
        );
        resources.extend(
            inputs
                .samplers
                .iter()
                .map(|s| wgpu::BindingResource::Sampler(s)),
        );
        let entries = resources
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect::<Vec<_>>();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("authored D0F2 explicit runtime resources"),
            layout: &self.pixel_layout,
            entries: &entries,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pixel: &wgpu::BindGroup,
        vertex: &wgpu::BindGroup,
        source: &wgpu::Buffer,
        uv: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        range: std::ops::Range<u32>,
    ) {
        assert!(
            range.start < range.end
                && u64::from(range.end)
                    * match self.index_format {
                        wgpu::IndexFormat::Uint16 => 2,
                        wgpu::IndexFormat::Uint32 => 4,
                    }
                    <= indices.size()
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, pixel, &[]);
        pass.set_bind_group(1, vertex, &[]);
        pass.set_vertex_buffer(0, source.slice(..));
        pass.set_vertex_buffer(1, uv.slice(..));
        pass.set_index_buffer(indices.slice(..), self.index_format);
        pass.draw_indexed(range, 0, 0..1);
    }
}
