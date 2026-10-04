//! Exact global-light accumulation ABI. Runtime scopes and attachment ownership
//! are caller contracts; this pass produces diffuse/specular, not final shading.

pub(crate) struct GlobalLightPipeline {
    pixel_layout: wgpu::BindGroupLayout,
    vertex_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

pub(crate) struct GlobalLightInputs<'a> {
    /// cb0 (98 rows), PS cb12 (29), Frame cb13 (15).
    pub pixel: [&'a wgpu::Buffer; 3],
    /// cb0 (98 rows), VS cb12 (27); stage-specific images stay separate.
    pub vertex: [&'a wgpu::Buffer; 2],
    /// t0 material properties, t1 packed normal/class, t2 depth,
    /// t3 shadow mask and t4 shadow depth. Packaged TFX proves these sources.
    pub textures: [&'a wgpu::TextureView; 5],
    pub samplers: [&'a wgpu::Sampler; 2],
}

pub(crate) struct GlobalLightBindings {
    pixel: wgpu::BindGroup,
    vertex: wgpu::BindGroup,
}

fn uniform(binding: u32, rows: u64, stage: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stage,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(rows * 16),
        },
    }
}

impl GlobalLightPipeline {
    pub fn new(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        targets: [Option<wgpu::ColorTargetState>; 2],
    ) -> Self {
        // Integer loads and nearest sampling preserve FP32 controls without
        // pretending the native filtering/storage format has been captured.
        let mut entries: Vec<_> = [98, 29, 15]
            .into_iter()
            .enumerate()
            .map(|(slot, rows)| uniform(slot as u32, rows, wgpu::ShaderStages::FRAGMENT))
            .collect();
        entries.extend((3..8).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        }));
        entries.extend((8..10).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        }));
        let pixel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global-light dense pixel ABI"),
            entries: &entries,
        });
        let vertex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global-light dense vertex ABI"),
            entries: &[
                uniform(0, 98, wgpu::ShaderStages::VERTEX),
                uniform(1, 27, wgpu::ShaderStages::VERTEX),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("authored global-light VS/PS"),
            bind_group_layouts: &[&pixel_layout, &vertex_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("authored 80A0620F + 80A06213"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: vs,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: ps,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pixel_layout,
            vertex_layout,
            pipeline,
        }
    }

    pub fn bind(&self, device: &wgpu::Device, input: GlobalLightInputs<'_>) -> GlobalLightBindings {
        let mut entries: Vec<_> = input
            .pixel
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.extend(input.textures.iter().enumerate().map(|(slot, view)| {
            wgpu::BindGroupEntry {
                binding: 3 + slot as u32,
                resource: wgpu::BindingResource::TextureView(view),
            }
        }));
        entries.extend(input.samplers.iter().enumerate().map(|(slot, sampler)| {
            wgpu::BindGroupEntry {
                binding: 8 + slot as u32,
                resource: wgpu::BindingResource::Sampler(sampler),
            }
        }));
        let pixel = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global-light pixel inputs"),
            layout: &self.pixel_layout,
            entries: &entries,
        });
        let vertex = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global-light vertex inputs"),
            layout: &self.vertex_layout,
            entries: &input
                .vertex
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        GlobalLightBindings { pixel, vertex }
    }

    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>, inputs: &GlobalLightBindings) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &inputs.pixel, &[]);
        pass.set_bind_group(1, &inputs.vertex, &[]);
        // Exact paired VS generates four fullscreen-strip corners from ID.
        pass.draw(0..4, 0..1);
    }
}
