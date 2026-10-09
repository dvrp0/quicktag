//! Dense descriptor ABIs for audited rigid body/head/hair mesh producers.
//! Inputs remain explicit: this component does not invent native runtime values.

pub(crate) struct BodyMeshProducer {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    hair: bool,
    procedural: bool,
    static_frame: Option<(wgpu::TextureView, wgpu::Sampler)>,
}

/// Caller-owned body/head images in original shader slot order. t0..t2 are
/// scalar source streams, t3 is expanded uint4 palette data, t4 is float4
/// bone data, t5..t8 are scalar images, and u0/u1 are generated packed
/// position/frame words.
pub(crate) struct BodyMeshBuffers<'a> {
    pub cb0: &'a wgpu::Buffer,
    pub t0: &'a wgpu::Buffer,
    pub t1: &'a wgpu::Buffer,
    pub t2: &'a wgpu::Buffer,
    pub t3: &'a wgpu::Buffer,
    pub t4: &'a wgpu::Buffer,
    pub t5: &'a wgpu::Buffer,
    pub t6: &'a wgpu::Buffer,
    pub t7: &'a wgpu::Buffer,
    pub t8: &'a wgpu::Buffer,
    pub u0: &'a wgpu::Buffer,
    pub u1: &'a wgpu::Buffer,
}

/// B762 procedural-body inputs in source order. cb1 is caller-owned scope-
/// skinning data; the producer owns its finite inactive Frame image/sampler.
/// Dense bindings: cb0=0, cb1=1, Frame t0=2, s1=3, t1..t10=4..13,
/// u0/u1=14/15. t4 (binding7) is uint4; t5 (binding8) and t9 (binding12)
/// are float4 buffers; every other source/output buffer is scalar u32.
pub(crate) struct BodyProceduralMeshBuffers<'a> {
    pub cb0: &'a wgpu::Buffer,
    pub cb1: &'a wgpu::Buffer,
    pub t1: &'a wgpu::Buffer,
    pub t2: &'a wgpu::Buffer,
    pub t3: &'a wgpu::Buffer,
    pub t4: &'a wgpu::Buffer,
    pub t5: &'a wgpu::Buffer,
    pub t6: &'a wgpu::Buffer,
    pub t7: &'a wgpu::Buffer,
    pub t8: &'a wgpu::Buffer,
    pub t9: &'a wgpu::Buffer,
    pub t10: &'a wgpu::Buffer,
    pub u0: &'a wgpu::Buffer,
    pub u1: &'a wgpu::Buffer,
}

/// Caller-owned HairCS images in original shader slot order. Hair keeps the
/// typed t3/t4 ABI and adds float4 t8 plus scalar t9 before its two UAVs.
pub(crate) struct HairMeshBuffers<'a> {
    pub cb0: &'a wgpu::Buffer,
    pub t0: &'a wgpu::Buffer,
    pub t1: &'a wgpu::Buffer,
    pub t2: &'a wgpu::Buffer,
    pub t3: &'a wgpu::Buffer,
    pub t4: &'a wgpu::Buffer,
    pub t5: &'a wgpu::Buffer,
    pub t6: &'a wgpu::Buffer,
    pub t7: &'a wgpu::Buffer,
    pub t8: &'a wgpu::Buffer,
    pub t9: &'a wgpu::Buffer,
    pub u0: &'a wgpu::Buffer,
    pub u1: &'a wgpu::Buffer,
}

impl BodyMeshProducer {
    /// Construct the 12-slot body/head ABI. Hair uses `new_for_abi` so
    /// its distinct 13-slot descriptor cannot be aliased accidentally.
    pub fn new(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Self {
        Self::new_with_cb0_rows(device, shader, 14)
    }

    /// Build the dense ABI with the exact uniform image size required by the
    /// registered producer. The 15-row variants retain the same 12 bindings;
    /// cb0's minimum range grows; B7C5 retains the original row indices.
    pub fn new_for_abi(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        abi: super::authored_program::DescriptorAbi,
    ) -> Self {
        let cb0_rows = match abi {
            super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage => 133,
            super::authored_program::DescriptorAbi::BodyProceduralComputeStorage => 81,
            super::authored_program::DescriptorAbi::HairMeshComputeStorage => 65,
            super::authored_program::DescriptorAbi::Cloth45RowComputeStorage => 45,
            super::authored_program::DescriptorAbi::Cloth46RowComputeStorage => 46,
            super::authored_program::DescriptorAbi::BodyMeshAA060BComputeStorage => 35,
            super::authored_program::DescriptorAbi::BodyMeshB152BEComputeStorage => 31,
            super::authored_program::DescriptorAbi::HeadMesh15RowA60035ComputeStorage
            | super::authored_program::DescriptorAbi::HeadMesh15RowB8BDComputeStorage
            | super::authored_program::DescriptorAbi::HeadMeshEC0BComputeStorage
            | super::authored_program::DescriptorAbi::BodyMesh15RowComputeStorage => 15,
            super::authored_program::DescriptorAbi::BodyMesh16RowD8C1ComputeStorage => 16,
            super::authored_program::DescriptorAbi::BodyMeshComputeStorage
            | super::authored_program::DescriptorAbi::BodyMeshB4CBComputeStorage
            | super::authored_program::DescriptorAbi::HeadMeshComputeStorage => 14,
            _ => panic!("unsupported rigid mesh producer ABI: {abi:?}"),
        };
        if abi == super::authored_program::DescriptorAbi::BodyProceduralComputeStorage {
            return Self::new_procedural(device, shader);
        }
        if abi == super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage {
            return Self::new_procedural_static(device, shader);
        }
        if matches!(abi, super::authored_program::DescriptorAbi::HairMeshComputeStorage
            | super::authored_program::DescriptorAbi::Cloth45RowComputeStorage
            | super::authored_program::DescriptorAbi::Cloth46RowComputeStorage
            | super::authored_program::DescriptorAbi::BodyMeshAA060BComputeStorage) {
            Self::new_with_cb0_rows_and_bindings(device, shader, cb0_rows, true)
        } else {
            Self::new_with_cb0_rows(device, shader, cb0_rows)
        }
    }

    fn new_with_cb0_rows(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        cb0_rows: u64,
    ) -> Self {
        Self::new_with_cb0_rows_and_bindings(device, shader, cb0_rows, false)
    }

    fn new_with_cb0_rows_and_bindings(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        cb0_rows: u64,
        hair: bool,
    ) -> Self {
        let binding_count = if hair { 13 } else { 12 };
        let entries: Vec<_> = (0..binding_count)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < if hair { 11 } else { 10 },
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(match binding {
                        0 => cb0_rows * 16,
                        4 | 5 if hair => 16,
                        9 if hair => 16,
                        4 | 5 => 16,
                        _ => 4,
                    }),
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(if hair {
                "authored hair mesh compute dense ABI"
            } else {
                "authored body mesh compute dense ABI"
            }),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(if hair {
                "authored hair mesh compute pipeline ABI"
            } else {
                "authored body mesh compute pipeline ABI"
            }),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(if hair {
                "authored hair mesh producer"
            } else {
                "authored rigid mesh producer"
            }),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline, hair, procedural: false, static_frame: None }
    }

    fn new_procedural(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Self {
        let entries: Vec<_> = (0..16)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: match binding {
                    2 => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    3 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    _ => wgpu::BindingType::Buffer {
                        ty: if binding == 0 || binding == 1 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 14,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(match binding {
                            0 => 81 * 16,
                            1 => 31 * 16,
                            7 | 8 | 12 => 16,
                            _ => 4,
                        }),
                    },
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("authored B762 procedural body compute ABI"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("authored B762 procedural body compute pipeline ABI"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("authored B762 procedural body producer"),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let static_frame = Self::create_static_frame(
            device,
            "finite inactive procedural Frame image",
            "bounded inactive procedural Frame sampler",
        );
        Self { layout, pipeline, hair: false, procedural: true, static_frame: Some(static_frame) }
    }

    fn create_static_frame(
        device: &wgpu::Device,
        texture_label: &'static str,
        sampler_label: &'static str,
    ) -> (wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(texture_label),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(sampler_label),
            ..Default::default()
        });
        (view, sampler)
    }

    /// B7BC keeps its original noise sample and procedural arithmetic. The
    /// viewer static branch has zero deformation weight; retain a bounded,
    /// finite Frame image for the output-inert sample, never a material texture.
    fn new_procedural_static(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Self {
        let entries: Vec<_> = (0..15).map(|binding| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE, count: None,
            ty: match binding {
                1 => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
                2 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                _ => wgpu::BindingType::Buffer {
                    ty: if binding == 0 { wgpu::BufferBindingType::Uniform }
                        else { wgpu::BufferBindingType::Storage { read_only: binding < 13 } },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(match binding {
                        0 => 133 * 16, 6 | 7 | 11 => 16, _ => 4,
                    }),
                },
            },
        }).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("authored B7BC static compute ABI"), entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("authored B7BC compute layout"), bind_group_layouts: &[&layout], push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("authored B7BC static mesh producer"), layout: Some(&pipeline_layout),
            module: shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let static_frame = Self::create_static_frame(
            device,
            "finite inactive static Frame image",
            "bounded inactive Frame sampler",
        );
        Self { layout, pipeline, hair: true, procedural: false, static_frame: Some(static_frame) }
    }

    /// Construct once per immutable set of source/output buffers; update buffer
    /// contents in place for runtime values without recreating this bind group.
    pub fn bind(&self, device: &wgpu::Device, buffers: BodyMeshBuffers<'_>) -> wgpu::BindGroup {
        assert!(!self.hair && !self.procedural, "specialized mesh ABI requires its explicit bind method");
        let slots = [
            buffers.cb0,
            buffers.t0,
            buffers.t1,
            buffers.t2,
            buffers.t3,
            buffers.t4,
            buffers.t5,
            buffers.t6,
            buffers.t7,
            buffers.t8,
            buffers.u0,
            buffers.u1,
        ];
        let entries: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("authored body mesh explicit inputs/outputs"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Bind B762's explicit 16-slot ABI. The producer-owned finite Frame view
    /// and nearest sampler occupy bindings 2/3; cb1 remains caller-owned.
    pub fn bind_procedural(
        &self,
        device: &wgpu::Device,
        buffers: BodyProceduralMeshBuffers<'_>,
    ) -> wgpu::BindGroup {
        assert!(self.procedural, "body/head producer is not the B762 procedural ABI");
        let (frame, sampler) = self
            .static_frame
            .as_ref()
            .expect("B762 procedural ABI requires its producer-owned Frame image");
        let sources = [
            buffers.t1,
            buffers.t2,
            buffers.t3,
            buffers.t4,
            buffers.t5,
            buffers.t6,
            buffers.t7,
            buffers.t8,
            buffers.t9,
            buffers.t10,
            buffers.u0,
            buffers.u1,
        ];
        let mut entries = Vec::with_capacity(16);
        entries.push(wgpu::BindGroupEntry { binding: 0, resource: buffers.cb0.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: 1, resource: buffers.cb1.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(frame) });
        entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(sampler) });
        entries.extend(sources.iter().enumerate().map(|(index, buffer)| wgpu::BindGroupEntry {
            binding: index as u32 + 4,
            resource: buffer.as_entire_binding(),
        }));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("authored B762 procedural body explicit inputs/outputs"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Bind HairCS's exact 13-slot dense ABI. Hair callers must use this
    /// method so its float4 t8 and scalar t9 cannot be confused with the
    /// body's scalar t8 path.
    pub fn bind_hair(&self, device: &wgpu::Device, buffers: HairMeshBuffers<'_>) -> wgpu::BindGroup {
        assert!(self.hair, "body/head producer cannot bind HairCS resources");
        let slots = [
            buffers.cb0,
            buffers.t0,
            buffers.t1,
            buffers.t2,
            buffers.t3,
            buffers.t4,
            buffers.t5,
            buffers.t6,
            buffers.t7,
            buffers.t8,
            buffers.t9,
            buffers.u0,
            buffers.u1,
        ];
        let mut entries: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32 + if self.static_frame.is_some() && binding > 0 { 2 } else { 0 },
                resource: buffer.as_entire_binding(),
            })
            .collect();
        if let Some((view, sampler)) = &self.static_frame {
            entries.push(wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) });
            entries.push(wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("authored hair mesh explicit inputs/outputs"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Dispatch dimensions are caller-owned runtime inputs, never inferred from
    /// a graphics index count. The audited local size is64x1x1.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &wgpu::BindGroup,
        groups: [u32; 3],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("authored body mesh producer"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bindings, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
}
