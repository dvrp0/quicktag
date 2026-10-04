//! Uninterpreted authored MRT storage and source-color inspection.
//! FP32 is a viewer intermediate, not a claim about native allocation formats.

pub(crate) struct SurfaceTargets {
    pub textures: [wgpu::Texture; 4],
    pub views: [wgpu::TextureView; 4],
}

impl SurfaceTargets {
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let textures = std::array::from_fn(|index| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(
                    [
                        "authored RT0 source color",
                        "authored RT1 packed normal/class",
                        "authored RT2 material properties",
                        "authored RT3 signed motion",
                    ][index],
                ),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | if cfg!(test) { wgpu::TextureUsages::COPY_DST } else { wgpu::TextureUsages::empty() },
                view_formats: &[],
            })
        });
        let views = std::array::from_fn(|i| textures[i].create_view(&Default::default()));
        Self { textures, views }
    }
}

/// Retained native receiver snapshots used while bridging generic receivers.
///
/// `original_native_normal` captures native coverage before generic import;
/// `opaque` captures native RT0..RT2 after import and before stage-2 decals;
/// `coverage` marks pixels whose native values should reach viewer G-buffer
/// projections. All snapshots use the native FP32 ABI, so comparisons stay
/// independent of generic target quantization.
pub(crate) struct ReceiverTargets {
    pub(crate) original_native_normal: wgpu::Texture,
    pub(crate) original_native_normal_view: wgpu::TextureView,
    pub(crate) opaque: [wgpu::Texture; 3],
    pub(crate) opaque_views: [wgpu::TextureView; 3],
    pub(crate) coverage: wgpu::Texture,
    pub(crate) coverage_view: wgpu::TextureView,
    size: [u32; 2],
}

impl ReceiverTargets {
    pub(crate) fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let snapshot = |label: &'static str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let original_native_normal = snapshot("original native RT1 coverage snapshot");
        let original_native_normal_view = original_native_normal.create_view(&Default::default());
        let opaque = [
            snapshot("opaque native RT0 snapshot"),
            snapshot("opaque native RT1 snapshot"),
            snapshot("opaque native RT2 snapshot"),
        ];
        let opaque_views =
            std::array::from_fn::<_, 3, _>(|index| opaque[index].create_view(&Default::default()));
        let coverage = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("native receiver projection coverage"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let coverage_view = coverage.create_view(&Default::default());
        Self {
            original_native_normal,
            original_native_normal_view,
            opaque,
            opaque_views,
            coverage,
            coverage_view,
            size,
        }
    }

    fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.size[0],
            height: self.size[1],
            depth_or_array_layers: 1,
        }
    }

    pub(crate) fn copy_original_native_normal(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            self.original_native_normal.as_image_copy(),
            self.extent(),
        );
    }

    pub(crate) fn copy_opaque(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &SurfaceTargets,
    ) {
        for index in 0..3 {
            encoder.copy_texture_to_texture(
                source.textures[index].as_image_copy(),
                self.opaque[index].as_image_copy(),
                self.extent(),
            );
        }
    }
}

/// Import generic viewer receiver values into the native authored MRT ABI.
///
/// Generic materials already produced the same semantic normal, roughness,
/// metalness, AO, and albedo values consumed by the viewer lighting pass. A
/// native stage-2 decal still needs those pixels present in `SurfaceTargets`
/// before its depth-equal draw, however. This pass performs that bridge in a
/// separate fullscreen draw; it does not touch the generic viewer targets.
pub(crate) struct ViewerReceiverImport {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl ViewerReceiverImport {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let mut entries = vec![texture(0), texture(1), texture(2)];
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(32),
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("generic receiver to authored MRT import ABI"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("generic receiver to authored MRT import"),
            source: wgpu::ShaderSource::Wgsl(VIEWER_RECEIVER_IMPORT.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("generic receiver to authored MRT import layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("generic receiver to authored MRT import"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &std::array::from_fn::<_, 4, _>(|_| {
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                }),
            }),
            multiview: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        normal: &wgpu::TextureView,
        properties: &wgpu::TextureView,
        albedo: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        original_native_normal: &wgpu::TextureView,
        scene: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("generic receiver to authored MRT import inputs"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normal),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(properties),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(original_native_normal),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scene.as_entire_binding(),
                },
            ],
        })
    }

    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>, inputs: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, inputs, &[]);
        pass.draw(0..3, 0..1);
    }
}

const VIEWER_RECEIVER_IMPORT: &str = r#"
@group(0) @binding(0) var generic_normal: texture_2d<f32>;
@group(0) @binding(1) var generic_properties: texture_2d<f32>;
@group(0) @binding(2) var generic_albedo: texture_2d<f32>;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var original_native_normal: texture_2d<f32>;

struct Camera { center: vec4<f32>, params0: vec4<f32> }
@group(0) @binding(5) var<uniform> camera: Camera;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[i], 0.0, 1.0);
}

struct Output {
    @location(0) source_color: vec4<f32>,
    @location(1) packed_normal: vec4<f32>,
    @location(2) native_properties: vec4<f32>,
    @location(3) motion: vec4<f32>,
}

@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> Output {
    let pixel = vec2<i32>(position.xy);
    let depth = textureLoad(scene_depth, pixel, 0);
    let albedo = textureLoad(generic_albedo, pixel, 0);
    // Depth 1 is viewer background. Albedo alpha <= 0 is an uncovered/cleared
    // generic pixel. A written packed normal can have material class zero;
    // only the all-zero cleared MRT is absent. Never use class as coverage.
    let original_native = textureLoad(original_native_normal, pixel, 0);
    if depth >= 1.0 || albedo.a <= 0.0 || any(original_native != vec4<f32>(0.0)) {
        discard;
    }

    let encoded = textureLoad(generic_normal, pixel, 0);
    let view_vector = 2.0 * encoded.rgb - vec3<f32>(1.0);
    let view_length = length(view_vector);
    if view_length <= 0.000001 {
        discard;
    }
    let view_normal = view_vector / view_length;

    let cy = cos(camera.params0.y);
    let sy = sin(camera.params0.y);
    let cp = cos(camera.params0.z);
    let sp = sin(camera.params0.z);
    // Invert ViewerMaterialProjection's pitch, yaw, and Y/Z remap exactly.
    let yawed = vec3<f32>(
        view_normal.x,
        view_normal.y * cp + view_normal.z * sp,
        -view_normal.y * sp + view_normal.z * cp,
    );
    let z_up = vec3<f32>(
        yawed.x * cy - yawed.z * sy,
        yawed.y,
        yawed.x * sy + yawed.z * cy,
    );
    let world = vec3<f32>(z_up.x, z_up.z, z_up.y);
    let radius = (4.0 - clamp(encoded.a, 0.0, 1.0)) * 0.25;
    let packed = world * (radius * 0.5) + vec3<f32>(0.5);
    let properties = textureLoad(generic_properties, pixel, 0);

    var output: Output;
    output.source_color = vec4<f32>(albedo.rgb, 0.0);
    output.packed_normal = vec4<f32>(packed, 0.67);
    output.native_properties = vec4<f32>(
        clamp(properties.r, 0.0, 1.0),
        clamp(properties.g, 0.0, 1.0) * 0.5,
        0.0,
        0.0,
    );
    output.motion = vec4<f32>(0.0);
    return output;
}
"#;

/// Mark native-only receivers and generic receivers changed by a stage-2
/// decal. Unchanged imported generic pixels stay on their original viewer
/// G-buffer path.
pub(crate) struct ViewerReceiverCoverage {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl ViewerReceiverCoverage {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("native receiver projection coverage ABI"),
            entries: &std::array::from_fn::<_, 7, _>(|binding| texture(binding as u32)),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native receiver projection coverage"),
            source: wgpu::ShaderSource::Wgsl(VIEWER_RECEIVER_COVERAGE.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("native receiver projection coverage layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("native receiver projection coverage"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    pub(crate) fn bind(
        &self,
        device: &wgpu::Device,
        current: [&wgpu::TextureView; 3],
        opaque: [&wgpu::TextureView; 3],
        original_native_normal: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut entries = Vec::with_capacity(7);
        for (binding, view) in current.into_iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        for (binding, view) in opaque.into_iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (binding + 3) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(original_native_normal),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native receiver projection coverage inputs"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    pub(crate) fn bind_targets(
        &self,
        device: &wgpu::Device,
        current: &SurfaceTargets,
        snapshots: &ReceiverTargets,
    ) -> wgpu::BindGroup {
        self.bind(
            device,
            [&current.views[0], &current.views[1], &current.views[2]],
            [
                &snapshots.opaque_views[0],
                &snapshots.opaque_views[1],
                &snapshots.opaque_views[2],
            ],
            &snapshots.original_native_normal_view,
        )
    }

    pub(crate) fn encode(&self, pass: &mut wgpu::RenderPass<'_>, inputs: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, inputs, &[]);
        pass.draw(0..3, 0..1);
    }
}

const VIEWER_RECEIVER_COVERAGE: &str = r#"
@group(0) @binding(0) var current_rt0: texture_2d<f32>;
@group(0) @binding(1) var current_rt1: texture_2d<f32>;
@group(0) @binding(2) var current_rt2: texture_2d<f32>;
@group(0) @binding(3) var opaque_rt0: texture_2d<f32>;
@group(0) @binding(4) var opaque_rt1: texture_2d<f32>;
@group(0) @binding(5) var opaque_rt2: texture_2d<f32>;
@group(0) @binding(6) var original_native_normal: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[i], 0.0, 1.0);
}

fn changed(a: vec4<f32>, b: vec4<f32>) -> bool {
    // Snapshots are exact FP32 copies. Compare bits so unchanged NaNs remain
    // unchanged and no arbitrary tolerance drops a real authored decal write.
    return any(bitcast<vec4<u32>>(a) != bitcast<vec4<u32>>(b));
}

@fragment fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let native_only = any(textureLoad(original_native_normal, pixel, 0) != vec4<f32>(0.0));
    let opaque_receiver = any(textureLoad(opaque_rt1, pixel, 0) != vec4<f32>(0.0));
    let modified = changed(textureLoad(current_rt0, pixel, 0), textureLoad(opaque_rt0, pixel, 0))
        || changed(textureLoad(current_rt1, pixel, 0), textureLoad(opaque_rt1, pixel, 0))
        || changed(textureLoad(current_rt2, pixel, 0), textureLoad(opaque_rt2, pixel, 0));
    // A decal write cannot create an opaque receiver in cleared background.
    let covered = native_only || (opaque_receiver && modified);
    return vec4<f32>(select(0.0, 1.0, covered));
}
"#;

pub(crate) struct SourceColorProjection {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::RenderPipeline; 2],
}

/// Convert audited opaque material outputs for the existing viewer light rigs.
/// This is a viewer ABI conversion, not the native final lighting shader.
pub(crate) struct ViewerMaterialProjection {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl ViewerMaterialProjection {
    pub fn new(device: &wgpu::Device, formats: [wgpu::TextureFormat; 3]) -> Self {
        let mut entries: Vec<_> = (0..3)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(32),
            },
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            count: None,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("authored material to viewer lighting ABI"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("authored opaque material viewer conversion"),
            source: wgpu::ShaderSource::Wgsl(VIEWER_MATERIAL.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("authored materials into viewer light rigs"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &formats.map(|format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                }),
            }),
            multiview: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        targets: &SurfaceTargets,
        scene: &wgpu::Buffer,
        coverage: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let mut entries: Vec<_> = (0..3)
            .map(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(&targets.views[i]),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: scene.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(
                coverage.unwrap_or(&targets.views[1]),
            ),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retained authored material viewer inputs"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>, inputs: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, inputs, &[]);
        pass.draw(0..3, 0..1);
    }
}

const VIEWER_MATERIAL: &str = r#"
@group(0) @binding(0) var source_color: texture_2d<f32>;
@group(0) @binding(1) var packed_normal: texture_2d<f32>;
@group(0) @binding(2) var native_properties: texture_2d<f32>;
struct Camera { center: vec4<f32>, params0: vec4<f32> }
@group(0) @binding(3) var<uniform> camera: Camera;
@group(0) @binding(4) var receiver_coverage: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3));
    return vec4<f32>(p[i],0,1);
}
struct Output {
    @location(0) normal_roughness: vec4<f32>,
    @location(1) properties: vec4<f32>,
    @location(2) albedo: vec4<f32>,
}
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> Output {
    let pixel = vec2<i32>(p.xy);
    let packed = textureLoad(packed_normal,pixel,0);
    if all(packed == vec4<f32>(0.0))
        || all(textureLoad(receiver_coverage,pixel,0) == vec4<f32>(0.0)) { discard; }
    let color = textureLoad(source_color,pixel,0);
    let props = textureLoad(native_properties,pixel,0);
    // Native normal decoder: radius and direction are independent. Never use
    // RT1's literal opaque class .67 as roughness or RT3 motion as emission.
    let vector = 2.0*packed.rgb-vec3<f32>(1.0);
    let radius = length(vector);
    let world = vector/max(radius,0.000001);
    let z_up = vec3<f32>(world.x,world.z,world.y);
    let cy=cos(camera.params0.y); let sy=sin(camera.params0.y);
    let cp=cos(camera.params0.z); let sp=sin(camera.params0.z);
    let yawed=vec3<f32>(z_up.x*cy+z_up.z*sy,z_up.y,-z_up.x*sy+z_up.z*cy);
    let view=vec3<f32>(yawed.x,yawed.y*cp-yawed.z*sp,yawed.y*sp+yawed.z*cp);
    // Explicit viewer GGX conversion of the native smoothness response.
    // Native final lighting uses this response differently per light family.
    let roughness=1.0-clamp(4.0*radius-3.0,0.0,1.0);
    // Packaged debug_metalness reads R directly. debug_texture_ao decodes
    // 2*G and excludes the high source-color class, before its display overlay.
    let ao=select(clamp(2.0*props.g,0.0,1.0),0.0,color.a>=0.995);
    var output: Output;
    output.normal_roughness=vec4<f32>(view*0.5+vec3<f32>(0.5),roughness);
    output.properties=vec4<f32>(clamp(props.r,0.0,1.0),ao,0,0);
    output.albedo=vec4<f32>(color.rgb,1);
    return output;
}
"#;

impl SourceColorProjection {
    pub fn new(device: &wgpu::Device, formats: [wgpu::TextureFormat; 2]) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("authored source-color inspection ABI"),
            entries: &std::array::from_fn::<_, 3, _>(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("authored RT0 source-color projection"),
            source: wgpu::ShaderSource::Wgsl(SOURCE_COLOR.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipelines = formats.map(|format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("authored source color to existing Base Color view"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::RED
                            | wgpu::ColorWrites::GREEN
                            | wgpu::ColorWrites::BLUE,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        });
        Self { layout, pipelines }
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        targets: &SurfaceTargets,
        coverage: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retained authored color and coverage"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        coverage.unwrap_or(&targets.views[1]),
                    ),
                },
            ],
        })
    }

    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>, inputs: &wgpu::BindGroup, hdr: bool) {
        pass.set_pipeline(&self.pipelines[usize::from(hdr)]);
        pass.set_bind_group(0, inputs, &[]);
        pass.draw(0..3, 0..1);
    }
}

const SOURCE_COLOR: &str = r#"
@group(0) @binding(0) var source_color: texture_2d<f32>;
@group(0) @binding(1) var packed_normal: texture_2d<f32>;
@group(0) @binding(2) var receiver_coverage: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    return vec4<f32>(p[i],0,1);
}
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(p.xy);
    // Original material class zero is valid; packed RGB still carries its
    // normal. The all-zero MRT identifies cleared pixels, not RT1.a alone.
    if all(textureLoad(packed_normal, pixel, 0) == vec4<f32>(0.0))
        || all(textureLoad(receiver_coverage, pixel, 0) == vec4<f32>(0.0)) { discard; }
    return vec4<f32>(textureLoad(source_color, pixel, 0).rgb, 1.0);
}
"#;
