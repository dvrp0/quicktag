//! Retained raw-IA submission for audited opaque VS/PS pairs.
//! Attachment formats and inherited states remain explicit caller contracts.

use std::ops::Range;

use super::{body_material::BodyPixelInputs, body_vertex};

pub(crate) struct BodyDrawPipeline {
    pixel_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    index_format: wgpu::IndexFormat,
}

/// A stable descriptor/IA set. Bind groups retain their bound resources; mutable
/// uniform contents can change without rebuilding descriptors every frame.
pub(crate) struct BodyDrawBindings {
    pub pixel: wgpu::BindGroup,
    pub vertex: wgpu::BindGroup,
    pub source: wgpu::Buffer,
    pub uv: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub range: Range<u32>,
}

pub(crate) struct BodyDrawInputs<'a> {
    pub pixel: BodyPixelInputs<'a>,
    pub vertex: &'a wgpu::BindGroup,
    pub source: &'a wgpu::Buffer,
    pub uv: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    /// Authored strip range, including fixed restart indices.
    pub range: Range<u32>,
}

impl BodyDrawPipeline {
    pub fn new(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        vertex_layout: &wgpu::BindGroupLayout,
        targets: [Option<wgpu::ColorTargetState>; 4],
        front_face: wgpu::FrontFace,
        cull_mode: Option<wgpu::Face>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        Self::new_dense(
            device,
            vs,
            ps,
            vertex_layout,
            super::dense_pixel::SurfacePixelAbi::Body,
            targets,
            front_face,
            cull_mode,
            depth_stencil,
            index_format,
        )
    }

    /// Independently verified PS families share raw body VS/IA submission.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dense(
        device: &wgpu::Device,
        vs: &wgpu::ShaderModule,
        ps: &wgpu::ShaderModule,
        vertex_layout: &wgpu::BindGroupLayout,
        abi: super::dense_pixel::SurfacePixelAbi,
        targets: [Option<wgpu::ColorTargetState>; 4],
        front_face: wgpu::FrontFace,
        cull_mode: Option<wgpu::Face>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        Self::new_with_vertex(device, ps, vertex_layout, abi, targets, front_face,
            cull_mode, depth_stencil, body_vertex::layout7_vertex_state(vs),
            wgpu::PrimitiveTopology::TriangleStrip, Some(index_format), index_format)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_direct_ia(
        device: &wgpu::Device, vs: &wgpu::ShaderModule, ps: &wgpu::ShaderModule,
        vertex_layout: &wgpu::BindGroupLayout, abi: super::dense_pixel::SurfacePixelAbi,
        targets: [Option<wgpu::ColorTargetState>; 4], front_face: wgpu::FrontFace,
        cull_mode: Option<wgpu::Face>, depth_stencil: Option<wgpu::DepthStencilState>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        Self::new_with_vertex(device, ps, vertex_layout, abi, targets, front_face,
            cull_mode, depth_stencil, super::rigid_vertex::vertex_state(vs),
            wgpu::PrimitiveTopology::TriangleList, None, index_format)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_vertex(
        device: &wgpu::Device, ps: &wgpu::ShaderModule,
        vertex_layout: &wgpu::BindGroupLayout, abi: super::dense_pixel::SurfacePixelAbi,
        targets: [Option<wgpu::ColorTargetState>; 4], front_face: wgpu::FrontFace,
        cull_mode: Option<wgpu::Face>, depth_stencil: Option<wgpu::DepthStencilState>,
        vertex: wgpu::VertexState<'_>, topology: wgpu::PrimitiveTopology,
        strip_index_format: Option<wgpu::IndexFormat>,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        let pixel_layout = super::dense_pixel::create_layout(device, abi);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("authored body VS/PS pipeline ABI"),
            bind_group_layouts: &[&pixel_layout, vertex_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("authored body 80A60DAD + 80A9D0E1"),
            layout: Some(&layout),
            vertex,
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format,
                front_face,
                cull_mode,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
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
            index_format,
        }
    }

    pub fn bind(&self, device: &wgpu::Device, inputs: BodyDrawInputs<'_>) -> BodyDrawBindings {
        self.bind_dense(
            device,
            [inputs.pixel.cb0, inputs.pixel.cb1, inputs.pixel.cb12],
            &inputs.pixel.textures,
            inputs.pixel.samplers,
            inputs.vertex,
            inputs.source,
            inputs.uv,
            inputs.indices,
            inputs.range,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_dense<'a>(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 3],
        textures: &[&wgpu::TextureView],
        samplers: impl AsRef<[&'a wgpu::Sampler]>,
        vertex: &wgpu::BindGroup,
        source: &wgpu::Buffer,
        uv: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        range: Range<u32>,
    ) -> BodyDrawBindings {
        assert!(
            range.start < range.end,
            "body draw requires an authored index range"
        );
        assert!(
            u64::from(range.end)
                * match self.index_format {
                    wgpu::IndexFormat::Uint16 => 2,
                    wgpu::IndexFormat::Uint32 => 4,
                }
                <= indices.size(),
            "body strip exceeds index image"
        );
        BodyDrawBindings {
            pixel: super::dense_pixel::bind(
                device,
                &self.pixel_layout,
                buffers,
                textures,
                samplers,
            ),
            vertex: vertex.clone(),
            source: source.clone(),
            uv: uv.clone(),
            indices: indices.clone(),
            range,
        }
    }

    /// Direct generated-buffer consumers stay in the same encoder as compute.
    /// No reconstructed vertex/index stream or CPU re-upload enters this draw.
    pub fn encode(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &BodyDrawBindings) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.pixel, &[]);
        pass.set_bind_group(1, &bindings.vertex, &[]);
        pass.set_vertex_buffer(0, bindings.source.slice(..));
        pass.set_vertex_buffer(1, bindings.uv.slice(..));
        pass.set_index_buffer(bindings.indices.slice(..), self.index_format);
        pass.draw_indexed(bindings.range.clone(), 0, 0..1);
    }
}
