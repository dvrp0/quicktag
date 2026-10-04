//! Shared descriptor ABI verified independently for PS80A9D103 and PS80A9D11A.
//! Shader identity/semantics remain separate; callers supply all runtime inputs.

pub(crate) struct LayeredPixelInputs<'a> {
    pub cb0: &'a wgpu::Buffer,
    pub cb1: &'a wgpu::Buffer,
    pub cb12: &'a wgpu::Buffer,
    pub textures: [&'a wgpu::TextureView; 9],
    pub samplers: [&'a wgpu::Sampler; 3],
}

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    super::dense_pixel::create_layout(device, super::dense_pixel::SurfacePixelAbi::Layered)
}

/// Retain an explicit resource set; uniform updates never rebuild descriptors.
pub(crate) fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    inputs: LayeredPixelInputs<'_>,
) -> wgpu::BindGroup {
    super::dense_pixel::bind(
        device,
        layout,
        [inputs.cb0, inputs.cb1, inputs.cb12],
        &inputs.textures,
        inputs.samplers,
    )
}
