//! Pixel descriptor ABI for the identity-verified body PS80A9D0E1.
//! Required values/views are explicit caller inputs, never generic substitutes.

pub(crate) struct BodyPixelInputs<'a> {
    pub cb0: &'a wgpu::Buffer,
    pub cb1: &'a wgpu::Buffer,
    pub cb12: &'a wgpu::Buffer,
    pub textures: [&'a wgpu::TextureView; 8],
    pub samplers: [&'a wgpu::Sampler; 3],
}

pub(crate) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    super::dense_pixel::create_layout(device, super::dense_pixel::SurfacePixelAbi::Body)
}

/// Retain an explicit resource set; uniform updates never rebuild descriptors.
pub(crate) fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    inputs: BodyPixelInputs<'_>,
) -> wgpu::BindGroup {
    super::dense_pixel::bind(
        device,
        layout,
        [inputs.cb0, inputs.cb1, inputs.cb12],
        &inputs.textures,
        inputs.samplers,
    )
}
