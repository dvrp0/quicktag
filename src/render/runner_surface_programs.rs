//! Exact opaque programs and their independently audited descriptor/gate contracts.
use super::authored_program::{AuthoredProgram, DescriptorAbi, decode_sha256};
use super::technique::ShaderStage;
use tiger_pkg::TagHash;

pub(crate) struct SurfaceProgram {
    pub program: AuthoredProgram,
    pub vertex_programs: &'static [DescriptorAbi],
    pub rows: usize,
    pub texture_count: u32,
    pub volume_slot: Option<u32>,
    pub sampler_count: u32,
    pub metadata_rows: &'static [(usize, Option<usize>)],
    pub unresolved_dependencies: &'static [&'static str],
}

include!("runner_surface_programs_data.rs");
