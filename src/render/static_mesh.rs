//! Viewer-owned bind pose for identity-verified body/head/hair producers.
//! This selects an authored static branch; it does not emulate animation or
//! claim that these constants are captured native runtime state.
use wgpu::util::DeviceExt;

/// Return the exclusive t3 `TypedBuffer<uint4>` record bound required by the
/// HairCS static identity path for an identity vertex remap. This mirrors the
/// DXIL source exactly: POSITION.w is signed-SNORM16 at byte offset 6, the
/// weighted branch uses `fptosi(abs(w * 32767.099609375))`, then computes
/// `(abs_value * 8 - 16384) | ((vertex << sign) & 7)`.
///
/// Rigid vertices still issue the shader's output-inert negative/out-of-range
/// fetch before mode selection; callers must retain the backend's bounded
/// static policy for that access. This helper validates the actual weighted
/// palette records without inventing native palette values.
pub(crate) fn hair_static_palette_record_end(source: &[u8]) -> Option<usize> {
    const STRIDE: usize = 24;
    const SNORM16_SCALE: f32 = f32::from_bits(0x38000100);
    const POSITION_W_SCALE: f32 = 32767.099609375;
    if source.is_empty() || source.len() % STRIDE != 0 {
        return None;
    }
    let mut end = 0usize;
    for (vertex, record) in source.chunks_exact(STRIDE).enumerate() {
        let raw = i16::from_le_bytes([record[6], record[7]]) as f32;
        let normalized = (-1.0f32).max(raw * SNORM16_SCALE);
        let value = (normalized * POSITION_W_SCALE) as i32;
        let absolute = value.unsigned_abs() as usize;
        if absolute > 2047 {
            let index = (absolute * 8 - 16_384)
                | ((vertex << usize::from(value < 0)) & 7);
            end = end.max(index + 1);
        }
    }
    Some(end)
}

pub(crate) fn hair_static_inputs_valid(source: &[u8], palette_bytes: usize) -> bool {
    // Zero source directions are valid authored data. HairCS normalizes them,
    // then its original NMax/NMin packing clamps NaN lanes to -32767. Rejecting
    // them here would incorrectly drop WHITE RABBIT's referenced vertex8156.
    // Preserve that source behavior; validate only the actual palette bounds.
    hair_static_palette_record_end(source)
        .and_then(|end| end.checked_mul(4))
        .is_some_and(|end| end <= palette_bytes)
}

/// Approved static-preview policy for the exact B152BE quaternion source.
/// Its two distance gates must remain active: an inactive pair normalizes a
/// zero quaternion axis before row13 masks deformation. Decode the original
/// SNORM16 positions exactly as the static stream descriptors do and enclose
/// both authored centers. Shader arithmetic and source pose remain unchanged.
/// Only simulation radius row16.x differs from the authored tail.
pub(crate) fn packed_rotation_static_radius(source: &[u8], constants: &[[f32;4]]) -> Option<f32> {
    if source.is_empty() || source.len()%24!=0 || constants.len()!=31
        || constants[14..].iter().flatten().any(|v|!v.is_finite()) { return None; }
    let axis: [f32;3]=std::array::from_fn(|lane|constants[14][lane]+constants[17][lane]);
    let axis_length=axis.iter().map(|v|v*v).sum::<f32>();
    if !axis_length.is_normal() || !axis_length.sqrt().recip().is_finite() { return None; }
    let mut radius=0.0_f32;
    for vertex in source.chunks_exact(24) {
        let position: [f32;3]=std::array::from_fn(|lane| {
            let raw=i16::from_le_bytes(vertex[lane*2..lane*2+2].try_into().unwrap());
            (-1.0_f32).max(f32::from(raw)*f32::from_bits(0x38000100))
        });
        for center in [constants[15],constants[18]] {
            let d: [f32;3]=std::array::from_fn(|lane|position[lane]-center[lane]);
            let distance=((d[0]*d[0]+d[1]*d[1])+d[2]*d[2]).sqrt();
            if !distance.is_finite() { return None; }
            radius=radius.max(distance);
        }
    }
    // Strict source '<' gates, with an outward margin for GPU dot/sqrt
    // rounding. The margin scales with the actual measured mesh/center range.
    let radius=radius+(radius.max(1.0)*32.0*f32::EPSILON);
    (radius.is_normal()).then_some(radius)
}

pub(crate) struct StaticMesh {
    pub producer: super::authored_program::DescriptorAbi,
    constants: wgpu::Buffer,
    procedural_object: Option<wgpu::Buffer>,
    source: wgpu::Buffer,
    palette: wgpu::Buffer,
    bones: wgpu::Buffer,
    batch: wgpu::Buffer,
    unused: wgpu::Buffer,
    remap: wgpu::Buffer,
    pub positions: wgpu::Buffer,
    pub frames: wgpu::Buffer,
    count: u32,
    generated: std::sync::atomic::AtomicBool,
}

impl StaticMesh {
    pub fn new(
        device: &wgpu::Device,
        source: &wgpu::Buffer,
        palette: &[u8],
        count: u32,
        producer: super::authored_program::DescriptorAbi,
        source_stride: u32,
        rigid_float_palette: Option<&wgpu::Buffer>,
        authored_compute_constants: Option<&[[f32; 4]]>,
    ) -> Self {
        let procedural_body = producer == super::authored_program::DescriptorAbi::BodyProceduralComputeStorage;
        let deformation = producer.deformation_constant_contract().is_some();
        let procedural_hair = producer == super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage;
        let procedural = procedural_body || procedural_hair;
        let hair = procedural_hair || producer == super::authored_program::DescriptorAbi::HairMeshComputeStorage;
        let shifted_head = matches!(
            producer,
            super::authored_program::DescriptorAbi::HeadMesh15RowA60035ComputeStorage
                | super::authored_program::DescriptorAbi::HeadMesh15RowB8BDComputeStorage
        );
        assert!(matches!(
            producer,
            super::authored_program::DescriptorAbi::BodyMeshComputeStorage
                | super::authored_program::DescriptorAbi::BodyMeshB4CBComputeStorage
                | super::authored_program::DescriptorAbi::BodyMesh15RowComputeStorage
                | super::authored_program::DescriptorAbi::HeadMeshComputeStorage
                | super::authored_program::DescriptorAbi::HeadMeshEC0BComputeStorage
                | super::authored_program::DescriptorAbi::HeadMesh15RowA60035ComputeStorage
                | super::authored_program::DescriptorAbi::HeadMesh15RowB8BDComputeStorage
                | super::authored_program::DescriptorAbi::HairMeshComputeStorage
                | super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage
                | super::authored_program::DescriptorAbi::BodyProceduralComputeStorage
                | super::authored_program::DescriptorAbi::Cloth45RowComputeStorage
                | super::authored_program::DescriptorAbi::Cloth46RowComputeStorage
                | super::authored_program::DescriptorAbi::BodyMeshAA060BComputeStorage
                | super::authored_program::DescriptorAbi::BodyMeshB152BEComputeStorage
        ));
        assert!(if hair || procedural_body || deformation {
            source_stride == 24
        } else {
            matches!(source_stride, 24 | 48)
        });
        assert!(count > 0 && source.size() == u64::from(count) * u64::from(source_stride));
        assert!(if source_stride==24 { !palette.is_empty() && palette.len()%4==0 } else {palette.is_empty()});
        let format=if source_stride==24 {11.0}else{4.0};
        let step=source_stride as f32/3.0;
        let constants: Vec<[f32; 4]> = if deformation {
            // Preserve the prepared procedural tail. B152BE's caller derives
            // only row16.x under the approved finite static-radius policy;
            // all other tail bits remain authored. Original row13.x selects
            // zero deformation with the explicit viewer bind-pose descriptors.
            let mut constants=authored_compute_constants.expect("authored deformation image").to_vec();
            let (authored_rows, declared_rows)=producer.deformation_constant_contract().unwrap();
            assert_eq!(constants.len(),authored_rows);
            // Source-verified unused declaration padding, never a numeric
            // substitute for a loaded row or unresolved TFX output.
            constants.resize(declared_rows,[0.0;4]);
            constants[0]=[0.0,source_stride as f32,format,0.0];
            constants[1]=[0.0,0.0,0.0,1.0];
            constants[2]=[step,source_stride as f32,format,0.0];
            constants[3]=[step*2.0,source_stride as f32,format,0.0];
            constants[4]=[0.0;4];
            constants[7]=[f32::from_bits(count),0.0,0.0,0.0];
            for row in [8,9,10,11,13] { constants[row]=[0.0;4]; }
            constants[12]=[f32::from_bits(count),0.0,0.0,0.0];
            constants
        } else if procedural {
            // Preserve all authored procedural parameters. Only the explicit
            // viewer mesh/Frame inputs and zero-deformation static branch are
            // selected here; the original source computes and packs the pose.
            let mut constants = authored_compute_constants.expect("authored procedural image").to_vec();
            assert_eq!(constants.len(), if procedural_body { 81 } else { 133 });
            constants[0] = [1.0, 0.0, 0.0, 0.0]; // finite wind direction, zero time
            constants[2] = [0.0; 4]; // zero wind-noise amplitude
            constants[7] = [0.0, source_stride as f32, format, 0.0];
            constants[8] = [0.0, 0.0, 0.0, 1.0];
            constants[9] = [step, source_stride as f32, format, 0.0];
            constants[10] = [step * 2.0, source_stride as f32, format, 0.0];
            constants[11] = [0.0; 4];
            constants[14] = [f32::from_bits(count), 0.0, 0.0, 0.0];
            constants[15] = [0.0; 4]; // original mode0
            constants[16] = [0.0; 4]; // batch offset
            constants[17] = [0.0; 4]; // bounded inactive auxiliary index
            constants[18] = [0.0; 4]; // identity-remap offset
            constants[19] = [f32::from_bits(count), 0.0, 0.0, 0.0];
            constants[20] = [0.0; 4]; // original procedural-delta weight
            constants
        } else if hair {
            // HairCS's viewer-owned static policy. Rows32/50 are finite unit
            // axes: both rows feed an unconditional Rsqrt before row13.x
            // masks every simulation delta. All other simulation rows remain
            // zero; this is a finite mode0/no-motion policy, not a native
            // animation upload.
            let mut constants = vec![[0.0f32; 4]; 65];
            constants[0] = [0.0, source_stride as f32, format, 0.0];
            constants[1] = [0.0, 0.0, 0.0, 1.0];
            constants[2] = [step, source_stride as f32, format, 0.0];
            constants[3] = [step * 2.0, source_stride as f32, format, 0.0];
            constants[7][0] = f32::from_bits(count);
            constants[8][0] = 0.0;
            constants[9][0] = 0.0;
            constants[10][0] = 0.0;
            constants[11][0] = 0.0;
            constants[12][0] = f32::from_bits(count);
            constants[13][0] = 0.0;
            constants[32][0] = 1.0;
            constants[50][0] = 1.0;
            constants
        } else if shifted_head {
            let mut constants = vec![[0.0f32; 4]; 15];
            constants[1] = [0.0, source_stride as f32, format, 0.0];
            constants[2][3] = 1.0;
            constants[3] = [step, source_stride as f32, format, 0.0];
            constants[4] = [step*2.0, source_stride as f32, format, 0.0];
            constants[8][0] = f32::from_bits(count);
            constants[13][0] = f32::from_bits(count);
            constants
        } else {
            let mut constants = [[0.0f32; 4]; 14];
            constants[0] = [0.0, source_stride as f32, format, 0.0];
            constants[1][3] = 1.0;
            constants[2] = [step, source_stride as f32, format, 0.0];
            constants[3] = [step*2.0, source_stride as f32, format, 0.0];
            constants[7][0] = f32::from_bits(count);
            constants[12][0] = f32::from_bits(count);
            let mut constants = constants.to_vec();
            if matches!(producer, super::authored_program::DescriptorAbi::BodyMesh15RowComputeStorage
                | super::authored_program::DescriptorAbi::HeadMeshEC0BComputeStorage) {
                constants.push([0.0; 4]);
            }
            constants
        };
        let buffer = |label, data: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
        };
        // Float POSITION.w=1 triggers an unconditional t3 uint4 read at
        // 245752..245759 before the inactive mode0 switch. Bind its full range,
        // not an out-of-bounds placeholder. Values are proved output-inert in
        // this viewer rigid branch; this is not an invented native palette.
        let palette_buffer=if source_stride==48 {
            let palette=rigid_float_palette.expect("bounded viewer Float48 palette image");
            assert_eq!(palette.size(),245_760*16);
            palette.clone()
        } else {
            assert!(rigid_float_palette.is_none());
            let expanded:Vec<u32>=palette.iter().map(|&v|u32::from(v)).collect();
            buffer("authored expanded byte palette",bytemuck::cast_slice(&expanded),wgpu::BufferUsages::STORAGE)
        };
        let remap: Vec<u32> = (0..count).collect();
        let output = vec![0u32; count as usize * 2 + 4];
        let storage = wgpu::BufferUsages::STORAGE;
        let output_usage = storage
            | if hair || procedural_body || deformation || matches!(producer,
                super::authored_program::DescriptorAbi::BodyMeshB4CBComputeStorage
                | super::authored_program::DescriptorAbi::HeadMeshEC0BComputeStorage) {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            };
        Self {
            producer,
            procedural_object: procedural_body.then(|| {
                // B762 reads cb1[5].w only, in its procedural delta graph.
                // Match the existing static viewer object-frequency policy.
                // Preserve the source's declared 31-row range; unused rows
                // cannot substitute for native animation state.
                let mut object = [[0.0f32; 4]; 31];
                object[5][3] = 1.0;
                buffer("viewer procedural object scope", bytemuck::cast_slice(&object),
                    wgpu::BufferUsages::UNIFORM)
            }),
            constants: buffer(
                "viewer rigid mesh constants",
                bytemuck::cast_slice(&constants),
                wgpu::BufferUsages::UNIFORM | if cfg!(test) {
                    wgpu::BufferUsages::COPY_SRC
                } else { wgpu::BufferUsages::empty() },
            ),
            source: source.clone(),
            palette: palette_buffer,
            bones: buffer(
                "viewer rigid branch unused bones",
                bytemuck::cast_slice(&[0u32; 12]),
                storage,
            ),
            batch: buffer(
                "viewer rigid mesh batch",
                bytemuck::cast_slice(&[2u32, 0, 1.0f32.to_bits(), 0, 0, 0]),
                storage,
            ),
            unused: buffer(
                "viewer rigid branch unused inputs",
                bytemuck::cast_slice(&[0u32; 16]),
                storage,
            ),
            remap: buffer(
                "viewer identity vertex remap",
                bytemuck::cast_slice(&remap),
                storage,
            ),
            positions: buffer(
                "authored generated positions",
                bytemuck::cast_slice(&output),
                output_usage,
            ),
            frames: buffer(
                "authored generated frames",
                bytemuck::cast_slice(&output),
                output_usage,
            ),
            count,
            generated: std::sync::atomic::AtomicBool::new(false),
        }
    }


    pub fn bind(
        &self,
        device: &wgpu::Device,
        producer: &super::body_mesh::BodyMeshProducer,
    ) -> wgpu::BindGroup {
        if self.producer == super::authored_program::DescriptorAbi::BodyProceduralComputeStorage {
            producer.bind_procedural(device, super::body_mesh::BodyProceduralMeshBuffers {
                cb0: &self.constants,
                cb1: self.procedural_object.as_ref().expect("B762 static object scope"),
                t1: &self.source,
                t2: &self.source,
                t3: &self.source,
                t4: &self.palette,
                t5: &self.bones,
                t6: &self.batch,
                t7: &self.unused,
                t8: &self.unused,
                t9: &self.bones,
                t10: &self.remap,
                u0: &self.positions,
                u1: &self.frames,
            })
        } else if matches!(self.producer, super::authored_program::DescriptorAbi::HairMeshComputeStorage
            | super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage
            | super::authored_program::DescriptorAbi::Cloth45RowComputeStorage
            | super::authored_program::DescriptorAbi::Cloth46RowComputeStorage
            | super::authored_program::DescriptorAbi::BodyMeshAA060BComputeStorage) {
            producer.bind_hair(
                device,
                super::body_mesh::HairMeshBuffers {
                    cb0: &self.constants,
                    t0: &self.source,
                    t1: &self.source,
                    t2: &self.source,
                    t3: &self.palette,
                    t4: &self.bones,
                    t5: &self.batch,
                    t6: &self.unused,
                    t7: &self.unused,
                    // HairCS reads t8 as float4 before the mode switch. The
                    // existing zeroed F4 bones image is bounded and finite.
                    t8: &self.bones,
                    t9: &self.remap,
                    u0: &self.positions,
                    u1: &self.frames,
                },
            )
        } else {
            producer.bind(
                device,
                super::body_mesh::BodyMeshBuffers {
                    cb0: &self.constants,
                    t0: &self.source,
                    t1: &self.source,
                    t2: &self.source,
                    t3: &self.palette,
                    t4: &self.bones,
                    t5: &self.batch,
                    t6: &self.unused,
                    t7: &self.unused,
                    t8: &self.remap,
                    u0: &self.positions,
                    u1: &self.frames,
                },
            )
        }
    }

    /// Bounded zero float4 image for the inert static auxiliary path and VS
    /// t0/TEXCOORD8. Admission requires a source-paired PS which never reads
    /// TEXCOORD8. Never bind the scalar remap image to a float4 descriptor.
    pub fn inert_vertex_auxiliary(&self) -> &wgpu::Buffer {
        assert!(matches!(self.producer, super::authored_program::DescriptorAbi::HairMeshComputeStorage
            | super::authored_program::DescriptorAbi::HairMesh133RowComputeStorage
            | super::authored_program::DescriptorAbi::BodyProceduralComputeStorage
            | super::authored_program::DescriptorAbi::Cloth46RowComputeStorage));
        &self.bones
    }

    pub fn dispatch(&self) -> [u32; 3] {
        [self.count.div_ceil(64), 1, 1]
    }

    pub fn encode_once(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        producer: &super::body_mesh::BodyMeshProducer,
        bindings: &wgpu::BindGroup,
    ) {
        if !self
            .generated
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            producer.encode(encoder, bindings, self.dispatch());
        }
    }
}
