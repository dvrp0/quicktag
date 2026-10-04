//! Audited Marathon shader programs ready for direct Vulkan execution.
//!
//! Selection uses the SHA-256 of the package payload. Tag hashes are labels
//! only: patches may replace a payload without changing its tag.

use sha2::{Digest, Sha256};
use tiger_pkg::{TagHash, package_manager};

use super::technique::ShaderStage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorAbi {
    /// Exact payload identity selects a separately audited variable resource ABI.
    RunnerSurface(u16),
    /// Identity-pinned stage-2 programs with independently described resource ABI.
    RunnerDecal(u16),
    /// Global-light PS80A06213: cb0/cb12/cb13, five D2 inputs, s0/s1.
    GlobalLightPixelDense,
    /// Paired screenspace VS80A0620F: SV_VertexID, cb0/cb12 at set1.
    GlobalLightVertexDense,
    /// Pixel bindings compacted to cb0/cb1/cb12, t0..t7, s1..s3.
    BodyPixelDense,
    /// Chest PS80A9D103: cb0/cb1/cb12, eight D2 textures, t8 D3, s1..s3.
    ChestPixelDense,
    /// Sleeve PS80A9D11A: same nine-texture ABI, distinct authored program.
    SleevesPixelDense,
    /// Hands PS80A9D10F: 138 cb0 rows, nine textures with D3 t7.
    HandsPixelDense,
    /// Hardware PS80A9D0F7: 117 cb0 rows, seven textures with D3 t6.
    HardwarePixelDense,
    /// Body vertex typed buffers lowered to scalar storage buffers.
    BodyVertexScalarStorage,
    /// Layout7 VS80A9D7E6 preserves authored raw tangent sign in TEXCOORD1.w.
    RigidSignVertexScalarStorage,
    /// Translator bindings retained for the textureless C827 pixel program.
    C827Pixel,
    /// Shared layout-7 vertex program 80A9A2D7 with typed buffers lowered to scalar storage.
    SharedLayout7VertexScalarStorage,
    /// Stage2 VS80A9E95B; distinct interface, same scalar resource layout.
    RunnerDecalVertexScalarStorage,
    /// D0F2 pixel bindings compacted to cb0/cb12, t2..t4, s1.
    D0F2PixelDense,
    /// Hair vertex t0/t2/t3/t4 texel buffers lowered to dense storage buffers.
    HairVertexStorage,
    /// 14580: distinct exact source, 51 authored rows, A9DE producer.
    Hair14580VertexStorage,
    /// B84F: seven-buffer vertex ABI, 120 authored cb0 rows (149 declared).
    Hair120RowVertexStorage,
    /// BA67: distinct source interface, 120 authored cb0 rows (149 declared).
    HairBA67VertexStorage,
    /// B761: seven-buffer vertex interface, 68 authored cb0 rows (97 declared).
    BodyProceduralVertexStorage,
    /// C9C5: original interface without TC7, exact source-paired C9DD/B762 CS.
    BodyC9C5VertexStorage,
    /// B1CB: seven buffers, 32 authored cb0 rows (61 declared), A9C4 producer.
    ClothVertexStorage,
    /// B15221: distinct original interface; exact B144DD/A9C4 compute graph.
    ClothB152VertexStorage,
    /// EC03: CB1/CB12, three generated scalar streams and 30-row CB0.
    DisplacementEC03VertexStorage,
    /// C107: t0 float4 colors, t2/t3/t4 scalar generated streams, cb12/cb1.
    VertexColorStorage,
    /// Hair pixel bindings compacted to cb0/cb1/cb12, t0..t4, s1/s2.
    HairPixelDense,
    /// Solid hair A9D3:170 rows, six textures with3D t5, two samplers.
    HairSolidPixelDense,
    /// Exact A9DE producer, with a distinct 65-row image and ten source buffers.
    HairMeshComputeStorage,
    /// B7BC: 133-row procedural mesh image, raw source buffers and Frame texture.
    HairMesh133RowComputeStorage,
    /// B762: 81-row procedural image plus scope_skinning cb1.
    BodyProceduralComputeStorage,
    /// Eye vertex typed buffers lowered to dense scalar storage buffers.
    EyesVertexStorage,
    /// VS80A60029, same scalar buffers with its own authored output interface.
    RunnerVertexScalarStorage,
    /// VS80A9B7D7 reads raw float IA13 plus RigidModel/View; no compute buffers.
    RigidVertexDirectIa,
    /// B7FC: Float48 IA plus exact generated scalar position/frame buffers.
    FloatVertexScalarStorage,
    /// Eye-detail pixel bindings compacted to three cbuffers, D2/D3 textures and s1/s2.
    EyeDetailPixelDense,
    /// Face pixel bindings compacted to three cbuffers, seven textures and s1/s2/s3.
    FacePixelDense,
    /// Mesh producer: dense cb0/t0..t8/u0/u1 with typed resources lowered to storage.
    BodyMeshComputeStorage,
    /// B4CB: independently pinned source with the 14-row body compute ABI.
    BodyMeshB4CBComputeStorage,
    /// B15080: distinct 45-row deformation source, ten SRVs and two UAVs.
    Cloth45RowComputeStorage,
    /// A9C4: distinct 46-row deformation source, ten SRVs and two UAVs.
    Cloth46RowComputeStorage,
    /// Exact AA060B source: 34 authored rows, 35-row declaration, 13 buffers.
    BodyMeshAA060BComputeStorage,
    /// B152BE: exact 31-row quaternion producer and 12-buffer source ABI.
    BodyMeshB152BEComputeStorage,
    /// B7C5: exact Body graph with 15 rows, no leading-row shift.
    BodyMesh15RowComputeStorage,
    /// Head producer80A9A2D3, exactsource module with the verified rigid ABI.
    HeadMeshComputeStorage,
    /// EC0B: original A2D3 math, unused trailing 15th row; no leading shift.
    HeadMeshEC0BComputeStorage,
    /// A60035 head producer: verified 15-row image with one leading cb0 row.
    HeadMesh15RowA60035ComputeStorage,
    /// B8BD head producer: verified 15-row body-shaped image with one leading cb0 row.
    HeadMesh15RowB8BDComputeStorage,
}

impl DescriptorAbi {
    pub(crate) fn vertex_constant_contract(self) -> Option<(usize, usize)> {
        if self == Self::DisplacementEC03VertexStorage {
            Some((1, 30))
        } else {
            self.auxiliary_vertex_contract().map(|(authored, declared, _)| (authored, declared))
        }
    }

    /// Producer pairing belongs to exact source identity, not IA or character.
    pub(crate) fn static_vertex_producer(self) -> Option<Self> {
        if self == Self::DisplacementEC03VertexStorage {
            Some(Self::HeadMeshEC0BComputeStorage)
        } else {
            self.auxiliary_vertex_contract().map(|(_, _, producer)| producer)
        }
    }

    /// Distinct source identities share a descriptor topology, not constants
    /// or a producer inferred from the character or vertex layout.
    pub(crate) fn auxiliary_vertex_contract(self) -> Option<(usize, usize, Self)> {
        match self {
            Self::HairVertexStorage => Some((51, 80, Self::HairMeshComputeStorage)),
            Self::Hair14580VertexStorage => Some((51, 80, Self::HairMeshComputeStorage)),
            Self::Hair120RowVertexStorage => Some((120, 149, Self::HairMesh133RowComputeStorage)),
            Self::HairBA67VertexStorage => Some((120, 149, Self::HairMesh133RowComputeStorage)),
            Self::BodyProceduralVertexStorage => Some((68, 97, Self::BodyProceduralComputeStorage)),
            Self::BodyC9C5VertexStorage => Some((68, 97, Self::BodyProceduralComputeStorage)),
            Self::ClothVertexStorage => Some((32, 61, Self::Cloth46RowComputeStorage)),
            Self::ClothB152VertexStorage => Some((32, 61, Self::Cloth46RowComputeStorage)),
            _ => None,
        }
    }

    /// Authored and declared CB0 rows for source-proved packed deformation
    /// producers. AA060B's final declared row is never loaded by its source.
    pub(crate) fn deformation_constant_contract(self) -> Option<(usize, usize)> {
        match self {
            Self::Cloth45RowComputeStorage => Some((45, 45)),
            Self::Cloth46RowComputeStorage => Some((46, 46)),
            Self::BodyMeshAA060BComputeStorage => Some((34, 35)),
            Self::BodyMeshB152BEComputeStorage => Some((31, 31)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AuthoredProgram {
    pub shader_tag: TagHash,
    pub stage: ShaderStage,
    pub source_sha256: [u8; 32],
    pub spirv_sha256: [u8; 32],
    pub descriptor_abi: DescriptorAbi,
    pub spirv: &'static [u8],
}

impl AuthoredProgram {
    pub fn create_shader_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        let label = format!("authored {} {:?}", self.shader_tag, self.stage);
        // SAFETY: every embedded artifact is selected by original payload
        // identity, pinned below, validated for Vulkan 1.2 and rehashed by the
        // unit test. ABI-specific pipeline code must still honor descriptor_abi.
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(&label),
                entry_point: "main".into(),
                spirv: Some(wgpu::util::make_spirv_raw(self.spirv)),
                ..Default::default()
            })
        }
    }
}

const BODY_PS: &[u8] = include_bytes!("../../assets/authored/goliath/body-80A9D0E1.ps.spv");
const CHEST_PS: &[u8] = include_bytes!("../../assets/authored/goliath/chest-80A9D103.ps.spv");
const SLEEVES_PS: &[u8] = include_bytes!("../../assets/authored/goliath/sleeves-80A9D11A.ps.spv");
const HANDS_PS: &[u8] = include_bytes!("../../assets/authored/goliath/hands-80A9D10F.ps.spv");
const HARDWARE_PS: &[u8] = include_bytes!("../../assets/authored/goliath/hardware-80A9D0F7.ps.spv");
const BODY_VS: &[u8] = include_bytes!("../../assets/authored/goliath/body-80A60DAD.vs.spv");
const C827_PS: &[u8] = include_bytes!("../../assets/authored/goliath/c827-80A9A9FB.ps.spv");
const C827_VS: &[u8] = include_bytes!("../../assets/authored/goliath/c827-80A9A2D7.vs.spv");
const D0F2_PS: &[u8] = include_bytes!("../../assets/authored/goliath/d0f2-80A9B4C7.ps.spv");
const HAIR_VS: &[u8] = include_bytes!("../../assets/authored/goliath/hair-80A9A9DB.vs.spv");
const HAIR_PS: &[u8] = include_bytes!("../../assets/authored/goliath/hair-80A9A9D5.ps.spv");
const EYES_VS: &[u8] = include_bytes!("../../assets/authored/goliath/eyes-80A9A2C7.vs.spv");
const EYE_DETAIL_PS: &[u8] =
    include_bytes!("../../assets/authored/goliath/eye-detail-80A9BE2C.ps.spv");
const FACE_PS: &[u8] = include_bytes!("../../assets/authored/goliath/face-80A9D0DF.ps.spv");
const BODY_MESH_CS: &[u8] = include_bytes!("../../assets/authored/goliath/body-80A9BE34.cs.spv");
const HEAD_MESH_15ROW_A60035_CS: &[u8] =
    include_bytes!("../../assets/authored/goliath/runner-80A60035.compute.ssbo.spv");
const HEAD_MESH_15ROW_B8BD_CS: &[u8] =
    include_bytes!("../../assets/authored/goliath/runner-80A9B8BD.compute.ssbo.spv");

pub const PROGRAMS: &[AuthoredProgram] = &[
    AuthoredProgram {
        shader_tag: TagHash(0x80B152BE), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("0c0e1c7b7dbada89d65262310e5fa5ed352d9d48f02220556828af31fda5cea5"),
        spirv_sha256: decode_sha256("9721964f66d169235d103e4347f4e85c694b03cab7acec3d5ab2748cbf3e8a6c"),
        descriptor_abi: DescriptorAbi::BodyMeshB152BEComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80B152BE.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9C9C5), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("26b4b950a1825a4ea65c6249467d7290c799a160b36f864b14caa9f867d2818f"),
        spirv_sha256: decode_sha256("c59c7a3adcb3214cf149cf9897e158c05a4902991d4a32472cac1aa86b66b45e"),
        descriptor_abi: DescriptorAbi::BodyC9C5VertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9C9C5.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9C9DD), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("435230e7ac936b0f47461538d315e42e84a9388c29b07f63a531207aa995a449"),
        spirv_sha256: decode_sha256("2091c0379bf8a96b6010b1c4b5d58f9e276bf978a079b9905be7d7424198319d"),
        // Independent source, byte-identical verified compute executable.
        descriptor_abi: DescriptorAbi::BodyProceduralComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9C9DD.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9EC03), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("3fa1d2733c51b4364ce8155a412f8400b68099ba17b133664e48b99010d0608b"),
        spirv_sha256: decode_sha256("4247940cd629434b5b7e2bc777f256be9264776c701fd785e1847573ae058809"),
        descriptor_abi: DescriptorAbi::DisplacementEC03VertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9EC03.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9EC0B), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("14d0d631bd375c1b31e07d598dbb0e2466e7f219d9fc82e121aba9a7b5f3a931"),
        spirv_sha256: decode_sha256("cd4270550f844b7873dfe210618ab580c0f33b76edce011daef89b83decabdc9"),
        descriptor_abi: DescriptorAbi::HeadMeshEC0BComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9EC0B.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80AA060B), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("c406943e4a6a613dcdd7ff477341158db8e43ef5075b5e31ca765d0c30245c4c"),
        spirv_sha256: decode_sha256("dbaf2bd3e47b3e2c1db9bc25581c6b62f6a5506eef57a83c6c91689ffc847194"),
        descriptor_abi: DescriptorAbi::BodyMeshAA060BComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/body-80AA060B.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80B15221), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("bfea289787b3dbdc7614c6b284423dcc275281ac9bbf7e2185f1b3f946daa000"),
        spirv_sha256: decode_sha256("d3852ac937764085a0f52c4225d1d2d49b55b9ca15b32293e92e8c3bde551b1b"),
        descriptor_abi: DescriptorAbi::ClothB152VertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cloth-80B15221.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80B144DD), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("c374a0ff2370f41f68ffab69d60da2510e3a0adceceecc0fbc0d566ae123e42b"),
        spirv_sha256: decode_sha256("7c2eb222f9902ed34b860961f2c3270994dbd20daef36fe8f25a5487116c6aa7"),
        // Independently proved byte-identical shader, so the cache ABI is
        // shared. The original package source identity remains distinct.
        descriptor_abi: DescriptorAbi::Cloth46RowComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cloth-80B144DD.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B1CB), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("09239067a159a349436e5cceef5a913d0fc538c3f5a35d7d4c378f8f084531a9"),
        spirv_sha256: decode_sha256("158fedb6be7b0d49f7dc172b7f3fa83bbb900aac800ce3f63c26a70ec06a7ee6"),
        descriptor_abi: DescriptorAbi::ClothVertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cloth-80A9B1CB.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9C4), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("fc0b9d1ec776ba68e7e5d9b3a034360d757ffb867af86c324789d75ea18ed722"),
        spirv_sha256: decode_sha256("7c2eb222f9902ed34b860961f2c3270994dbd20daef36fe8f25a5487116c6aa7"),
        descriptor_abi: DescriptorAbi::Cloth46RowComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cloth-80A9A9C4.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80B15080), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("e20774672aa79e97a34e41fcbdc8d3954f7e24e9f8b9e4912d4ce20dd8650562"),
        spirv_sha256: decode_sha256("ad42caf2d4c8f92738c120f1a4c10b9b6d8584c2e86393018251ed22be280434"),
        descriptor_abi: DescriptorAbi::Cloth45RowComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cloth-80B15080.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9C107), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("694a53889fb448a50a3bd02ae8e59b26d41a98215af0ae19887966b2cf8dc083"),
        spirv_sha256: decode_sha256("6b353bf257089c1ddc164e5aa0ec50edb9bc56030756197b884f96587f74101a"),
        descriptor_abi: DescriptorAbi::VertexColorStorage,
        spirv: include_bytes!("../../assets/authored/goliath/color-80A9C107.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B4CB), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("1df93693a5f3752ce9a8b1c27db175e07f87765cf3f33d58b37729b3017b8e1a"),
        spirv_sha256: decode_sha256("a095c0d2b92798c6e892c3c82a0c0270d0462e3ba40537cb0e3f6a0299357ab7"),
        descriptor_abi: DescriptorAbi::BodyMeshB4CBComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/body-80A9B4CB.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B761), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("7eaf9a7ed704dba604547c6bdb05c63c91a72f0b4a768014c79d5eab2e57d9ba"),
        spirv_sha256: decode_sha256("8a166e5bf6688ae5286c7319982e0b7f50ddd43ea52acb58b9fb2e49a1b3abdf"),
        descriptor_abi: DescriptorAbi::BodyProceduralVertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/body-procedural-80A9B761.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B762), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("25f8bbe7a227113622ded54c5e42e6092d372716ca98217bfe421e14529b3707"),
        spirv_sha256: decode_sha256("2091c0379bf8a96b6010b1c4b5d58f9e276bf978a079b9905be7d7424198319d"),
        descriptor_abi: DescriptorAbi::BodyProceduralComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/body-procedural-80A9B762.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9BA67), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("37dd5ff032f2aa6a7f1800b36cf226abf252175a3b9c02d2b6dfea4bad814f9a"),
        spirv_sha256: decode_sha256("c8b0ba8d6cbf582445cefd6c126a6a9a9f4bfe66902513f694077bde8862ea25"),
        descriptor_abi: DescriptorAbi::HairBA67VertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80A9BA67.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B7BC), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("bfd692e2de4b9a1ededa068ce6b59487a78556741d59775a0ba8614fd3afe6ed"),
        spirv_sha256: decode_sha256("fad25659b0cfc96f984d37c05fc36fd3d81017c14f6e888174d28e719fb10083"),
        descriptor_abi: DescriptorAbi::HairMesh133RowComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80A9B7BC.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B84F), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("fd7b7eb9de309e46b36453f63806ea67397d1c44b2baae3fa5b967959a722d47"),
        spirv_sha256: decode_sha256("a4e18c781381151a6ab5dea6a4cc3e75888cc7084e6d6df0bbaeef15ac00f174"),
        descriptor_abi: DescriptorAbi::Hair120RowVertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80A9B84F.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9D3), stage: ShaderStage::Pixel,
        source_sha256: decode_sha256("f988a7d4f4b67f337badf2bf4d9c25d5b4d53cf9712bae65beecd062fc4868cd"),
        spirv_sha256: decode_sha256("e804db4ea2830590a17eb24c62800e9f6113bb6d298fd68b0b174d560a5742e1"),
        descriptor_abi: DescriptorAbi::HairSolidPixelDense,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80A9A9D3.ps.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9DE), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("c0735f6428b3d2b9d1b6acc7b6ff1e671f8eacc87d773bc7ac0fa64d663f4ff5"),
        spirv_sha256: decode_sha256("f8e1f49ebcd793b4b3636fac7a50fd30ce1c817d1f6e83c62623ef35b9d14ad2"),
        descriptor_abi: DescriptorAbi::HairMeshComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80A9A9DE.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B7FC), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("67ac7622d5aa14d14f6cc0e0134495f57d519f6b206aac47374a477f589bbd07"),
        spirv_sha256: decode_sha256("c50d7a71e4a2e45dd4cee59e8cf87b0f99162bea989a85fa8f99defd3f28ef81"),
        descriptor_abi: DescriptorAbi::FloatVertexScalarStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9B7FC.vertex.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B7C5), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("a30ab41f71e4c19a8e6fbe4a6deba19b1c696cd29c6754cebd5ae03274616345"),
        spirv_sha256: decode_sha256("df42467c79703f57bf80763a84f4c9dcaeaea06a8c557842911534acaa6c2d21"),
        descriptor_abi: DescriptorAbi::BodyMesh15RowComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9B7C5.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B7D7), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("af201bfa106102273382302d82e61f1a87e0aa8498d0ff11d0319f61c9aa4184"),
        spirv_sha256: decode_sha256("8f4a2afd031d1e4b04fc6bcfbc61c1edebb6525740079125b7a09a73047a39fa"),
        descriptor_abi: DescriptorAbi::RigidVertexDirectIa,
        spirv: include_bytes!("../../assets/authored/goliath/rigid-80A9B7D7.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A60029), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("0ed49e3be5a96d513ba49ad7c604775aa563ba72eef781934975389a5ad2f51b"),
        spirv_sha256: decode_sha256("1b708fc523b2869f9489dc99d54e7c2b65d70ab40b2aa926e44f25b3ce0d7813"),
        descriptor_abi: DescriptorAbi::RunnerVertexScalarStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A60029.vertex.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A2D3), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("254f74fa6cf3e44598d55e62bd156a8637289972e543333550a0f48666bccd69"),
        spirv_sha256: decode_sha256("d5ad8ec6d23977ba6070ccc5bd2f0ac0f9ec814a86370161a2b9e5bd760b1238"),
        descriptor_abi: DescriptorAbi::HeadMeshComputeStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9A2D3.compute.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A60035), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("43dbec873fb63bf32f9ac15dc28b9de749f9b94eaeb54870b65398b07b73dfc4"),
        spirv_sha256: decode_sha256("cc893e3b1738b7e48c7268d3f5e8a7f7d7e8d23665287272c54cc58f3c25ce49"),
        descriptor_abi: DescriptorAbi::HeadMesh15RowA60035ComputeStorage,
        spirv: HEAD_MESH_15ROW_A60035_CS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B8BD), stage: ShaderStage::Compute,
        source_sha256: decode_sha256("d56962c1ad10d9c2e073b3572fbac97f9aa51fdc8559184cfad74f1b98618130"),
        spirv_sha256: decode_sha256("961618a87f6c5b441bb587c457d2342246a66991039861036576f4538daa8c9c"),
        descriptor_abi: DescriptorAbi::HeadMesh15RowB8BDComputeStorage,
        spirv: HEAD_MESH_15ROW_B8BD_CS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9E95B), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("4060211d2fd5cd5955af1b1acb847d77fb8136d5ad1d43447732bf7070bd9aae"),
        spirv_sha256: decode_sha256("6d84191b30cf27208c50d79d9132bcdeb3c01bc689398ba9833e59ce9ff4fb1a"),
        descriptor_abi: DescriptorAbi::RunnerDecalVertexScalarStorage,
        spirv: include_bytes!("../../assets/authored/goliath/runner-80A9E95B.vertex.ssbo.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D7E6), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("3dbc1ddc76b3f7c66d202e25a74e3f7775413cc05f55995d91b1d9d71f988601"),
        spirv_sha256: decode_sha256("172e169a4179c8e9c62eb9ec4f741f7034463d0fe06f939d65a0ec28499f86bc"),
        descriptor_abi: DescriptorAbi::RigidSignVertexScalarStorage,
        spirv: include_bytes!("../../assets/authored/goliath/cyber-80A9D7E6.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A06213), stage: ShaderStage::Pixel,
        source_sha256: decode_sha256("ce8dd555c74526c2fe95086dbc321b8fbcac56853bbed602a06864265139f1d9"),
        spirv_sha256: decode_sha256("075195044e347f4e392b7c2e715332dd19a75e3e744e9484eccbc969745afae2"),
        descriptor_abi: DescriptorAbi::GlobalLightPixelDense,
        spirv: include_bytes!("../../assets/authored/goliath/global-light-80A06213.ps.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A0620F), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256("46c58d92cba8670847a7144f3e2c317cfc0ce50740addae8d352c8074329621e"),
        spirv_sha256: decode_sha256("722c3f3907816290bdc05fdaab1fd790d30d430d235a10843063542f8af3f454"),
        descriptor_abi: DescriptorAbi::GlobalLightVertexDense,
        spirv: include_bytes!("../../assets/authored/goliath/global-light-80A0620F.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D0F7), stage: ShaderStage::Pixel,
        source_sha256: decode_sha256("f183fc229b26183267dcf253bf5ccb90cfab5c5798a58dbd06ff1bfcbcb29628"),
        spirv_sha256: decode_sha256("98462e1458cf256f5dee61481a0ba11890aab67be56f61e59bc3d8d48146a6d9"),
        descriptor_abi: DescriptorAbi::HardwarePixelDense, spirv: HARDWARE_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D10F), stage: ShaderStage::Pixel,
        source_sha256: decode_sha256("b18cb6d2fe722b9b01d6415b4f48f85687fe60286540279d1269ece3029c03b7"),
        spirv_sha256: decode_sha256("28a13b9e16e771fdf6b5634dcd33b20041c0d5bd7b7f670153175d114ac7740b"),
        descriptor_abi: DescriptorAbi::HandsPixelDense, spirv: HANDS_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D0E1),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "a1bd629f733a0f46d1e28d823066c9e820c53b18f2622d3baa8de2db31602855",
        ),
        spirv_sha256: decode_sha256(
            "8b058a35f68f8268f99cf483ed12eca1129e83fe8d8f56d4a4440f79ed32ae41",
        ),
        descriptor_abi: DescriptorAbi::BodyPixelDense,
        spirv: BODY_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D103),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "af6c988a9deb826f33623f04bceb8d34637e3e304f7902ed40e30c0e975637ed",
        ),
        spirv_sha256: decode_sha256(
            "267654ed5477a1f2f771fbb5970cd45157b2cec8eff8d6ecccfbc4fb583cc738",
        ),
        descriptor_abi: DescriptorAbi::ChestPixelDense,
        spirv: CHEST_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D11A),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "8801f1c2fcefd75d1bf4996372758dc822db1666f61d83384df3376257474608",
        ),
        spirv_sha256: decode_sha256(
            "a0b8a7345619c412e5fcce41259b6cbb483a1fdb0fb4fd0b4b5428bba02d1b20",
        ),
        descriptor_abi: DescriptorAbi::SleevesPixelDense,
        spirv: SLEEVES_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A60DAD),
        stage: ShaderStage::Vertex,
        source_sha256: decode_sha256(
            "4635e508deb96ec0cbaff3fad0a5e3fef094baa31784e428e27ce58124306380",
        ),
        spirv_sha256: decode_sha256(
            "5daa6b17e30824aa32e88eb3dcef27ed9bf557e54efc6ba17177e02bb99e0b40",
        ),
        descriptor_abi: DescriptorAbi::BodyVertexScalarStorage,
        spirv: BODY_VS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A2D7),
        stage: ShaderStage::Vertex,
        source_sha256: decode_sha256(
            "e6b272fdbe6d10bcc5214fe0f84546ec375fe541716885f5a89ee5885fc8a76b",
        ),
        spirv_sha256: decode_sha256(
            "88062c236d8fecb4720c8895d56887473031ca743d04be7386d81304fdcc4657",
        ),
        descriptor_abi: DescriptorAbi::SharedLayout7VertexScalarStorage,
        spirv: C827_VS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9FB),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "35f80c341dff4287fff1efb94e5458596c4ce7f76483bfdeb6d5ce53d1e84a00",
        ),
        spirv_sha256: decode_sha256(
            "cbd256c91f781732ea3eeec298a0ae648c96187c968322a95ae58a701c261fa2",
        ),
        descriptor_abi: DescriptorAbi::C827Pixel,
        spirv: C827_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9B4C7),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "df3bca83eac6b7b48150e31d4cfe936a393865eb8071359a2a73ce5dac04e56c",
        ),
        spirv_sha256: decode_sha256(
            "39b9d7ed0da8e22264c3887f0024da7e912fb0c8723867f8613219966d851bd0",
        ),
        descriptor_abi: DescriptorAbi::D0F2PixelDense,
        spirv: D0F2_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9DB),
        stage: ShaderStage::Vertex,
        source_sha256: decode_sha256(
            "f4e8602fb4de520e9d978f5c20c083074ae84ba3fb97595eb023f7bbe6852a10",
        ),
        spirv_sha256: decode_sha256(
            "e83a197280ede31f28ad7dbb3d148c116938a54c7f3cf23a28b374143bd544bf",
        ),
        descriptor_abi: DescriptorAbi::HairVertexStorage,
        spirv: HAIR_VS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80B14580), stage: ShaderStage::Vertex,
        source_sha256: decode_sha256(
            "cb048590ba7a6c485fb8f721cd60e0a8e18bd52a712589aa1a1e4859d917ef8c",
        ),
        spirv_sha256: decode_sha256(
            "da7206295d3b38073488498ad611c8cb487d7fb9327f840b02a0e3f893a11f8e",
        ),
        descriptor_abi: DescriptorAbi::Hair14580VertexStorage,
        spirv: include_bytes!("../../assets/authored/goliath/hair-80B14580.vs.spv"),
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A9D5),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "9a78c416fbf9086a5dd9ec6747f982be6a8d76faead3837e54c2e5adcd5d4a4c",
        ),
        spirv_sha256: decode_sha256(
            "68da50ca531c74b9cb276ffab35dacacae7fa3c1bb088af1ebc0c864a0876cf9",
        ),
        descriptor_abi: DescriptorAbi::HairPixelDense,
        spirv: HAIR_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A2C7),
        stage: ShaderStage::Vertex,
        source_sha256: decode_sha256(
            "b45b7d5b32d67909f37153cdee7a120cf2b29b411efe1487d23b91fbfcfb57e4",
        ),
        spirv_sha256: decode_sha256(
            "cd43cf2e41b9d699187274fc39c46a76b64de754127420553707d650b4da502c",
        ),
        descriptor_abi: DescriptorAbi::EyesVertexStorage,
        spirv: EYES_VS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9BE2C),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "df49f149dc553d7c365ce8f4cce50c9b53fe5df036ef249951100306084d0824",
        ),
        spirv_sha256: decode_sha256(
            "a5fcadd69a2c91fd1a3bc8505677148c2cd34615da8609a5a0197f2676ebaa55",
        ),
        descriptor_abi: DescriptorAbi::EyeDetailPixelDense,
        spirv: EYE_DETAIL_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9D0DF),
        stage: ShaderStage::Pixel,
        source_sha256: decode_sha256(
            "8ba785bad518764b723024cfbcb58a16b09ff4cb7fa0c0c137af7f50a8083f78",
        ),
        spirv_sha256: decode_sha256(
            "61f59afb7a63f1f852685261a609a624d04521b5e37af18f4fc2da47217f13d6",
        ),
        descriptor_abi: DescriptorAbi::FacePixelDense,
        spirv: FACE_PS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9BE34),
        stage: ShaderStage::Compute,
        source_sha256: decode_sha256(
            "6f32d69ba3be5202fff8856b6514b36930b304bb4302fd64cd68c0164dcb41aa",
        ),
        spirv_sha256: decode_sha256(
            "a095c0d2b92798c6e892c3c82a0c0270d0462e3ba40537cb0e3f6a0299357ab7",
        ),
        descriptor_abi: DescriptorAbi::BodyMeshComputeStorage,
        spirv: BODY_MESH_CS,
    },
    AuthoredProgram {
        shader_tag: TagHash(0x80A9A2E2),
        stage: ShaderStage::Compute,
        source_sha256: decode_sha256(
            "23a2a3340da9ae3dd074fc4e36886f9fa7a2e1c3c4072cd2b681fa9f23272356",
        ),
        spirv_sha256: decode_sha256(
            "a095c0d2b92798c6e892c3c82a0c0270d0462e3ba40537cb0e3f6a0299357ab7",
        ),
        descriptor_abi: DescriptorAbi::BodyMeshComputeStorage,
        spirv: BODY_MESH_CS,
    },
];

pub fn programs() -> impl Iterator<Item = &'static AuthoredProgram> {
    PROGRAMS.iter().chain(super::runner_surface_programs::SURFACES.iter().map(|s| &s.program))
        .chain(super::runner_decal_programs::DECALS.iter().map(|s| &s.program))
}

pub struct AuthoredProgramModules {
    device: wgpu::Device,
    entries: Vec<(&'static AuthoredProgram, std::sync::OnceLock<wgpu::ShaderModule>)>,
}

impl AuthoredProgramModules {
    pub fn new(device: &wgpu::Device) -> Self {
        // module() caches by descriptor identity. Sharing that identity is
        // valid only for byte-identical executable shaders, even when the
        // original package payload hashes differ.
        let mut identities = std::collections::HashMap::new();
        let entries = programs().map(|program| {
            let identity = (program.stage, program.spirv_sha256);
            if let Some(previous) = identities.insert(program.descriptor_abi, identity) {
                assert_eq!(previous, identity, "Conflicting shader cache identity {:?}",
                    program.descriptor_abi);
            }
            (program, std::sync::OnceLock::new())
        }).collect();
        Self {
            device: device.clone(),
            entries,
        }
    }

    pub fn module(&self, abi: DescriptorAbi) -> Option<&wgpu::ShaderModule> {
        self.entries
            .iter()
            .find(|(program, _module)| program.descriptor_abi == abi)
            .map(|(program, module)| module.get_or_init(|| program.create_shader_module(&self.device)))
    }
}

pub fn find_by_source_sha256(
    stage: ShaderStage,
    source_sha256: [u8; 32],
) -> Option<&'static AuthoredProgram> {
    programs()
        .find(|program| program.stage == stage && program.source_sha256 == source_sha256)
}

pub fn resolve_package_program(
    shader_tag: TagHash,
    stage: ShaderStage,
) -> Result<Option<&'static AuthoredProgram>, String> {
    let Some(entry) = package_manager().get_entry(shader_tag) else {
        return Err(format!("missing shader tag {shader_tag}"));
    };
    let payload = package_manager()
        .read_tag(TagHash(entry.reference))
        .map_err(|error| format!("failed reading shader payload {shader_tag}: {error}"))?;
    let digest: [u8; 32] = Sha256::digest(payload).into();
    Ok(find_by_source_sha256(stage, digest))
}

pub(crate) const fn decode_sha256(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    assert!(bytes.len() == 64);
    let mut result = [0; 32];
    let mut index = 0;
    while index < result.len() {
        result[index] = (hex_digit(bytes[index * 2]) << 4) | hex_digit(bytes[index * 2 + 1]);
        index += 1;
    }
    result
}

const fn hex_digit(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        b'A'..=b'F' => value - b'A' + 10,
        _ => panic!("invalid SHA-256 hex digit"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_programs_match_pinned_artifact_hashes() {
        for program in programs() {
            let actual: [u8; 32] = Sha256::digest(program.spirv).into();
            assert_eq!(actual, program.spirv_sha256, "{}", program.shader_tag);
        }
    }

    #[test]
    fn lookup_requires_stage_and_payload_identity() {
        let body = PROGRAMS.iter().find(|program| program.descriptor_abi == DescriptorAbi::BodyPixelDense).unwrap();
        assert_eq!(
            find_by_source_sha256(ShaderStage::Pixel, body.source_sha256)
                .map(|program| program.shader_tag),
            Some(TagHash(0x80A9D0E1))
        );
        assert!(find_by_source_sha256(ShaderStage::Vertex, body.source_sha256).is_none());
        assert!(find_by_source_sha256(ShaderStage::Pixel, [0; 32]).is_none());
    }
}
