//! Marathon animation clips (`8080AE01`) and the humanoid runner rig.
//!
//! A clip stores a static pose chunk plus one animated chunk in one of three
//! codecs. Tracks address clip slots, not skeleton node indices: for the
//! runner rig the 26 leading slots follow the authoring order and mostly hold
//! object-space rotations, while the remaining slots are rotations applied on
//! top of a node's bind-local rotation. Skeleton nodes are matched by the
//! FNV-1 hash of their `b_*` names so one clip drives every runner shell.

use std::sync::OnceLock;

use rayon::iter::{IntoParallelRefIterator, ParallelExtend, ParallelIterator};

use quicktag_core::util::fnv1;
use tiger_pkg::{TagHash, package_manager};

const CLASS_ANIMATION_CLIP: u32 = 0x8080AE01;
/// Ranged per-frame samples of the object channels a clip animates.
const CLASS_CHANNEL_SAMPLES: u32 = 0x8080B14A;
const CLASS_PATTERN: u32 = 0x8080BADB;
const CLASS_ARRAY: u32 = 0x8080BFCD;
const CLASS_SKELETON_NODE_HIERARCHY: u32 = 0x8080AF42;
const CLASS_SKELETON_TRANSFORMS: u32 = 0x8080BF47;
const CODEC_RAW: u32 = 0x8080B13B;
const CODEC_RANGED: u32 = 0x8080B13E;
const CODEC_CURVE: u32 = 0x8080B13F;
/// Slot count of the shared runner rig every shell's common clips target.
const RUNNER_CLIP_NODES: usize = 81;
pub const CLIP_FRAMES_PER_SECOND: f32 = 30.0;

pub type Quat = [f32; 4];

fn u16_at(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn u32_at(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn f32_at(data: &[u8], offset: usize) -> Option<f32> {
    u32_at(data, offset).map(f32::from_bits)
}

/// (count, data offset) of the array whose (count, relative pointer) pair sits at `offset`.
fn array_at(data: &[u8], offset: usize) -> Option<(usize, usize)> {
    let count = u64::from_le_bytes(data.get(offset..offset + 8)?.try_into().ok()?) as usize;
    let pointer = u64::from_le_bytes(data.get(offset + 8..offset + 16)?.try_into().ok()?) as usize;
    Some((count, offset + 8 + pointer + 0x10))
}

fn u16_array(data: &[u8], offset: usize) -> Option<Vec<u16>> {
    let (count, start) = array_at(data, offset)?;
    data.get(start..start + count * 2).map(|bytes| {
        bytes
            .chunks_exact(2)
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect()
    })
}

fn f32_array(data: &[u8], offset: usize) -> Option<Vec<f32>> {
    let (count, start) = array_at(data, offset)?;
    data.get(start..start + count * 4).map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    })
}

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_conjugate(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn quat_rotate(q: Quat, v: [f32; 3]) -> [f32; 3] {
    let p = quat_mul(quat_mul(q, [v[0], v[1], v[2], 0.0]), quat_conjugate(q));
    [p[0], p[1], p[2]]
}

fn normalize(q: Quat) -> Quat {
    let length = q.iter().map(|c| c * c).sum::<f32>().sqrt();
    if length > 1e-9 {
        q.map(|c| c / length)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn nlerp(a: Quat, b: Quat, t: f32) -> Quat {
    let sign = if a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>() < 0.0 {
        -1.0
    } else {
        1.0
    };
    normalize(std::array::from_fn(|c| a[c] * (1.0 - t) + b[c] * sign * t))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub rotation: Quat,
    pub translation: [f32; 3],
    /// Uniform scale, applied before the rotation.
    pub scale: f32,
}

/// Skeleton carried by an entity pattern, in the node order vertices index.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub tag: TagHash,
    pub names: Vec<u32>,
    pub parents: Vec<i32>,
    /// Object-space bind transform per node.
    pub bind: Vec<Transform>,
}

impl Skeleton {
    pub fn load(tag: TagHash) -> Option<Self> {
        let data = package_manager().read_tag(tag).ok()?;
        let (mut names, mut parents, mut bind) = (vec![], vec![], vec![]);
        for offset in (0..data.len().saturating_sub(20)).step_by(4) {
            if u32_at(&data, offset) != Some(CLASS_ARRAY) {
                continue;
            }
            let count = u32_at(&data, offset + 4)? as usize;
            let start = offset + 20;
            match u32_at(&data, offset + 12)? {
                CLASS_SKELETON_NODE_HIERARCHY if names.is_empty() => {
                    for record in data.get(start..start + count * 0x10)?.chunks_exact(0x10) {
                        names.push(u32_at(record, 0)?);
                        parents.push(u32_at(record, 4)? as i32);
                    }
                }
                // The first full-length transform array after the hierarchy is the bind pose.
                CLASS_SKELETON_TRANSFORMS
                    if !names.is_empty() && count == names.len() && bind.is_empty() =>
                {
                    for record in data.get(start..start + count * 0x20)?.chunks_exact(0x20) {
                        let value = |index: usize| f32_at(record, index * 4);
                        bind.push(Transform {
                            rotation: [value(0)?, value(1)?, value(2)?, value(3)?],
                            translation: [value(4)?, value(5)?, value(6)?],
                            scale: 1.0,
                        });
                    }
                }
                _ => {}
            }
        }
        (!names.is_empty() && bind.len() == names.len()).then_some(Self {
            tag,
            names,
            parents,
            bind,
        })
    }

    pub fn node(&self, name: &str) -> Option<usize> {
        let hash = fnv1(name.as_bytes());
        self.names.iter().position(|candidate| *candidate == hash)
    }

    pub fn is_runner(&self) -> bool {
        ["b_pelvis", "b_spine_3", "b_l_hand", "b_r_foot"]
            .iter()
            .all(|name| self.node(name).is_some())
    }

    /// A runner body or head rig from the packages the player's own entities live in.
    pub fn is_player_rig(&self) -> bool {
        self.node("b_head").is_some()
            && package_manager()
                .package_paths
                .get(&self.tag.pkg_id())
                .is_some_and(|path| matches!(path.name.as_str(), "sr_sandbox" | "sr_gear"))
    }

    fn bind_local(&self, node: usize) -> Transform {
        let bind = self.bind[node];
        let Some(parent) = usize::try_from(self.parents[node])
            .ok()
            .and_then(|p| self.bind.get(p))
        else {
            return bind;
        };
        let inverse = quat_conjugate(parent.rotation);
        Transform {
            rotation: quat_mul(inverse, bind.rotation),
            translation: quat_rotate(
                inverse,
                std::array::from_fn(|c| bind.translation[c] - parent.translation[c]),
            ),
            scale: 1.0,
        }
    }

    /// Object-space node transforms following another rig: every node that
    /// rig also has (by name) moves as that rig's node does, and the rest are
    /// carried along under their posed parents.
    ///
    /// `moved` holds the other rig's skinning transforms, not its poses: two
    /// rigs may rest the same node differently (a head rig's jaw and eyes
    /// against the body rig's), and it is the movement that is shared.
    pub fn retarget(&self, moved: &rustc_hash::FxHashMap<u32, Transform>) -> Vec<Transform> {
        let mut pose: Vec<Transform> = Vec::with_capacity(self.names.len());
        for node in 0..self.names.len() {
            let parent = usize::try_from(self.parents[node])
                .ok()
                .and_then(|p| pose.get(p).copied());
            pose.push(match (moved.get(&self.names[node]), parent) {
                (Some(moved), _) => compose(*moved, self.bind[node]),
                (None, Some(parent)) => compose(parent, self.bind_local(node)),
                (None, None) => self.bind[node],
            });
        }
        pose
    }

    /// Object-space node transforms for a clip storing plain parent-local
    /// transforms, one slot per node. Nodes the clip leaves out keep their
    /// bind-local transform.
    pub fn local_pose(&self, samples: &[SlotSample]) -> Vec<Transform> {
        let mut pose: Vec<Transform> = Vec::with_capacity(self.names.len());
        for node in 0..self.names.len() {
            let bind = self.bind_local(node);
            let sample = samples.get(node).copied().unwrap_or_default();
            let local = Transform {
                rotation: sample.rotation.unwrap_or(bind.rotation),
                translation: sample.translation.unwrap_or(bind.translation),
                scale: sample.scale.unwrap_or(bind.scale),
            };
            let parent = usize::try_from(self.parents[node])
                .ok()
                .and_then(|p| pose.get(p).copied());
            pose.push(match parent {
                Some(parent) => compose(parent, local),
                None => local,
            });
        }
        pose
    }

    /// Per node: the transform carrying bind-pose object space to posed object space.
    pub fn skinning_transforms(&self, pose: &[Transform]) -> Vec<Transform> {
        self.bind
            .iter()
            .zip(pose)
            .map(|(bind, posed)| {
                let rotation = normalize(quat_mul(posed.rotation, quat_conjugate(bind.rotation)));
                let scale = posed.scale / bind.scale;
                let rotated = quat_rotate(rotation, bind.translation);
                Transform {
                    rotation,
                    translation: std::array::from_fn(|c| posed.translation[c] - rotated[c] * scale),
                    scale,
                }
            })
            .collect()
    }
}

/// Every skeleton any entity pattern carries, loaded once, in tag order.
pub fn all_skeletons() -> &'static [Skeleton] {
    static SKELETONS: OnceLock<Vec<Skeleton>> = OnceLock::new();
    SKELETONS.get_or_init(|| {
        let mut tags = package_manager()
            .get_all_by_reference(CLASS_PATTERN)
            .into_iter()
            .map(|(tag, _)| tag)
            .collect::<Vec<_>>();
        tags.sort_by_key(|tag| tag.0);
        tags.par_iter()
            .filter_map(|tag| Skeleton::load(*tag))
            .collect()
    })
}

/// One vertex's dominant bone and object-space position, used to match a mesh to its skeleton.
#[derive(Debug, Clone, Copy)]
pub struct BoneSample {
    pub bone: u32,
    pub position: [f32; 3],
}

/// Dominant bone of a packed layout-7 vertex (`w` is the signed 16-bit POSITION.w).
/// Mirrors the skinning producer: large magnitudes address a palette record, small
/// non-negative values are a rigid bone index.
pub fn dominant_bone(vertex_index: usize, w: i16, palette: &[u8]) -> Option<u32> {
    let w = i32::from(w);
    if w.abs() <= 2047 {
        return u32::try_from(w).ok();
    }
    let record = (((w.abs() - 2048) * 8) as usize) | ((vertex_index << usize::from(w < 0)) & 7);
    let bytes = palette.get(record * 4..record * 4 + 4)?;
    Some(u32::from(if bytes[2] >= bytes[3] {
        bytes[0]
    } else {
        bytes[1]
    }))
}

/// The skeleton whose bind pose best explains which bone each sampled vertex follows.
pub fn match_skeleton(samples: &[BoneSample]) -> Option<&'static Skeleton> {
    match_skeleton_where(samples, |_| true)
}

/// `match_skeleton` among the skeletons `accept` allows.
pub fn match_skeleton_where(
    samples: &[BoneSample],
    accept: impl Fn(&Skeleton) -> bool,
) -> Option<&'static Skeleton> {
    let highest = samples.iter().map(|sample| sample.bone as usize).max()?;
    all_skeletons()
        .iter()
        .filter(|skeleton| skeleton.names.len() > highest && accept(skeleton))
        .map(|skeleton| {
            let error = samples
                .iter()
                .map(|sample| {
                    let bone = skeleton.bind[sample.bone as usize].translation;
                    (0..3)
                        .map(|c| (bone[c] - sample.position[c]).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .sum::<f32>()
                / samples.len() as f32;
            (skeleton, error)
        })
        .min_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then(a.0.names.len().cmp(&b.0.names.len()))
        })
        .map(|(skeleton, _)| skeleton)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotSample {
    pub rotation: Option<Quat>,
    pub translation: Option<[f32; 3]>,
    pub scale: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Clip {
    pub tag: TagHash,
    pub name_hash: u32,
    pub frames: usize,
    pub slots: usize,
    /// Type byte 0: a complete pose, as opposed to an overlay or replacement layer.
    pub base: bool,
    /// Checksum of the rig the clip was authored for.
    pub rig: u32,
    /// Slots hold the runner-style control rig (object-space limbs) rather than
    /// plain parent-local transforms in skeleton order.
    pub control_rig: bool,
    data: Vec<u8>,
}

impl Clip {
    pub fn load(tag: TagHash) -> Option<Self> {
        if package_manager().get_entry(tag)?.reference != CLASS_ANIMATION_CLIP {
            return None;
        }
        let data = package_manager().read_tag(tag).ok()?;
        Some(Self {
            tag,
            name_hash: u32_at(&data, 0x120)?,
            frames: usize::from(u16_at(&data, 0x140)?).max(1),
            slots: usize::from(u16_at(&data, 0x142)?),
            base: *data.get(0x181)? == 0,
            rig: array_at(&data, 0x150)
                .and_then(|(_, start)| u32_at(&data, start))
                .unwrap_or(0),
            control_rig: u32_at(&data, 0x11c)? != 0,
            data,
        })
    }

    /// Object channels the clip animates, as (channel, value) at `frame`.
    ///
    /// Besides bone tracks a clip carries tracks for object channels, the
    /// values materials read: the array at 0x90 names them, and a chunk of
    /// ranged per-frame samples holds one track for each name after those the
    /// keyframed chunk covers. The game writes them onto the entity while the
    /// clip plays (Sentinel's mask shows another icon mid-fidget this way).
    pub fn channel_values(&self, frame: f32) -> Vec<(u32, f32)> {
        let data = &self.data;
        let sampled = || {
            let (count, names) = array_at(data, 0x90)?;
            let pointer = u64::from_le_bytes(data.get(0x20..0x28)?.try_into().ok()?) as usize;
            let chunk = 0x20 + Some(pointer).filter(|pointer| *pointer != 0)?;
            (u32_at(data, chunk.checked_sub(4)?)? == CLASS_CHANNEL_SAMPLES).then_some(())?;
            let tracks = usize::from(u16_at(data, chunk + 2)?);
            let frames = (u32_at(data, chunk + 0xc)? as usize).max(1);
            let samples = u16_array(data, chunk + 0x20)?;
            let ranges = f32_array(data, chunk + 0x40)?;
            let minimums = f32_array(data, chunk + 0x50)?;
            let frame = (frame.max(0.0).round() as usize).min(frames - 1);
            (0..tracks)
                .map(|track| {
                    let name = u32_at(data, names + (count.checked_sub(tracks)? + track) * 4)?;
                    let sample = f32::from(*samples.get(track * frames + frame)?) / 65535.0;
                    Some((name, minimums.get(track)? + sample * ranges.get(track)?))
                })
                .collect::<Option<Vec<_>>>()
        };
        sampled().unwrap_or_default()
    }

    pub fn info(&self) -> ClipInfo {
        ClipInfo {
            tag: self.tag,
            name_hash: self.name_hash,
            frames: self.frames,
            slots: self.slots,
            rig: self.rig,
            base: self.base,
        }
    }

    pub fn duration_seconds(&self) -> f32 {
        (self.frames.saturating_sub(1)) as f32 / CLIP_FRAMES_PER_SECOND
    }

    /// Slot values at a fractional frame, blending the two neighbouring stored frames.
    pub fn sample(&self, frame: f32) -> Vec<SlotSample> {
        let frame = frame.clamp(0.0, (self.frames - 1) as f32);
        let (low, t) = (frame.floor() as usize, frame.fract());
        let mut samples = self.sample_frame(low);
        if t > 0.0 && low + 1 < self.frames {
            for (sample, next) in samples.iter_mut().zip(self.sample_frame(low + 1)) {
                if let (Some(a), Some(b)) = (sample.rotation, next.rotation) {
                    sample.rotation = Some(nlerp(a, b, t));
                }
                if let (Some(a), Some(b)) = (sample.translation, next.translation) {
                    sample.translation = Some(std::array::from_fn(|c| a[c] * (1.0 - t) + b[c] * t));
                }
                if let (Some(a), Some(b)) = (sample.scale, next.scale) {
                    sample.scale = Some(a * (1.0 - t) + b * t);
                }
            }
        }
        samples
    }

    fn sample_frame(&self, frame: usize) -> Vec<SlotSample> {
        let mut out = vec![SlotSample::default(); self.slots];
        let _ = self.decode(frame, &mut out);
        out
    }

    fn decode(&self, frame: usize, out: &mut [SlotSample]) -> Option<()> {
        let data = &self.data;
        // Static scale/rotation/translation then animated scale/rotation/translation.
        let maps = [0xA8, 0xB8, 0xC8, 0xD8, 0xE8, 0xF8]
            .map(|offset| u16_array(data, offset).unwrap_or_default());
        let pointer = |offset: usize| {
            let relative =
                u64::from_le_bytes(data.get(offset..offset + 8)?.try_into().ok()?) as usize;
            (relative != 0).then_some(offset + relative)
        };
        decode_raw(data, pointer(0x10)?, 0, &maps[1], &maps[2], out)?;
        decode_raw_scales(data, pointer(0x10)?, 0, &maps[0], out)?;
        if let Some(chunk) = pointer(0x18) {
            match u32_at(data, chunk - 4)? {
                CODEC_RAW => {
                    decode_raw(data, chunk, frame, &maps[4], &maps[5], out)?;
                    decode_raw_scales(data, chunk, frame, &maps[3], out)?
                }
                CODEC_RANGED => decode_ranged(data, chunk, frame, &maps[4], &maps[5], out)?,
                CODEC_CURVE => decode_curve(data, chunk, frame, &maps[4], &maps[5], out)?,
                _ => {}
            }
        }
        Some(())
    }

    /// How much of the clip's static pose agrees with `skeleton`'s bind pose: the
    /// share of statically posed slots whose translation equals the node's
    /// bind-local one. A plain-local clip keeps most bones at their bind offset,
    /// so a low share means the clip was authored for another rig of this size.
    /// `None` when the clip is not plain-local or poses nothing statically.
    pub fn rest_fit(&self, skeleton: &Skeleton) -> Option<f32> {
        if self.control_rig || self.slots != skeleton.names.len() {
            return None;
        }
        let data = &self.data;
        let relative = u64::from_le_bytes(data.get(0x10..0x18)?.try_into().ok()?) as usize;
        let mut samples = vec![SlotSample::default(); self.slots];
        decode_raw(
            data,
            0x10 + relative,
            0,
            &u16_array(data, 0xB8).unwrap_or_default(),
            &u16_array(data, 0xC8).unwrap_or_default(),
            &mut samples,
        )?;
        let posed = samples
            .iter()
            .enumerate()
            .filter_map(|(node, sample)| {
                Some((skeleton.bind_local(node).translation, sample.translation?))
            })
            .collect::<Vec<_>>();
        let agreeing = posed
            .iter()
            .filter(|(bind, clip)| (0..3).all(|c| (bind[c] - clip[c]).abs() < 0.002))
            .count();
        (!posed.is_empty()).then(|| agreeing as f32 / posed.len() as f32)
    }
}

fn slot<'a>(out: &'a mut [SlotSample], map: &[u16], track: usize) -> Option<&'a mut SlotSample> {
    out.get_mut(usize::from(*map.get(track)?))
}

/// `B13B` uniform scales: one value per track and frame over a shared range.
fn decode_raw_scales(
    data: &[u8],
    chunk: usize,
    frame: usize,
    scales: &[u16],
    out: &mut [SlotSample],
) -> Option<()> {
    let count = usize::from(u16_at(data, chunk + 2)?);
    if count == 0 {
        return Some(());
    }
    let frames = (u32_at(data, chunk + 0x10)? as usize).max(1);
    let frame = frame.min(frames - 1);
    let (range, minimum) = (f32_at(data, chunk + 0x14)?, f32_at(data, chunk + 0x18)?);
    let values = u16_array(data, chunk + 0x38)?;
    for track in 0..count {
        let raw = *values.get(track * frames + frame)?;
        if let Some(sample) = slot(out, scales, track) {
            sample.scale = Some(f32::from(raw) / 65535.0 * range + minimum);
        }
    }
    Some(())
}

/// `B13B`: every frame stored; rotations as offset-binary xyzw, one translation range per axis.
fn decode_raw(
    data: &[u8],
    chunk: usize,
    frame: usize,
    rotations: &[u16],
    translations: &[u16],
    out: &mut [SlotSample],
) -> Option<()> {
    let [scales, rotation_count, translation_count] =
        [2, 4, 6].map(|offset| u16_at(data, chunk + offset).map(usize::from));
    let (scales, rotation_count, translation_count) =
        (scales?, rotation_count?, translation_count?);
    let frames = (u32_at(data, chunk + 0x10)? as usize).max(1);
    let frame = frame.min(frames - 1);
    let range: [f32; 3] = [
        f32_at(data, chunk + 0x1c)?,
        f32_at(data, chunk + 0x20)?,
        f32_at(data, chunk + 0x24)?,
    ];
    let minimum: [f32; 3] = [
        f32_at(data, chunk + 0x28)?,
        f32_at(data, chunk + 0x2c)?,
        f32_at(data, chunk + 0x30)?,
    ];
    let values = u16_array(data, chunk + 0x38)?;
    // Samples are grouped per kind, then [track][frame][component].
    let rotation_base = frames * scales;
    let translation_base = rotation_base + frames * rotation_count * 4;
    for track in 0..rotation_count {
        let start = rotation_base + (track * frames + frame) * 4;
        let raw = values.get(start..start + 4)?;
        if let Some(sample) = slot(out, rotations, track) {
            sample.rotation = Some(normalize(std::array::from_fn(|c| {
                (f32::from(raw[c]) - 32767.0) / 32767.0
            })));
        }
    }
    for track in 0..translation_count {
        let start = translation_base + (track * frames + frame) * 3;
        let raw = values.get(start..start + 3)?;
        if let Some(sample) = slot(out, translations, track) {
            sample.translation = Some(std::array::from_fn(|c| {
                f32::from(raw[c]) / 65535.0 * range[c] + minimum[c]
            }));
        }
    }
    Some(())
}

/// `B13E`: every frame stored with one minimum/range per component.
fn decode_ranged(
    data: &[u8],
    chunk: usize,
    frame: usize,
    rotations: &[u16],
    translations: &[u16],
    out: &mut [SlotSample],
) -> Option<()> {
    let scales = usize::from(u16_at(data, chunk + 2)?);
    let rotation_count = usize::from(u16_at(data, chunk + 4)?);
    let translation_count = usize::from(u16_at(data, chunk + 6)?);
    let frames = (u32_at(data, chunk + 0x10)? as usize).max(1);
    let frame = frame.min(frames - 1);
    let values = u16_array(data, chunk + 0x18)?;
    let ranges = f32_array(data, chunk + 0x28)?;
    let minimums = f32_array(data, chunk + 0x38)?;
    let value = |first: usize, width: usize, c: usize| {
        Some(
            minimums.get(first + c)?
                + f32::from(*values.get(first * frames + frame * width + c)?) / 65535.0
                    * ranges.get(first + c)?,
        )
    };
    for track in 0..rotation_count {
        let first = scales + track * 4;
        let q = [
            value(first, 4, 0)?,
            value(first, 4, 1)?,
            value(first, 4, 2)?,
            value(first, 4, 3)?,
        ];
        if let Some(sample) = slot(out, rotations, track) {
            sample.rotation = Some(normalize(q));
        }
    }
    for track in 0..translation_count {
        let first = scales + rotation_count * 4 + track * 3;
        let t = [
            value(first, 3, 0)?,
            value(first, 3, 1)?,
            value(first, 3, 2)?,
        ];
        if let Some(sample) = slot(out, translations, track) {
            sample.translation = Some(t);
        }
    }
    Some(())
}

/// `B13F`: keyframed tracks. Keys are signed 16-bit; each segment lasts a stored number
/// of frames; rotation keys hold w, x, y. The per-segment tangent nibbles are not
/// applied: keys blend linearly.
fn decode_curve(
    data: &[u8],
    chunk: usize,
    frame: usize,
    rotations: &[u16],
    translations: &[u16],
    out: &mut [SlotSample],
) -> Option<()> {
    let scales = usize::from(u16_at(data, chunk + 2)?);
    let rotation_count = usize::from(u16_at(data, chunk + 4)?);
    let translation_count = usize::from(u16_at(data, chunk + 6)?);
    let keys = u16_array(data, chunk + 0x20)?;
    let (duration_count, duration_start) = array_at(data, chunk + 0x30)?;
    let durations = data.get(duration_start..duration_start + duration_count)?;
    let track_scales = f32_array(data, chunk + 0x50)?;
    let track_offsets = f32_array(data, chunk + 0x60)?;
    let starts = u16_array(data, chunk + 0x70)?;
    for track in scales..scales + rotation_count + translation_count {
        let Some(cursor) = starts
            .get(track)
            .map(|start| usize::from(*start).saturating_sub(1))
        else {
            break;
        };
        let (Some(first_segment), Some(segments)) = (keys.get(cursor), keys.get(cursor + 2)) else {
            continue;
        };
        let (first_segment, segments) = (usize::from(*first_segment), usize::from(*segments));
        let (mut segment, mut start) = (0usize, 0usize);
        while segment + 1 < segments {
            let duration = usize::from(*durations.get(first_segment + segment)?);
            if frame < start + duration {
                break;
            }
            start += duration;
            segment += 1;
        }
        let duration = f32::from((*durations.get(first_segment + segment).unwrap_or(&1)).max(1));
        let t = ((frame.saturating_sub(start)) as f32 / duration).clamp(0.0, 1.0);
        let key = |index: usize, c: usize| {
            keys.get(cursor + 3 + index * 3 + c)
                .map(|raw| f32::from(*raw as i16) / 32767.0)
        };
        let (Some(a), Some(b)) = (
            (|| Some([key(segment, 0)?, key(segment, 1)?, key(segment, 2)?]))(),
            (|| {
                Some([
                    key(segment + 1, 0)?,
                    key(segment + 1, 1)?,
                    key(segment + 1, 2)?,
                ])
            })(),
        ) else {
            continue;
        };
        if track < scales + rotation_count {
            // Keys store w, x, y; z is the non-negative remainder.
            let full = |v: [f32; 3]| {
                [
                    v[1],
                    v[2],
                    (1.0 - v.iter().map(|c| c * c).sum::<f32>()).max(0.0).sqrt(),
                    v[0],
                ]
            };
            if let Some(sample) = slot(out, rotations, track - scales) {
                sample.rotation = Some(nlerp(full(a), full(b), t));
            }
        } else {
            let index = track - rotation_count;
            let (Some(scale), Some(offset)) = (track_scales.get(index), track_offsets.get(index))
            else {
                continue;
            };
            if let Some(sample) = slot(out, translations, track - scales - rotation_count) {
                sample.translation = Some(std::array::from_fn(|c| {
                    offset + scale * (a[c] * (1.0 - t) + b[c] * t)
                }));
            }
        }
    }
    Some(())
}

/// Clip slots holding object-space rotations, by node name.
const RUNNER_OBJECT_SLOTS: [(usize, &str); 18] = [
    (2, "b_pelvis"),
    (3, "b_l_thigh"),
    (4, "b_l_calf"),
    (5, "b_l_foot"),
    (7, "b_r_thigh"),
    (8, "b_r_calf"),
    (9, "b_r_foot"),
    (11, "b_spine_1"),
    (12, "b_spine_3"),
    (13, "b_l_clav"),
    (14, "b_r_clav"),
    (15, "b_l_upperarm"),
    (16, "b_l_forearm"),
    (17, "b_l_hand"),
    (18, "b_r_upperarm"),
    (19, "b_r_forearm"),
    (20, "b_r_hand"),
    (22, "b_head"),
];
/// Clip slots holding a rotation relative to the node's bind-local rotation.
const RUNNER_LOCAL_SLOTS: [(usize, &str); 3] = [(6, "b_l_toe"), (10, "b_r_toe"), (21, "b_neck_1")];
const RUNNER_PELVIS_SLOT: usize = 2;
/// Finger joints start here: two hands of five fingers of three joints each.
const RUNNER_FINGER_SLOT: usize = 30;
const RUNNER_FINGERS: [&str; 5] = ["thumb", "index", "middle", "ring", "pinky"];

/// How a clip slot's rotation applies to its node.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SlotSpace {
    /// The node's object-space rotation.
    Object,
    /// A rotation on top of the node's bind-local rotation.
    Delta,
    /// The node's parent-local rotation and translation, as stored.
    Local,
    /// An object-space rotation and a position relative to this node (the
    /// clavicle of the same side): how the rig's grip targets are authored.
    Target(usize),
}

/// Name hash of the unnamed aim node every runner rig keeps in slot 1.
const RUNNER_AIM_NODE: u32 = 0xA91077F2;
/// Control-rig slots this viewer does not pose: they keep the bind pose.
const RUNNER_UNPOSED_SLOTS: [&str; 2] = ["b_pedestal", "b_utility"];
/// Grip slots: what a hand holds rides these rather than the hands.
/// Each is authored against the clavicle of its side, so a holstered item
/// stays put on the body whatever the arms do.
const RUNNER_TARGET_SLOTS: [(usize, &str, &str); 2] =
    [(23, "b_r_grip", "b_r_clav"), (24, "b_l_grip", "b_l_clav")];
/// The control rig, the four finger roots and the thirty finger joints.
const RUNNER_NAMED_SLOTS: usize = 60;

/// Maps a runner skeleton onto clip slots.
///
/// The first 26 slots are the control rig in authoring order, the next four
/// the roots of the ring and little fingers, then thirty finger joints. Every
/// node left over follows from slot 60 in skeleton order with its plain
/// parent-local transform: spine_2, neck_2, the twist and face bones, and
/// whatever a shell adds to the base rig, such as the bones of its gear.
#[derive(Debug, Clone)]
pub struct RunnerRig {
    pub skeleton: &'static Skeleton,
    /// Per skeleton node: the clip slot driving it and how to apply it.
    slots: Vec<Option<(usize, SlotSpace)>>,
    pelvis: Option<usize>,
}

impl RunnerRig {
    /// `new`, built once per skeleton and clip size: a playing clip asks for
    /// its mapping every frame.
    pub fn cached(skeleton: &'static Skeleton, clip_slots: usize) -> std::sync::Arc<Self> {
        type Rigs = rustc_hash::FxHashMap<(TagHash, usize), std::sync::Arc<RunnerRig>>;
        static RIGS: OnceLock<std::sync::Mutex<Rigs>> = OnceLock::new();
        let mut rigs = RIGS.get_or_init(Default::default).lock().unwrap();
        rigs.entry((skeleton.tag, clip_slots))
            .or_insert_with(|| std::sync::Arc::new(Self::new(skeleton, clip_slots)))
            .clone()
    }

    /// The mapping for clips of `clip_slots` slots played on `skeleton`.
    pub fn new(skeleton: &'static Skeleton, clip_slots: usize) -> Self {
        let mut slots = vec![None; skeleton.names.len()];
        // Nodes the first sixty slots account for, whether or not they are posed.
        let mut named = vec![false; skeleton.names.len()];
        for (slot, name) in RUNNER_OBJECT_SLOTS {
            if let Some(node) = skeleton.node(name) {
                slots[node] = Some((slot, SlotSpace::Object));
                named[node] = true;
            }
        }
        for (slot, name) in RUNNER_LOCAL_SLOTS {
            if let Some(node) = skeleton.node(name) {
                slots[node] = Some((slot, SlotSpace::Delta));
                named[node] = true;
            }
        }
        for (slot, name, origin) in RUNNER_TARGET_SLOTS {
            if let Some((node, origin)) = skeleton.node(name).zip(skeleton.node(origin)) {
                slots[node] = Some((slot, SlotSpace::Target(origin)));
                named[node] = true;
            }
        }
        for name in RUNNER_UNPOSED_SLOTS {
            if let Some(node) = skeleton.node(name) {
                named[node] = true;
            }
        }
        if let Some(node) = skeleton.names.iter().position(|name| *name == RUNNER_AIM_NODE) {
            named[node] = true;
        }
        for (hand, side) in ["l", "r"].into_iter().enumerate() {
            let hand_node = skeleton.node(&format!("b_{side}_hand"));
            for (finger, name) in RUNNER_FINGERS.into_iter().enumerate() {
                for joint in 0..3 {
                    let Some(node) = skeleton.node(&format!("b_{side}_{name}_{}", joint + 1)) else {
                        continue;
                    };
                    slots[node] = Some((
                        RUNNER_FINGER_SLOT + hand * 15 + finger * 3 + joint,
                        SlotSpace::Delta,
                    ));
                    named[node] = true;
                    // A finger hanging off the hand through one more node has a
                    // root of its own among the four slots before the fingers.
                    if joint == 0 {
                        let parent = usize::try_from(skeleton.parents[node]).ok();
                        if let Some(root) = parent.filter(|parent| Some(*parent) != hand_node) {
                            named[root] = true;
                        }
                    }
                }
            }
        }
        // The leftover nodes fill the upper slots in the skeleton order of the
        // rig the clip was authored for: this one when the clip has a slot per
        // node, else the runner rig of the clip's size, matched here by name.
        // Only a complete base rig leaves them in the order the slots use.
        if named.iter().filter(|named| **named).count() == RUNNER_NAMED_SLOTS {
            let rest = (0..skeleton.names.len()).filter(|node| !named[*node]).collect::<Vec<_>>();
            if clip_slots == skeleton.names.len() {
                for (index, node) in rest.into_iter().enumerate() {
                    slots[node] = Some((RUNNER_NAMED_SLOTS + index, SlotSpace::Local));
                }
            } else if let Some(authored) = all_skeletons()
                .iter()
                .find(|other| other.is_runner() && other.names.len() == clip_slots)
            {
                let authored_rig = Self::new(authored, clip_slots);
                for node in rest {
                    let slot = authored
                        .names
                        .iter()
                        .position(|name| *name == skeleton.names[node])
                        .and_then(|node| authored_rig.slots[node]);
                    slots[node] = slot;
                }
            }
        }
        Self {
            skeleton,
            slots,
            pelvis: skeleton.node("b_pelvis"),
        }
    }

    /// One sampled clip frame as object-space transforms keyed by node name hash.
    pub fn named_pose(&self, samples: &[SlotSample]) -> rustc_hash::FxHashMap<u32, Transform> {
        self.skeleton
            .names
            .iter()
            .copied()
            .zip(self.pose(samples))
            .collect()
    }

    /// Object-space node transforms for one sampled clip frame. Nodes the clip
    /// does not drive keep their bind-local transform under the posed parent.
    pub fn pose(&self, samples: &[SlotSample]) -> Vec<Transform> {
        let skeleton = self.skeleton;
        let mut pose: Vec<Transform> = Vec::with_capacity(skeleton.names.len());
        for node in 0..skeleton.names.len() {
            let Some(parent) = usize::try_from(skeleton.parents[node])
                .ok()
                .and_then(|p| pose.get(p).copied())
            else {
                pose.push(skeleton.bind[node]);
                continue;
            };
            let mut local = skeleton.bind_local(node);
            let sample = self.slots[node].and_then(|(slot, space)| Some((samples.get(slot)?, space)));
            let rotation = match sample.and_then(|(sample, space)| Some((sample.rotation?, space))) {
                Some((rotation, SlotSpace::Object | SlotSpace::Target(_))) => rotation,
                Some((delta, SlotSpace::Delta)) => {
                    normalize(quat_mul(parent.rotation, quat_mul(local.rotation, delta)))
                }
                Some((rotation, SlotSpace::Local)) => normalize(quat_mul(parent.rotation, rotation)),
                None => normalize(quat_mul(parent.rotation, local.rotation)),
            };
            if let Some((sample, SlotSpace::Local)) = sample {
                local.translation = sample.translation.unwrap_or(local.translation);
            }
            let rotated = quat_rotate(parent.rotation, local.translation);
            let mut translation: [f32; 3] =
                std::array::from_fn(|c| parent.translation[c] + rotated[c]);
            if let Some((sample, SlotSpace::Target(origin))) = sample {
                if let (Some(offset), Some(origin)) = (sample.translation, pose.get(origin)) {
                    translation = std::array::from_fn(|c| origin.translation[c] + offset[c]);
                }
            }
            if Some(node) == self.pelvis {
                // The pelvis slot stores its offset from the bind position.
                if let Some(offset) = samples
                    .get(RUNNER_PELVIS_SLOT)
                    .and_then(|sample| sample.translation)
                {
                    translation =
                        std::array::from_fn(|c| skeleton.bind[node].translation[c] + offset[c]);
                }
            }
            pose.push(Transform {
                rotation,
                translation,
                scale: 1.0,
            });
        }
        pose
    }
}

/// How far, in units of a source's packed position range, `skinning` carries
/// its vertices from where they were authored.
pub fn packed_space_reach(skinning: &[Transform], scale: f32, offset: [f32; 3]) -> f32 {
    skinning
        .iter()
        .map(|transform| {
            let rotated = quat_rotate(transform.rotation, offset).map(|c| c * transform.scale);
            (0..3)
                .map(|c| ((rotated[c] + transform.translation[c] - offset[c]) / scale).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .fold(0.0, f32::max)
}

/// Dual quaternion rows (real, dual) the skinning producer blends, for a mesh whose
/// packed positions decode as `object = packed * scale + offset`.
pub fn packed_space_dual_quaternions(
    skinning: &[Transform],
    scale: f32,
    offset: [f32; 3],
) -> Vec<[f32; 4]> {
    skinning
        .iter()
        .flat_map(|transform| {
            let rotated = quat_rotate(transform.rotation, offset);
            let t: [f32; 3] = std::array::from_fn(|c| {
                (rotated[c] + transform.translation[c] - offset[c]) / scale
            });
            let dual = quat_mul(
                [t[0] * 0.5, t[1] * 0.5, t[2] * 0.5, 0.0],
                transform.rotation,
            );
            [transform.rotation, dual]
        })
        .collect()
}

/// Whether any bone of `skinning` resizes what it carries; dual quaternions
/// cannot express that, so such a mesh is blended with matrices instead.
pub fn skinning_scales(skinning: &[Transform]) -> bool {
    skinning.iter().any(|transform| (transform.scale - 1.0).abs() > 1e-4)
}

/// Matrix rows (three per bone, each `[x axis, y axis, z axis, translation]`
/// components of one output coordinate) the skinning producer blends linearly,
/// for a mesh whose packed positions decode as `object = packed * scale + offset`.
pub fn packed_space_matrices(skinning: &[Transform], scale: f32, offset: [f32; 3]) -> Vec<[f32; 4]> {
    skinning
        .iter()
        .flat_map(|transform| {
            let axes: [[f32; 3]; 3] = std::array::from_fn(|axis| {
                let mut unit = [0.0; 3];
                unit[axis] = transform.scale;
                quat_rotate(transform.rotation, unit)
            });
            let moved = quat_rotate(transform.rotation, offset).map(|c| c * transform.scale);
            let rows: [[f32; 4]; 3] = std::array::from_fn(|c| {
                [
                    axes[0][c],
                    axes[1][c],
                    axes[2][c],
                    (moved[c] + transform.translation[c] - offset[c]) / scale,
                ]
            });
            rows
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct ClipInfo {
    pub tag: TagHash,
    pub name_hash: u32,
    pub frames: usize,
    pub slots: usize,
    pub rig: u32,
    /// A complete pose, playable on its own.
    pub base: bool,
}

/// Every full-body clip authored for the shared runner rig, sorted by tag.
pub fn runner_clips() -> &'static [ClipInfo] {
    static CLIPS: OnceLock<Vec<ClipInfo>> = OnceLock::new();
    CLIPS.get_or_init(|| {
        let mut clips = package_manager()
            .get_all_by_reference(CLASS_ANIMATION_CLIP)
            .into_iter()
            .filter_map(|(tag, _)| {
                let clip = Clip::load(tag)?;
                // Overlay and replacement clips only make sense layered on a base pose.
                (clip.slots == RUNNER_CLIP_NODES && clip.base).then(|| clip.info())
            })
            .collect::<Vec<_>>();
        clips.sort_by_key(|clip| clip.tag.0);
        clips
    })
}

/// One model's pose: per geometry source, the per-bone transforms carrying its
/// bind-pose object space to posed object space. `None` leaves a source unposed.
#[derive(Debug, Clone, Default)]
pub struct ModelPose {
    pub sources: Vec<Option<Vec<Transform>>>,
    /// Draws left out of this frame, by index: a prop not spawned yet, or the
    /// holstered copy of one being held.
    pub hidden_draws: Vec<usize>,
}

impl ModelPose {
    /// Pose every source skeleton from one clip frame; a negative frame yields
    /// the bind pose.
    ///
    /// The clip drives the skeleton it was authored for: the runner body for a
    /// control-rig clip, else the source skeleton with one node per clip slot.
    /// Every other skeleton of the model follows it by node name. `None` when no
    /// source skeleton can play the clip.
    pub fn from_clip(
        skeletons: &[Option<&'static Skeleton>],
        clip: &Clip,
        frame: f32,
    ) -> Option<Self> {
        let (driven, pose) = clip.driven_pose(skeletons, frame)?;
        let moved = driven
            .names
            .iter()
            .copied()
            .zip(driven.skinning_transforms(&pose))
            .collect();
        let named = driven.names.iter().copied().zip(pose).collect();
        // Props the clip spawns ride a socket of the driven skeleton and play
        // their own paired clip.
        let props = if frame < 0.0 { Default::default() } else { clip.props() };
        Some(Self {
            sources: skeletons
                .iter()
                .map(|skeleton| {
                    let skeleton = (*skeleton)?;
                    match props.iter().find(|prop| prop.skeleton.tag == skeleton.tag) {
                        Some(prop) => Some(skeleton.skinning_transforms(&prop.pose(driven, &named, frame)?)),
                        None => Some(skeleton.skinning_transforms(&skeleton.retarget(&moved))),
                    }
                })
                .collect(),
            hidden_draws: vec![],
        })
    }
}

fn fnv1_extend(state: u32, bytes: &[u8]) -> u32 {
    bytes.iter().fold(state, |hash, byte| {
        hash.wrapping_mul(quicktag_core::util::FNV1_PRIME) ^ u32::from(*byte)
    })
}

/// How many of the wordlist's most frequent tokens are tried as a mode or weapon.
const NAME_TOKEN_LIMIT: usize = 20_000;

/// Names of the runner clips, keyed by the FNV-1 hash each clip stores.
///
/// Direct hits come from quicktag's wordlist. Clip names follow
/// `<mode>_<weapon>_<action>`, so the rest are searched in that shape using only
/// the wordlist's own vocabulary: its frequent tokens as modes and weapons, its
/// short identifiers as actions. Only one of the three parts is ever unknown at
/// a time, which keeps chance hash collisions out.
static CLIP_NAMES: OnceLock<rustc_hash::FxHashMap<u32, String>> = OnceLock::new();

/// `runner_clip_names` without the wait: starts the search on a worker thread the
/// first time and returns `None` until it has finished.
pub fn runner_clip_names_if_ready() -> Option<&'static rustc_hash::FxHashMap<u32, String>> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        std::thread::spawn(runner_clip_names);
    });
    CLIP_NAMES.get()
}

pub fn runner_clip_names() -> &'static rustc_hash::FxHashMap<u32, String> {
    CLIP_NAMES.get_or_init(|| {
        use rustc_hash::{FxHashMap, FxHashSet};
        let wanted = runner_clips()
            .iter()
            .map(|clip| clip.name_hash)
            .collect::<FxHashSet<_>>();
        let mut names = FxHashMap::<u32, String>::default();
        let mut token_counts = FxHashMap::<String, u32>::default();
        let mut phrases = FxHashSet::<String>::default();
        quicktag_strings::wordlist::load_wordlist(|word, hash| {
            if wanted.contains(&hash) {
                names.entry(hash).or_insert_with(|| word.to_owned());
            }
            let identifier = word.len() <= 24
                && word
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
            if identifier && word.split('_').all(|token| !token.is_empty()) {
                let tokens = word.split('_').count();
                if tokens <= 3 && word.split('_').all(|token| token.len() <= 10) {
                    phrases.insert(word.to_owned());
                }
                for token in word
                    .split('_')
                    .filter(|token| (2..=10).contains(&token.len()))
                {
                    *token_counts.entry(token.to_owned()).or_default() += 1;
                }
            }
        });
        let mut tokens = token_counts.into_iter().collect::<Vec<_>>();
        tokens.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let tokens = tokens
            .into_iter()
            .take(NAME_TOKEN_LIMIT)
            .map(|(token, _)| token)
            .collect::<Vec<_>>();

        // Seeds: modes, weapons and actions of the directly named clips. A name
        // sharing neither its mode nor its weapon with another direct hit is most
        // likely a chance collision and must not seed the search.
        let direct = names
            .values()
            .filter_map(|name| {
                let parts = name.split('_').collect::<Vec<_>>();
                (parts.len() >= 3
                    && parts
                        .iter()
                        .all(|part| part.bytes().all(|b| b.is_ascii_alphanumeric())))
                .then(|| {
                    (
                        parts[0].to_owned(),
                        parts[1].to_owned(),
                        parts[2..].join("_"),
                    )
                })
            })
            .collect::<Vec<_>>();
        let (mut modes, mut weapons, mut actions) = (
            FxHashSet::default(),
            FxHashSet::default(),
            FxHashSet::default(),
        );
        for (mode, weapon, action) in &direct {
            let shared = direct
                .iter()
                .filter(|other| &other.0 == mode || &other.1 == weapon)
                .count()
                > 1;
            if shared {
                modes.insert(mode.clone());
                weapons.insert(weapon.clone());
                actions.insert(action.clone());
            }
        }
        let prefix_state = |mode: &str, weapon: &str| {
            fnv1_extend(
                quicktag_core::util::FNV1_BASE,
                format!("{mode}_{weapon}_").as_bytes(),
            )
        };
        let phrases = phrases.into_iter().collect::<Vec<_>>();
        let mut scanned_pairs = FxHashSet::<(String, String)>::default();
        // Grow one part at a time: with the other two parts already trusted the
        // candidate space stays small enough that a single hit is not chance.
        for _ in 0..8 {
            let known = (names.len(), modes.len(), weapons.len(), actions.len());
            let mut found = Vec::<(String, String, String, u32)>::new();
            for token in &tokens {
                for weapon in &weapons {
                    let state = prefix_state(token, weapon);
                    for action in &actions {
                        let hash = fnv1_extend(state, action.as_bytes());
                        if wanted.contains(&hash) {
                            found.push((token.clone(), weapon.clone(), action.clone(), hash));
                        }
                    }
                }
                for mode in &modes {
                    let state = prefix_state(mode, token);
                    for action in &actions {
                        let hash = fnv1_extend(state, action.as_bytes());
                        if wanted.contains(&hash) {
                            found.push((mode.clone(), token.clone(), action.clone(), hash));
                        }
                    }
                }
            }
            // Every wordlist phrase as an action, once per trusted pair.
            let pairs = modes
                .iter()
                .flat_map(|mode| {
                    weapons
                        .iter()
                        .map(move |weapon| (mode.clone(), weapon.clone()))
                })
                .filter(|pair| scanned_pairs.insert(pair.clone()))
                .collect::<Vec<_>>();
            let (wanted_ref, phrases_ref) = (&wanted, &phrases);
            found.par_extend(pairs.par_iter().flat_map_iter(|(mode, weapon)| {
                let state = prefix_state(mode, weapon);
                phrases_ref.iter().filter_map(move |action| {
                    let hash = fnv1_extend(state, action.as_bytes());
                    wanted_ref
                        .contains(&hash)
                        .then(|| (mode.clone(), weapon.clone(), action.clone(), hash))
                })
            }));
            for (mode, weapon, action, hash) in found {
                names
                    .entry(hash)
                    .or_insert_with(|| format!("{mode}_{weapon}_{action}"));
                modes.insert(mode);
                weapons.insert(weapon);
                actions.insert(action);
            }
            if known == (names.len(), modes.len(), weapons.len(), actions.len()) {
                break;
            }
        }
        // Weapon-independent clips drop the middle part: `<mode>_<action>`.
        for mode in &modes {
            let state = fnv1_extend(
                quicktag_core::util::FNV1_BASE,
                format!("{mode}_").as_bytes(),
            );
            for action in &actions {
                let hash = fnv1_extend(state, action.as_bytes());
                if wanted.contains(&hash) {
                    names
                        .entry(hash)
                        .or_insert_with(|| format!("{mode}_{action}"));
                }
            }
        }
        // A direct hit whose mode nothing else shares is a chance collision.
        names.retain(|_, name| {
            name.split('_')
                .next()
                .is_some_and(|mode| modes.contains(mode))
        });
        log::info!(
            "Named {} of {} runner clips ({} modes, {} weapons, {} actions)",
            names.len(),
            wanted.len(),
            modes.len(),
            weapons.len(),
            actions.len()
        );
        names
    })
}

const CLASS_ANIMATION_SET: u32 = 0x80803BE9;
/// Entry names every runner has, as `<codename>_<action>`.
const RUNNER_ENTRY_ACTIONS: [&str; 3] = ["lobby_idle", "loadout", "lobby_idle_fidget"];
/// Runner rigs start at the shared 81 slots; each shell appends its own nodes.
const RUNNER_MIN_SLOTS: usize = 81;

/// Runner codenames quicktag's wordlist records under `heroes.<codename>.`.
fn runner_codenames() -> Vec<String> {
    let mut codenames = rustc_hash::FxHashSet::default();
    quicktag_strings::wordlist::load_wordlist(|word, _| {
        if let Some(codename) = word
            .strip_prefix("heroes.")
            .and_then(|rest| rest.split('.').next())
        {
            codenames.insert(codename.to_owned());
        }
    });
    let mut codenames = codenames.into_iter().collect::<Vec<_>>();
    codenames.sort();
    codenames
}

struct AnimationSetEntry {
    /// Entry name and the per-entry hash of the set's companion table.
    hashes: [u32; 2],
    clip: TagHash,
}

fn animation_set_entries(
    tag: TagHash,
    clips_by_hash64: &rustc_hash::FxHashMap<u64, TagHash>,
) -> Vec<AnimationSetEntry> {
    let Ok(data) = package_manager().read_tag(tag) else {
        return vec![];
    };
    let companion = u32_at(&data, 0x78)
        .and_then(|tag| package_manager().read_tag(TagHash(tag)).ok())
        .unwrap_or_default();
    let companion_start = array_at(&companion, 0x8).map_or(0, |(_, start)| start);
    let Some((count, start)) = array_at(&data, 0x58) else {
        return vec![];
    };
    (0..count)
        .filter_map(|index| {
            let entry = data.get(start + index * 0x30..start + (index + 1) * 0x30)?;
            let hash64 = u64::from_le_bytes(entry[0x20..0x28].try_into().ok()?);
            Some(AnimationSetEntry {
                hashes: [
                    u32_at(entry, 0x10)?,
                    u32_at(&companion, companion_start + index * 4).unwrap_or(0),
                ],
                clip: *clips_by_hash64.get(&hash64)?,
            })
        })
        .collect()
}

/// Body clips of one runner rig.
#[derive(Debug, Clone)]
pub struct RunnerClipGroup {
    /// Runner codename (`thief`, `stealth`, ...) when the packages name the rig's owner.
    pub codename: Option<String>,
    /// Distinct slot counts of the group's clips. A runner's rig has one or two;
    /// a group spanning many is a family of rigs rather than one runner's.
    pub slot_counts: Vec<usize>,
    pub clips: Vec<ClipInfo>,
}

fn in_runner_packages(tag: TagHash) -> bool {
    package_manager()
        .package_paths
        .get(&tag.pkg_id())
        .is_some_and(|path| {
            matches!(
                path.name.as_str(),
                "sr_sandbox" | "sr_sandbox_anims" | "sr_gear" | "sr_ui"
            )
        })
}

/// Who a rig, an animation set or a name belongs to while groups are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RigOwner {
    Runner(&'static str),
    /// A runner the packages do not name, known only by its main rig.
    Rig(u32),
}

fn most_voted(votes: rustc_hash::FxHashMap<RigOwner, usize>) -> Option<RigOwner> {
    votes
        .into_iter()
        .max_by_key(|(owner, count)| (*count, std::cmp::Reverse(format!("{owner:?}"))))
        .map(|(owner, _)| owner)
}

/// Runner body clips grouped by the rig they were authored for.
///
/// Every clip records a checksum of its rig and every runner shell has rigs of
/// its own, so the checksum is what separates one runner's clips from another's.
/// Rigs are attributed in three steps, each read from the animation sets:
///
/// 1. The shared body sets file `<codename>_lobby_idle` and its siblings under
///    each runner's clips, which names the rig those clips use. A rig several
///    runners' names lead to is shared, like the base rig.
/// 2. Each runner also has sets of its own (its face rig, its props) holding no
///    body clip. Their entry names are the ones the body sets file that
///    runner's clips under, so such a set belongs to whichever runner most of
///    its names lead to, and carries that runner's remaining rigs with it.
/// 3. A rig still unclaimed goes to the one runner whose rigs have its size,
///    when exactly one does.
///
/// A clip on a shared rig is filed under a runner only when a name held by that
/// runner's sets alone points at it. What is left on a shared rig other than
/// the base one stays together as an unnamed group: the clips of runners that
/// have no rig of their own.
pub fn runner_clip_groups() -> &'static [RunnerClipGroup] {
    static GROUPS: OnceLock<Vec<RunnerClipGroup>> = OnceLock::new();
    GROUPS.get_or_init(|| {
        use rustc_hash::{FxHashMap, FxHashSet};
        let pm = package_manager();
        // Playable body clips, and the tags of every body-rig clip of any type.
        let mut clips = FxHashMap::<TagHash, ClipInfo>::default();
        let mut body_rig_tags = FxHashSet::<TagHash>::default();
        for (tag, _) in pm.get_all_by_reference(CLASS_ANIMATION_CLIP) {
            if !in_runner_packages(tag) {
                continue;
            }
            let Some(clip) = Clip::load(tag).filter(|clip| clip.slots >= RUNNER_MIN_SLOTS) else {
                continue;
            };
            body_rig_tags.insert(tag);
            if clip.base {
                clips.insert(tag, clip.info());
            }
        }
        // The rig every runner shares is the commonest among clips of the base size.
        let mut base_rig_sizes = FxHashMap::<u32, usize>::default();
        for clip in clips.values().filter(|clip| clip.slots == RUNNER_MIN_SLOTS) {
            *base_rig_sizes.entry(clip.rig).or_default() += 1;
        }
        let base_rig = base_rig_sizes
            .iter()
            .max_by_key(|(_, count)| **count)
            .map_or(0, |(rig, _)| *rig);

        let clips_by_hash64 = pm
            .lookup
            .tag32_to_tag64
            .iter()
            .map(|(tag32, tag64)| (tag64.0, *tag32))
            .collect::<FxHashMap<_, _>>();
        let mut sets = pm
            .get_all_by_reference(CLASS_ANIMATION_SET)
            .into_iter()
            .map(|(tag, _)| tag)
            .filter(|tag| in_runner_packages(*tag))
            .collect::<Vec<_>>();
        sets.sort_by_key(|tag| tag.0);
        // Body clips by every hash a set files them under, and the sets holding none.
        let mut body_by_hash = FxHashMap::<u32, Vec<ClipInfo>>::default();
        let mut owner_sets = Vec::<FxHashSet<u32>>::new();
        for set in sets {
            let entries = animation_set_entries(set, &clips_by_hash64);
            let names = |entry: &AnimationSetEntry| {
                entry
                    .hashes
                    .into_iter()
                    .filter(|hash| *hash != quicktag_core::util::FNV1_BASE)
            };
            for entry in &entries {
                if let Some(clip) = clips.get(&entry.clip) {
                    for hash in names(entry) {
                        body_by_hash.entry(hash).or_default().push(*clip);
                    }
                }
            }
            if !entries.is_empty()
                && entries
                    .iter()
                    .all(|entry| !body_rig_tags.contains(&entry.clip))
            {
                owner_sets.push(entries.iter().flat_map(names).collect());
            }
        }
        let filed_under = |hash: &u32| body_by_hash.get(hash).into_iter().flatten();

        // 1. Rigs a runner's own entry names lead to.
        let codenames: &'static [String] = Box::leak(runner_codenames().into_boxed_slice());
        let own_names = |codename: &str| {
            RUNNER_ENTRY_ACTIONS.map(|action| fnv1(format!("{codename}_{action}").as_bytes()))
        };
        let mut direct = FxHashMap::<u32, FxHashMap<RigOwner, usize>>::default();
        for codename in codenames {
            for hash in own_names(codename) {
                for clip in filed_under(&hash).filter(|clip| clip.rig != base_rig) {
                    *direct
                        .entry(clip.rig)
                        .or_default()
                        .entry(RigOwner::Runner(codename))
                        .or_default() += 1;
                }
            }
        }
        let mut shared_rigs = FxHashSet::from_iter([base_rig]);
        let mut rig_owner = FxHashMap::<u32, RigOwner>::default();
        for (rig, votes) in direct {
            if votes.len() > 1 {
                shared_rigs.insert(rig);
            } else if let Some(owner) = most_voted(votes) {
                rig_owner.insert(rig, owner);
            }
        }
        let own_rig_clips =
            |hash: &u32| filed_under(hash).filter(|clip| !shared_rigs.contains(&clip.rig));

        // 2. Each owner set follows the named runner most of its names lead to, or
        // failing any, the unclaimed rig most of them lead to. A set whose names
        // mostly lead to shared-rig clips is generic (weapons, first person) and
        // owns nothing.
        let set_owners = owner_sets
            .iter()
            .map(|hashes| {
                let (mut runners, mut rigs) = (FxHashMap::default(), FxHashMap::default());
                for clip in hashes.iter().flat_map(&own_rig_clips) {
                    match rig_owner.get(&clip.rig) {
                        Some(owner) => *runners.entry(*owner).or_default() += 1,
                        None => *rigs.entry(RigOwner::Rig(clip.rig)).or_default() += 1,
                    }
                }
                let own = runners.values().chain(rigs.values()).sum::<usize>();
                let shared = hashes
                    .iter()
                    .filter(|hash| filed_under(hash).any(|clip| shared_rigs.contains(&clip.rig)))
                    .count();
                (own > shared)
                    .then(|| most_voted(runners).or_else(|| most_voted(rigs)))
                    .flatten()
            })
            .collect::<Vec<_>>();
        let mut carried = FxHashMap::<u32, FxHashMap<RigOwner, usize>>::default();
        for (hashes, owner) in owner_sets.iter().zip(&set_owners) {
            let Some(owner) = owner else { continue };
            for clip in hashes.iter().flat_map(&own_rig_clips) {
                if !rig_owner.contains_key(&clip.rig) {
                    *carried
                        .entry(clip.rig)
                        .or_default()
                        .entry(*owner)
                        .or_default() += 1;
                }
            }
        }
        for (rig, votes) in carried {
            if let Some(owner) = most_voted(votes) {
                rig_owner.insert(rig, owner);
            }
        }

        // 3. Unclaimed rigs by size, when one named runner alone has rigs of that size.
        let mut rig_slots = FxHashMap::<u32, FxHashSet<usize>>::default();
        for clip in clips.values() {
            rig_slots.entry(clip.rig).or_default().insert(clip.slots);
        }
        let mut runners_of_size = FxHashMap::<usize, FxHashSet<RigOwner>>::default();
        for (rig, owner) in rig_owner
            .iter()
            .filter(|(_, owner)| matches!(owner, RigOwner::Runner(_)))
        {
            for slots in &rig_slots[rig] {
                runners_of_size.entry(*slots).or_default().insert(*owner);
            }
        }
        for (rig, sizes) in &rig_slots {
            // A checksum spanning several sizes is a family of rigs, not a runner's.
            let [slots] = sizes.iter().copied().collect::<Vec<_>>()[..] else {
                continue;
            };
            if slots == RUNNER_MIN_SLOTS || shared_rigs.contains(rig) || rig_owner.contains_key(rig)
            {
                continue;
            }
            if let Some(owners) = runners_of_size
                .get(&slots)
                .filter(|owners| owners.len() == 1)
            {
                rig_owner.insert(*rig, *owners.iter().next().expect("one owner"));
            }
        }

        // Names held by one owner's sets alone file shared-rig clips under it; a
        // runner's own entry names always do.
        let mut name_owner = FxHashMap::<u32, RigOwner>::default();
        let mut contested = FxHashSet::<u32>::default();
        for (hashes, owner) in owner_sets.iter().zip(&set_owners) {
            let Some(owner) = owner else { continue };
            for hash in hashes {
                if name_owner
                    .insert(*hash, *owner)
                    .is_some_and(|other| other != *owner)
                {
                    contested.insert(*hash);
                }
            }
        }
        name_owner.retain(|hash, _| !contested.contains(hash));
        for codename in codenames {
            for hash in own_names(codename) {
                name_owner.insert(hash, RigOwner::Runner(codename));
            }
        }

        let mut grouped = FxHashMap::<RigOwner, Vec<ClipInfo>>::default();
        let mut filed = FxHashSet::<TagHash>::default();
        for (hash, owner) in &name_owner {
            for clip in filed_under(hash).filter(|clip| shared_rigs.contains(&clip.rig)) {
                grouped.entry(*owner).or_default().push(*clip);
                filed.insert(clip.tag);
            }
        }
        for clip in clips.values() {
            if !shared_rigs.contains(&clip.rig) {
                let owner = rig_owner
                    .get(&clip.rig)
                    .copied()
                    .unwrap_or(RigOwner::Rig(clip.rig));
                grouped.entry(owner).or_default().push(*clip);
            } else if clip.rig != base_rig && !filed.contains(&clip.tag) {
                grouped
                    .entry(RigOwner::Rig(clip.rig))
                    .or_default()
                    .push(*clip);
            }
        }
        let mut groups = grouped
            .into_iter()
            .map(|(owner, mut clips)| {
                clips.sort_by_key(|clip| clip.tag.0);
                clips.dedup_by_key(|clip| clip.tag);
                let mut slot_counts = clips.iter().map(|clip| clip.slots).collect::<Vec<_>>();
                slot_counts.sort_unstable();
                slot_counts.dedup();
                let codename = match owner {
                    RigOwner::Runner(codename) => Some(codename.to_owned()),
                    RigOwner::Rig(_) => None,
                };
                RunnerClipGroup {
                    codename,
                    slot_counts,
                    clips,
                }
            })
            .collect::<Vec<_>>();
        // Named runners alphabetically, then unnamed rigs from the largest down.
        groups.sort_by(|a, b| {
            (
                a.codename.is_none(),
                &a.codename,
                std::cmp::Reverse(a.clips.len()),
                a.clips[0].tag.0,
            )
                .cmp(&(
                    b.codename.is_none(),
                    &b.codename,
                    std::cmp::Reverse(b.clips.len()),
                    b.clips[0].tag.0,
                ))
        });
        groups
    })
}

const CLASS_ENTITY_DEFINITION: u32 = 0x8080BAAD;
/// Clips sampled per rig when checking that it rests in a skeleton's bind pose.
const REST_FIT_SAMPLES: usize = 8;
/// Least share of statically posed bones that must sit at their bind offset.
const REST_FIT_SHARE: f32 = 0.75;

/// Clips the packages tie to one model's skeleton.
#[derive(Debug, Clone, Default)]
pub struct ModelClips {
    /// Clips of the animation sets the skeleton's entity definitions play.
    pub own: Vec<ClipInfo>,
    /// Further clips authored for the same rig, held by other sets.
    pub same_rig: Vec<ClipInfo>,
}

/// How skeletons, animation sets and clips are tied together.
///
/// An entity definition (`8080BAAD`) gathers the patterns of one entity: one
/// carries its skeleton, another references its animation set. A clip belongs
/// to a skeleton when a definition holding that skeleton also holds a set
/// listing the clip, and the clip has one slot per skeleton node. The rig
/// checksum of those clips then identifies the rig in every other set too.
pub struct RigIndex {
    /// Animation sets reachable from each skeleton pattern.
    sets_of_skeleton: rustc_hash::FxHashMap<TagHash, Vec<TagHash>>,
    set_clips: rustc_hash::FxHashMap<TagHash, Vec<ClipInfo>>,
    clips_of_rig: rustc_hash::FxHashMap<u32, Vec<ClipInfo>>,
}

fn embedded_words(data: &[u8]) -> impl Iterator<Item = u32> + '_ {
    data.chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
}

impl RigIndex {
    fn build() -> Self {
        use rustc_hash::{FxHashMap, FxHashSet};
        let pm = package_manager();
        let clips = pm
            .get_all_by_reference(CLASS_ANIMATION_CLIP)
            .par_iter()
            .filter_map(|(tag, _)| Some((*tag, Clip::load(*tag)?.info())))
            .collect::<FxHashMap<_, _>>();
        let clips_by_hash64 = pm
            .lookup
            .tag32_to_tag64
            .iter()
            .map(|(tag32, tag64)| (tag64.0, *tag32))
            .collect::<FxHashMap<_, _>>();
        let set_clips = pm
            .get_all_by_reference(CLASS_ANIMATION_SET)
            .into_iter()
            .map(|(set, _)| {
                let mut listed = animation_set_entries(set, &clips_by_hash64)
                    .into_iter()
                    .filter_map(|entry| clips.get(&entry.clip).copied())
                    .collect::<Vec<_>>();
                listed.sort_by_key(|clip| clip.tag.0);
                listed.dedup_by_key(|clip| clip.tag);
                (set, listed)
            })
            .collect::<FxHashMap<_, _>>();

        let skeletons = all_skeletons()
            .iter()
            .map(|skeleton| skeleton.tag)
            .collect::<FxHashSet<_>>();
        // The set each pattern references, if any.
        let set_of_pattern = pm
            .get_all_by_reference(CLASS_PATTERN)
            .par_iter()
            .filter_map(|(pattern, _)| {
                let data = pm.read_tag(*pattern).ok()?;
                let set = embedded_words(&data)
                    .map(TagHash)
                    .find(|tag| set_clips.contains_key(tag))?;
                Some((*pattern, set))
            })
            .collect::<FxHashMap<_, _>>();
        let links = pm
            .get_all_by_reference(CLASS_ENTITY_DEFINITION)
            .par_iter()
            .filter_map(|(definition, _)| {
                let data = pm.read_tag(*definition).ok()?;
                let patterns = embedded_words(&data).map(TagHash).collect::<Vec<_>>();
                let sets = patterns
                    .iter()
                    .filter_map(|pattern| set_of_pattern.get(pattern))
                    .copied()
                    .collect::<Vec<_>>();
                let held = patterns
                    .into_iter()
                    .filter(|pattern| skeletons.contains(pattern))
                    .collect::<Vec<_>>();
                (!sets.is_empty() && !held.is_empty()).then_some((held, sets))
            })
            .collect::<Vec<_>>();
        let mut sets_of_skeleton = FxHashMap::<TagHash, Vec<TagHash>>::default();
        for (held, sets) in links {
            for skeleton in held {
                sets_of_skeleton
                    .entry(skeleton)
                    .or_default()
                    .extend(sets.iter().copied());
            }
        }
        for sets in sets_of_skeleton.values_mut() {
            sets.sort_by_key(|set| set.0);
            sets.dedup();
        }
        let mut clips_of_rig = FxHashMap::<u32, Vec<ClipInfo>>::default();
        for clip in clips.values() {
            clips_of_rig.entry(clip.rig).or_default().push(*clip);
        }
        log::info!(
            "Animation index: {} clips, {} sets, {} skeletons with a set",
            clips.len(),
            set_clips.len(),
            sets_of_skeleton.len()
        );
        Self {
            sets_of_skeleton,
            set_clips,
            clips_of_rig,
        }
    }

    /// Playable clips for the entities built on `skeleton`. Identical copies
    /// of a skeleton in other patterns count as the same skeleton.
    pub fn model_clips(&self, skeleton: &Skeleton) -> ModelClips {
        let nodes = skeleton.names.len();
        let fits = |clip: &&ClipInfo| clip.base && clip.slots == nodes;
        let mut own = all_skeletons()
            .iter()
            .filter(|other| {
                other.names == skeleton.names
                    && other.parents == skeleton.parents
                    && other.bind == skeleton.bind
            })
            .filter_map(|other| self.sets_of_skeleton.get(&other.tag))
            .flatten()
            .filter_map(|set| self.set_clips.get(set))
            .flatten()
            .filter(fits)
            .copied()
            .collect::<Vec<_>>();
        own.sort_by_key(|clip| clip.tag.0);
        own.dedup_by_key(|clip| clip.tag);
        let listed = own
            .iter()
            .map(|clip| clip.tag)
            .collect::<rustc_hash::FxHashSet<_>>();
        let rigs = own
            .iter()
            .map(|clip| clip.rig)
            .collect::<rustc_hash::FxHashSet<_>>();
        let mut same_rig = rigs
            .iter()
            .filter_map(|rig| self.clips_of_rig.get(rig))
            .flatten()
            .filter(fits)
            .filter(|clip| !listed.contains(&clip.tag))
            .copied()
            .collect::<Vec<_>>();
        same_rig.sort_by_key(|clip| clip.tag.0);
        ModelClips { own, same_rig }
    }
}

static RIG_INDEX: OnceLock<RigIndex> = OnceLock::new();

pub fn rig_index() -> &'static RigIndex {
    RIG_INDEX.get_or_init(RigIndex::build)
}

/// `rig_index` without the wait: builds it on a worker thread the first time
/// and returns `None` until it is ready.
pub fn rig_index_if_ready() -> Option<&'static RigIndex> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        std::thread::spawn(rig_index);
    });
    RIG_INDEX.get()
}

impl RigIndex {
    /// Clips to offer for a model built on `skeleton`.
    ///
    /// A runner the packages name gets the clips filed under it. A runner they do
    /// not name gets what its entity definitions list once everything filed under
    /// a named runner, the rig all runners share and multi-size rig families are
    /// taken out. Anything else gets its definitions' clips and the rest of its rig.
    pub fn clips_for_model(&self, skeleton: &Skeleton, codename: Option<&str>) -> Vec<ClipInfo> {
        let groups = runner_clip_groups();
        if !skeleton.is_runner() {
            // Several rigs of one size can share a set; keep those whose clips
            // rest in this skeleton's bind pose.
            let clips = self.model_clips(skeleton);
            let candidates = clips
                .own
                .into_iter()
                .chain(clips.same_rig)
                .collect::<Vec<_>>();
            let mut fits = rustc_hash::FxHashMap::<u32, (f32, usize)>::default();
            for clip in &candidates {
                let fit = fits.entry(clip.rig).or_default();
                if fit.1 < REST_FIT_SAMPLES {
                    if let Some(share) =
                        Clip::load(clip.tag).and_then(|loaded| loaded.rest_fit(skeleton))
                    {
                        *fit = (fit.0 + share, fit.1 + 1);
                    }
                }
            }
            return candidates
                .into_iter()
                .filter(|clip| {
                    // A rig with nothing to measure is kept rather than hidden.
                    fits.get(&clip.rig).is_none_or(|(sum, count)| {
                        *count == 0 || sum / *count as f32 >= REST_FIT_SHARE
                    })
                })
                .collect();
        }
        if let Some(group) = groups
            .iter()
            .find(|group| codename.is_some() && group.codename.as_deref() == codename)
        {
            return group.clips.clone();
        }
        let named = groups
            .iter()
            .filter(|group| group.codename.is_some())
            .flat_map(|group| group.clips.iter().map(|clip| clip.tag))
            .collect::<rustc_hash::FxHashSet<_>>();
        let base_rig = runner_clips().first().map(|_| {
            let mut sizes = rustc_hash::FxHashMap::<u32, usize>::default();
            for clip in runner_clips() {
                *sizes.entry(clip.rig).or_default() += 1;
            }
            sizes
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map_or(0, |(rig, _)| rig)
        });
        let is_family = |rig: u32| {
            self.clips_of_rig.get(&rig).is_some_and(|clips| {
                clips
                    .iter()
                    .map(|clip| clip.slots)
                    .collect::<rustc_hash::FxHashSet<_>>()
                    .len()
                    > 2
            })
        };
        let clips = self.model_clips(skeleton);
        clips
            .own
            .into_iter()
            .chain(clips.same_rig)
            .filter(|clip| {
                !named.contains(&clip.tag) && Some(clip.rig) != base_rig && !is_family(clip.rig)
            })
            .collect()
    }
}

/// Sparse (dominant bone, object-space position) pairs from a packed layout-7
/// source, enough to match the mesh to the skeleton it indexes.
pub fn bone_samples(source: &crate::geometry::AuthoredGeometryInput) -> Vec<BoneSample> {
    if let Some(samples) = float_bone_samples(source) {
        return samples;
    }
    let (Some(stream), Some(palette), Some(transform)) = (
        source.vertex_streams.iter().find(|stream| stream.stream_index == 0 && stream.stride == 24),
        source.skinning_buffer.as_ref(),
        source.position_transform,
    ) else {
        return Vec::new();
    };
    let (Ok(vertices), Ok(palette)) = (
        package_manager().read_tag(stream.data_tag),
        package_manager().read_tag(palette.data_tag),
    ) else {
        return Vec::new();
    };
    let step = (vertices.len() / 24 / 512).max(1);
    vertices
        .chunks_exact(24)
        .enumerate()
        .step_by(step)
        .filter_map(|(index, vertex)| {
            let component = |lane: usize| i16::from_le_bytes([vertex[lane * 2], vertex[lane * 2 + 1]]);
            Some(BoneSample {
                bone: dominant_bone(index, component(3), &palette)?,
                position: std::array::from_fn(|lane| {
                    f32::from(component(lane)) / 32767.0 * transform.scale[lane] + transform.offset[lane]
                }),
            })
        })
        .collect()
}

/// Bone influences of a float-format vertex: its skin stream holds four
/// weights then four bone indices, a byte each, with unused lanes weighted 0.
pub fn float_vertex_skin(record: &[u8]) -> [(u16, f32); 4] {
    std::array::from_fn(|lane| match (record.get(lane), record.get(4 + lane)) {
        (Some(weight), Some(bone)) if *weight > 0 => (u16::from(*bone), f32::from(*weight) / 255.0),
        _ => (0, 0.0),
    })
}

/// The position and skin streams of a float-format source (48-byte vertices
/// of float position, normal and tangent; 8-byte skin records).
pub fn float_source_streams(
    source: &crate::geometry::AuthoredGeometryInput,
) -> Option<(&crate::geometry::AuthoredVertexStreamRef, &crate::geometry::AuthoredVertexStreamRef)> {
    let vertices = source.vertex_streams.iter().find(|stream| stream.stream_index == 0 && stream.stride == 48)?;
    let skin = source
        .vertex_streams
        .iter()
        .find(|stream| stream.stream_index == 2 && stream.stride == 8 && stream.element_count == vertices.element_count)?;
    Some((vertices, skin))
}

/// `bone_samples` for a float-format source.
fn float_bone_samples(source: &crate::geometry::AuthoredGeometryInput) -> Option<Vec<BoneSample>> {
    let (vertices, skin) = float_source_streams(source)?;
    let transform = source.position_transform?;
    let vertices = package_manager().read_tag(vertices.data_tag).ok()?;
    let skin = package_manager().read_tag(skin.data_tag).ok()?;
    let step = (vertices.len() / 48 / 512).max(1);
    Some(
        vertices
            .chunks_exact(48)
            .zip(skin.chunks_exact(8))
            .step_by(step)
            .filter_map(|(vertex, record)| {
                let (bone, _) = float_vertex_skin(record)
                    .into_iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .filter(|(_, weight)| *weight > 0.0)?;
                Some(BoneSample {
                    bone: u32::from(bone),
                    position: std::array::from_fn(|lane| {
                        f32_at(vertex, lane * 4).unwrap_or(0.0) * transform.scale[lane] + transform.offset[lane]
                    }),
                })
            })
            .collect(),
    )
}

/// Entity definitions gathering each skeleton pattern.
fn definitions_by_skeleton() -> &'static rustc_hash::FxHashMap<TagHash, Vec<TagHash>> {
    static DEFINITIONS: OnceLock<rustc_hash::FxHashMap<TagHash, Vec<TagHash>>> = OnceLock::new();
    DEFINITIONS.get_or_init(|| {
        let pm = package_manager();
        let skeletons = all_skeletons().iter().map(|skeleton| skeleton.tag).collect::<rustc_hash::FxHashSet<_>>();
        let held = pm
            .get_all_by_reference(CLASS_ENTITY_DEFINITION)
            .par_iter()
            .filter_map(|(definition, _)| {
                let data = pm.read_tag(*definition).ok()?;
                let mut held = embedded_words(&data).map(TagHash).filter(|tag| skeletons.contains(tag)).collect::<Vec<_>>();
                held.sort_by_key(|tag| tag.0);
                held.dedup();
                (!held.is_empty()).then_some((*definition, held))
            })
            .collect::<Vec<_>>();
        let mut definitions = rustc_hash::FxHashMap::<TagHash, Vec<TagHash>>::default();
        for (definition, held) in held {
            for skeleton in held {
                definitions.entry(skeleton).or_default().push(definition);
            }
        }
        for list in definitions.values_mut() {
            list.sort_by_key(|tag| tag.0);
        }
        definitions
    })
}

/// The runner body skeleton a model's sources are skinned to, if any.
pub fn runner_body_skeleton(sources: &[crate::geometry::AuthoredGeometryInput]) -> Option<&'static Skeleton> {
    sources
        .iter()
        .filter_map(|source| match_skeleton_where(&bone_samples(source), Skeleton::is_player_rig))
        .filter(|skeleton| skeleton.is_runner())
        .max_by_key(|skeleton| skeleton.names.len())
}

/// Every entity definition built on `skeleton` or an identical copy of it.
pub fn definitions_of_skeleton(skeleton: &Skeleton) -> Vec<TagHash> {
    let definitions = definitions_by_skeleton();
    let mut found = all_skeletons()
        .iter()
        .filter(|other| other.names == skeleton.names && other.parents == skeleton.parents && other.bind == skeleton.bind)
        .filter_map(|other| definitions.get(&other.tag))
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    found.sort_by_key(|tag| tag.0);
    found.dedup();
    found
}

const CLASS_GEOMETRY: u32 = 0x8080881C;
/// Hash of the empty string: the key of entries no other set pairs with.
const NO_SYNC_KEY: u32 = 0x811C9DC5;

/// 32-bit tags by their 64-bit hash.
fn tags_by_hash64() -> &'static rustc_hash::FxHashMap<u64, TagHash> {
    static TAGS: OnceLock<rustc_hash::FxHashMap<u64, TagHash>> = OnceLock::new();
    TAGS.get_or_init(|| {
        package_manager().lookup.tag32_to_tag64.iter().map(|(tag32, tag64)| (tag64.0, *tag32)).collect()
    })
}

fn tag_class(tag: TagHash) -> Option<u32> {
    Some(package_manager().get_entry(tag)?.reference)
}

/// Components an entity definition gathers.
fn definition_components(definition: TagHash) -> Vec<TagHash> {
    let Ok(data) = package_manager().read_tag(definition) else {
        return vec![];
    };
    let mut components = embedded_words(&data)
        .map(TagHash)
        .filter(|tag| tag_class(*tag) == Some(CLASS_PATTERN))
        .collect::<Vec<_>>();
    components.sort_by_key(|tag| tag.0);
    components.dedup();
    components
}

/// Sync keys of every animation set entry playing a clip. An entity spawned
/// alongside plays the entry of its own set carrying the same key.
fn clip_sync_keys(clip: TagHash) -> &'static [u32] {
    static KEYS: OnceLock<rustc_hash::FxHashMap<TagHash, Vec<u32>>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut keys = rustc_hash::FxHashMap::<TagHash, Vec<u32>>::default();
        for (set, _) in package_manager().get_all_by_reference(CLASS_ANIMATION_SET) {
            for entry in animation_set_entries(set, tags_by_hash64()) {
                if entry.hashes[1] != NO_SYNC_KEY {
                    keys.entry(entry.clip).or_default().push(entry.hashes[1]);
                }
            }
        }
        keys
    })
    .get(&clip)
    .map_or(&[], Vec::as_slice)
}

/// A prop a clip spawns into one of its entity's sockets for as long as it
/// plays: Sentinel takes the Defender System off his thigh this way.
#[derive(Debug, Clone)]
pub struct ClipProp {
    /// Name hash of the socket the prop is attached to.
    pub socket: u32,
    /// Frame of the spawning clip at which the prop appears.
    pub start: usize,
    pub geometry: Vec<TagHash>,
    pub skeleton: &'static Skeleton,
    /// The prop's own clip paired with the spawning one, when its set has one.
    pub clip: Option<std::sync::Arc<Clip>>,
}

impl Clip {
    /// Props the clip spawns.
    ///
    /// The clip lists what it starts while playing (array at 0x160): sounds
    /// and sequences, each with the socket it concerns. A sequence is an
    /// entity definition whose components name the prop's definition; the
    /// prop brings a skeleton, geometry and an animation set of its own.
    pub fn props(&self) -> std::sync::Arc<Vec<ClipProp>> {
        type Props = rustc_hash::FxHashMap<TagHash, std::sync::Arc<Vec<ClipProp>>>;
        static PROPS: OnceLock<std::sync::Mutex<Props>> = OnceLock::new();
        let cache = PROPS.get_or_init(Default::default);
        if let Some(props) = cache.lock().unwrap().get(&self.tag) {
            return props.clone();
        }
        let props = std::sync::Arc::new(self.spawned_props());
        cache.lock().unwrap().insert(self.tag, props.clone());
        props
    }

    fn spawned_props(&self) -> Vec<ClipProp> {
        let data = &self.data;
        let Some((count, start)) = array_at(data, 0x160) else {
            return vec![];
        };
        let sync_keys = clip_sync_keys(self.tag);
        (0..count)
            .filter_map(|index| {
                // Each entry points at its record: timing, socket, name, tag.
                let pointer = start + index * 8;
                let record = pointer + u64::from_le_bytes(data.get(pointer..pointer + 8)?.try_into().ok()?) as usize;
                let start_frame = usize::from(u16_at(data, record)?);
                let socket = u32_at(data, record + 0x8)?;
                let started = *tags_by_hash64()
                    .get(&u64::from_le_bytes(data.get(record + 0x20..record + 0x28)?.try_into().ok()?))?;
                (tag_class(started) == Some(CLASS_ENTITY_DEFINITION)).then_some(())?;
                // The sequence's components name the definitions it spawns.
                let spawned = definition_components(started)
                    .into_iter()
                    .filter_map(|component| package_manager().read_tag(component).ok())
                    .flat_map(|component| {
                        component
                            .windows(8)
                            .step_by(4)
                            .filter_map(|hash| tags_by_hash64().get(&u64::from_le_bytes(hash.try_into().ok()?)).copied())
                            .collect::<Vec<_>>()
                    })
                    .filter(|tag| *tag != started && tag_class(*tag) == Some(CLASS_ENTITY_DEFINITION));
                spawned.into_iter().find_map(|definition| {
                    let components = definition_components(definition);
                    let skeleton = all_skeletons().iter().find(|skeleton| components.contains(&skeleton.tag))?;
                    let referenced = |class: u32| {
                        let mut tags = components
                            .iter()
                            .filter_map(|component| package_manager().read_tag(*component).ok())
                            .flat_map(|component| embedded_words(&component).map(TagHash).collect::<Vec<_>>())
                            .filter(|tag| tag_class(*tag) == Some(class))
                            .collect::<Vec<_>>();
                        tags.sort_by_key(|tag| tag.0);
                        tags.dedup();
                        tags
                    };
                    let geometry = referenced(CLASS_GEOMETRY);
                    (!geometry.is_empty()).then_some(())?;
                    let clip = referenced(CLASS_ANIMATION_SET)
                        .into_iter()
                        .flat_map(|set| animation_set_entries(set, tags_by_hash64()))
                        .find(|entry| sync_keys.contains(&entry.hashes[1]))
                        .and_then(|entry| Clip::load(entry.clip))
                        .map(std::sync::Arc::new);
                    Some(ClipProp { socket, start: start_frame, geometry, skeleton, clip })
                })
            })
            .collect()
    }
}

/// Where a named socket sits on a skeleton: the node it hangs off and its
/// offset from that node.
///
/// Entities built on the skeleton list their sockets in a component, each a
/// rotation, a translation, the node index and the socket's name hash.
fn skeleton_socket(skeleton: &'static Skeleton, socket: u32) -> Option<(u32, Transform)> {
    type Sockets = rustc_hash::FxHashMap<(TagHash, u32), Option<(u32, Transform)>>;
    static SOCKETS: OnceLock<std::sync::Mutex<Sockets>> = OnceLock::new();
    let cache = SOCKETS.get_or_init(Default::default);
    if let Some(found) = cache.lock().unwrap().get(&(skeleton.tag, socket)) {
        return *found;
    }
    let found = definitions_of_skeleton(skeleton)
        .into_iter()
        .flat_map(definition_components)
        .filter_map(|component| package_manager().read_tag(component).ok())
        .find_map(|data| {
            (0x28..data.len().saturating_sub(4)).step_by(4).find_map(|offset| {
                (u32_at(&data, offset)? == socket).then_some(())?;
                let node = *skeleton.names.get(u32_at(&data, offset - 4)? as usize)?;
                let value = |index: usize| f32_at(&data, offset - 0x28 + index * 4);
                let rotation = [value(0)?, value(1)?, value(2)?, value(3)?];
                let length = rotation.iter().map(|c| c * c).sum::<f32>();
                // A socket record: unit rotation, unit scale, a zero word
                // before the node index and an instance id after the name.
                ((length - 1.0).abs() < 1e-3
                    && (value(7)? - 1.0).abs() < 1e-2
                    && u32_at(&data, offset - 8)? == 0
                    && u32_at(&data, offset + 4)? >> 24 == 0x84)
                    .then_some(())?;
                Some((node, Transform { rotation, translation: [value(4)?, value(5)?, value(6)?], scale: 1.0 }))
            })
        });
    cache.lock().unwrap().insert((skeleton.tag, socket), found);
    found
}

fn compose(parent: Transform, child: Transform) -> Transform {
    let rotated = quat_rotate(parent.rotation, child.translation);
    Transform {
        rotation: normalize(quat_mul(parent.rotation, child.rotation)),
        translation: std::array::from_fn(|c| parent.translation[c] + rotated[c] * parent.scale),
        scale: parent.scale * child.scale,
    }
}

/// `child` in the frame of `parent`.
fn relative(parent: Transform, child: Transform) -> Transform {
    let inverse = quat_conjugate(parent.rotation);
    Transform {
        rotation: normalize(quat_mul(inverse, child.rotation)),
        translation: quat_rotate(
            inverse,
            std::array::from_fn(|c| child.translation[c] - parent.translation[c]),
        ),
        scale: 1.0,
    }
}

/// How far apart two placements are: metres, and one minus the cosine of
/// half the angle between their orientations.
fn placement_gap(a: Transform, b: Transform) -> (f32, f32) {
    let distance = (0..3)
        .map(|c| (a.translation[c] - b.translation[c]).powi(2))
        .sum::<f32>()
        .sqrt();
    let dot = (0..4).map(|c| a.rotation[c] * b.rotation[c]).sum::<f32>().abs();
    (distance, 1.0 - dot.min(1.0))
}

/// A stowed socket stays within this of where the clip parks it: 5 mm and
/// about a degree and a half.
const STOWED_GAP: (f32, f32) = (0.005, 1.0e-4);
/// Object channel a clip raises for as long as the item it spawns is out of
/// its holster (Triage's drone fidget: from the spawn frame until the frame
/// the drone is seated again).
const CHANNEL_ITEM_OUT: u32 = 0xB74F5FD9;

impl Clip {
    /// The skeleton of `skeletons` this clip plays on and its object-space
    /// pose at `frame` (the bind pose for a negative frame).
    fn driven_pose(
        &self,
        skeletons: &[Option<&'static Skeleton>],
        frame: f32,
    ) -> Option<(&'static Skeleton, Vec<Transform>)> {
        let mut candidates = skeletons.iter().flatten().copied();
        let driven = if self.control_rig {
            candidates
                .filter(|skeleton| skeleton.is_runner())
                .max_by_key(|skeleton| skeleton.names.len())?
        } else {
            candidates.find(|skeleton| skeleton.names.len() == self.slots)?
        };
        let pose = if frame < 0.0 {
            driven.bind.clone()
        } else if self.control_rig {
            RunnerRig::cached(driven, self.slots).pose(&self.sample(frame))
        } else {
            driven.local_pose(&self.sample(frame))
        };
        Some((driven, pose))
    }

    /// Whether a prop this clip spawns is stowed at `frame`: its socket is
    /// where the clip leaves it on the body once the item is put away. A clip
    /// takes an item from its holster and returns it there, so the socket sits
    /// in one place on one body node both just before the spawn and at the end
    /// of the clip. The holder's own stowed copy is what shows then.
    pub fn prop_stowed(
        &self,
        prop: &ClipProp,
        skeletons: &[Option<&'static Skeleton>],
        frame: f32,
    ) -> bool {
        // A clip that says outright when its item is out is taken at its word.
        if let Some((_, out)) = self
            .channel_values(frame.max(0.0))
            .into_iter()
            .find(|(channel, _)| *channel == CHANNEL_ITEM_OUT)
        {
            return out < 0.5;
        }
        let Some((driven, _)) = self.driven_pose(skeletons, frame.max(0.0)) else {
            return false;
        };
        // Once the item is back it stays put away: the end of a clip eases
        // the whole rig back to its idle pose, and the socket drifts off its
        // holster again on the way without anything being taken out.
        type Returns = rustc_hash::FxHashMap<(TagHash, u32, TagHash), Option<(usize, usize)>>;
        static RETURNS: OnceLock<std::sync::Mutex<Returns>> = OnceLock::new();
        let key = (self.tag, prop.socket, driven.tag);
        let cached = RETURNS.get_or_init(Default::default).lock().unwrap().get(&key).copied();
        let out = cached.unwrap_or_else(|| {
            let out = self.out_of_holster(prop, skeletons, driven);
            RETURNS.get_or_init(Default::default).lock().unwrap().insert(key, out);
            out
        });
        match out {
            Some((taken, returned)) => frame < taken as f32 || frame >= returned as f32,
            None => false,
        }
    }

    /// The frames a spawned prop spends away from its holster: the first on
    /// which its socket has left its stowed place, and the first after that on
    /// which it is back. `None` when the clip never stows the item.
    fn out_of_holster(
        &self,
        prop: &ClipProp,
        skeletons: &[Option<&'static Skeleton>],
        driven: &'static Skeleton,
    ) -> Option<(usize, usize)> {
        let (socket, anchor, place) = self.parked(prop, skeletons, driven)?;
        let stowed = |frame: usize| {
            self.driven_pose(skeletons, frame as f32).is_some_and(|(_, pose)| {
                let gap = placement_gap(relative(pose[anchor], pose[socket]), place);
                gap.0 <= STOWED_GAP.0 && gap.1 <= STOWED_GAP.1
            })
        };
        let taken = (prop.start..self.frames).find(|frame| !stowed(*frame))?;
        let returned = (taken..self.frames).find(|frame| stowed(*frame)).unwrap_or(self.frames);
        Some((taken, returned))
    }

    /// `parked_socket`, worked out once per clip, prop and skeleton.
    fn parked(
        &self,
        prop: &ClipProp,
        skeletons: &[Option<&'static Skeleton>],
        driven: &'static Skeleton,
    ) -> Option<(usize, usize, Transform)> {
        type Parked = rustc_hash::FxHashMap<(TagHash, u32, TagHash), Option<(usize, usize, Transform)>>;
        static PARKED: OnceLock<std::sync::Mutex<Parked>> = OnceLock::new();
        let key = (self.tag, prop.socket, driven.tag);
        let cached = PARKED.get_or_init(Default::default).lock().unwrap().get(&key).copied();
        cached.unwrap_or_else(|| {
            let parked = self.parked_socket(prop, skeletons);
            PARKED.get_or_init(Default::default).lock().unwrap().insert(key, parked);
            parked
        })
    }

    /// Where a spawned prop's socket is relative to its holster at `frame`:
    /// the body node the holster is on, how far the socket is from its stowed
    /// place there, and the socket's object-space position. Only the prop
    /// hand-off correction (`gui/prop_handoff.rs`) uses this.
    pub fn prop_stow(
        &self,
        prop: &ClipProp,
        skeletons: &[Option<&'static Skeleton>],
        frame: f32,
    ) -> Option<(usize, f32, [f32; 3])> {
        let (driven, pose) = self.driven_pose(skeletons, frame.max(0.0))?;
        let (socket, anchor, place) = self.parked(prop, skeletons, driven)?;
        let gap = placement_gap(relative(pose[anchor], pose[socket]), place);
        Some((anchor, gap.0, pose[socket].translation))
    }

    /// The socket node of a spawned prop, the body node it rests on while the
    /// item is put away, and its placement on that node.
    fn parked_socket(
        &self,
        prop: &ClipProp,
        skeletons: &[Option<&'static Skeleton>],
    ) -> Option<(usize, usize, Transform)> {
        // A prop out from the first frame was never stowed.
        let before = prop.start.checked_sub(1)?;
        let last = self.frames.checked_sub(1).filter(|last| *last > prop.start)?;
        let (driven, first) = self.driven_pose(skeletons, last as f32)?;
        let socket = skeleton_socket(driven, prop.socket)?.0;
        let socket = driven.names.iter().position(|name| *name == socket)?;
        // The frame before the spawn, and a little before the end.
        let later = [before, last.saturating_sub(4).max(prop.start)]
            .map(|frame| self.driven_pose(skeletons, frame as f32).map(|(_, pose)| pose));
        let [Some(middle), Some(end)] = later else {
            return None;
        };
        // The limb carrying the socket moves with it whether or not the item
        // is stowed: everything from where that limb leaves the torso.
        let torso = ["b_pelvis", "b_spine_1", "b_spine_2", "b_spine_3"].map(|name| driven.node(name));
        let parent = |node: usize| usize::try_from(driven.parents[node]).ok();
        let mut limb = socket;
        while let Some(above) = parent(limb).filter(|above| !torso.contains(&Some(*above)) && parent(*above).is_some()) {
            limb = above;
        }
        let carried = (0..driven.names.len())
            .map(|node| std::iter::successors(Some(node), |node| parent(*node)).any(|node| node == limb))
            .collect::<Vec<_>>();
        (0..driven.names.len())
            .filter(|node| !carried[*node])
            .map(|node| {
                let place = relative(first[node], first[socket]);
                let gap = [&middle, &end]
                    .map(|pose| placement_gap(relative(pose[node], pose[socket]), place));
                let worst = (gap[0].0.max(gap[1].0), gap[0].1.max(gap[1].1));
                (node, place, worst)
            })
            .filter(|(_, _, worst)| worst.0 <= STOWED_GAP.0 && worst.1 <= STOWED_GAP.1)
            .min_by(|a, b| a.2.0.total_cmp(&b.2.0))
            .map(|(node, place, _)| (socket, node, place))
    }
}

impl ClipProp {
    /// The prop's nodes in the object space of the entity holding it, whose
    /// own posed nodes are `holder` (by node name) on skeleton `held_by`.
    fn pose(&self, held_by: &'static Skeleton, holder: &rustc_hash::FxHashMap<u32, Transform>, frame: f32) -> Option<Vec<Transform>> {
        let (node, offset) = skeleton_socket(held_by, self.socket)?;
        let socket = compose(*holder.get(&node)?, offset);
        let local = match &self.clip {
            Some(clip) => self.skeleton.local_pose(&clip.sample(frame.min((clip.frames - 1) as f32))),
            None => self.skeleton.bind.clone(),
        };
        Some(local.into_iter().map(|node| compose(socket, node)).collect())
    }
}
