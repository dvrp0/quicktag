//! Prop hand-off correction: a viewer approximation, not game data.
//!
//! A clip that takes an item from its holster parks the item's socket on the
//! body, but for some runners the prop then sits a few centimetres and degrees
//! away from the holstered copy the runner's own mesh carries, so the swap
//! between the two jumps. This measures that offset from the two meshes (they
//! share one texture layout, so equal UVs are the same surface point) and
//! applies it to the prop while it is stowed, fading it out over the first
//! stretch the prop travels from its holster.
//!
//! To drop the correction: delete this file, its `mod` line in `gui/mod.rs`,
//! the marked block in `GpuModelPreview::clip_pose`, and `Clip::prop_stow`.

use crate::animation::{Quat, Transform, quat_mul, quat_rotate};

/// How far from its holster the socket is when the correction has faded out.
pub(crate) const REACH: f32 = 0.15;
/// Two surface points are the same when their UVs agree this closely.
const UV_MATCH: f32 = 3.0e-4;
/// Matched points this close after the fit count as agreeing with it.
const INLIER: f32 = 0.0015;
/// A fit needs this many agreeing points to be believed.
const MINIMUM_INLIERS: usize = 50;

/// A surface point: position and texture coordinates.
pub(crate) type Point = ([f32; 3], [f32; 2]);

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn unit(a: [f32; 3]) -> Option<[f32; 3]> {
    let size = length(a);
    (size > 1.0e-5).then(|| a.map(|c| c / size))
}

pub(crate) fn apply(transform: Transform, point: [f32; 3]) -> [f32; 3] {
    let rotated = quat_rotate(transform.rotation, point);
    std::array::from_fn(|c| rotated[c] + transform.translation[c])
}

pub(crate) fn compose(parent: Transform, child: Transform) -> Transform {
    let length = |q: Quat| q.iter().map(|c| c * c).sum::<f32>().sqrt();
    let rotation = quat_mul(parent.rotation, child.rotation);
    let size = length(rotation).max(1.0e-12);
    Transform {
        rotation: rotation.map(|c| c / size),
        translation: apply(parent, child.translation.map(|c| c * parent.scale)),
        scale: parent.scale * child.scale,
    }
}

pub(crate) fn inverse(transform: Transform) -> Transform {
    let rotation = [-transform.rotation[0], -transform.rotation[1], -transform.rotation[2], transform.rotation[3]];
    let moved = quat_rotate(rotation, transform.translation);
    Transform { rotation, translation: moved.map(|c| -c), scale: 1.0 }
}

/// Rotation taking the frame of three points onto the frame of three others.
fn frame(points: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let first = unit(sub(points[1], points[0]))?;
    let normal = unit(cross(first, sub(points[2], points[0])))?;
    Some([first, cross(normal, first), normal])
}

fn quaternion(columns: [[f32; 3]; 3]) -> Quat {
    // `columns[c][r]` is row r of column c of a rotation matrix.
    let m = |r: usize, c: usize| columns[c][r];
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, 0.25 * s]
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        [0.25 * s, (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s, (m(2, 1) - m(1, 2)) / s]
    } else if m(1, 1) > m(2, 2) {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        [(m(0, 1) + m(1, 0)) / s, 0.25 * s, (m(1, 2) + m(2, 1)) / s, (m(0, 2) - m(2, 0)) / s]
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        [(m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s, 0.25 * s, (m(1, 0) - m(0, 1)) / s]
    };
    let size = q.iter().map(|c| c * c).sum::<f32>().sqrt().max(1.0e-12);
    q.map(|c| c / size)
}

/// The rigid transform carrying three points exactly onto three others.
fn from_triangles(from: [[f32; 3]; 3], to: [[f32; 3]; 3]) -> Option<Transform> {
    let (a, b) = (frame(from)?, frame(to)?);
    // R = B * A^T, by columns.
    let columns: [[f32; 3]; 3] = std::array::from_fn(|c| {
        std::array::from_fn(|r| (0..3).map(|k| b[k][r] * a[k][c]).sum())
    });
    let rotation = quaternion(columns);
    let moved = quat_rotate(rotation, from[0]);
    Some(Transform { rotation, translation: sub(to[0], moved), scale: 1.0 })
}

/// Least-squares rigid fit of paired points (Horn's quaternion method).
fn fit(pairs: &[([f32; 3], [f32; 3])]) -> Option<Transform> {
    let count = pairs.len() as f32;
    (pairs.len() >= 3).then_some(())?;
    let mean = |pick: fn(&([f32; 3], [f32; 3])) -> [f32; 3]| -> [f32; 3] {
        let sum = pairs.iter().fold([0.0f32; 3], |sum, pair| std::array::from_fn(|c| sum[c] + pick(pair)[c]));
        sum.map(|c| c / count)
    };
    let (from, to) = (mean(|pair| pair.0), mean(|pair| pair.1));
    let mut s = [[0.0f64; 3]; 3];
    for (a, b) in pairs {
        let (a, b) = (sub(*a, from), sub(*b, to));
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += f64::from(a[r]) * f64::from(b[c]);
            }
        }
    }
    let n = [
        [s[0][0] + s[1][1] + s[2][2], s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]],
        [s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]],
        [s[2][0] - s[0][2], s[0][1] + s[1][0], s[1][1] - s[0][0] - s[2][2], s[1][2] + s[2][1]],
        [s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], s[2][2] - s[0][0] - s[1][1]],
    ];
    // Largest eigenvector by power iteration on a shifted, positive matrix.
    let shift: f64 = n.iter().flatten().map(|value| value.abs()).sum::<f64>() + 1.0e-12;
    let mut q = [1.0f64, 0.0, 0.0, 0.0];
    for _ in 0..200 {
        let next: [f64; 4] = std::array::from_fn(|r| (0..4).map(|c| n[r][c] * q[c]).sum::<f64>() + shift * q[r]);
        let size = next.iter().map(|c| c * c).sum::<f64>().sqrt();
        q = next.map(|c| c / size);
    }
    let rotation = [q[1] as f32, q[2] as f32, q[3] as f32, q[0] as f32];
    let moved = quat_rotate(rotation, from);
    Some(Transform { rotation, translation: sub(to, moved), scale: 1.0 })
}

/// The rigid transform carrying `prop` onto the part of `copy` showing the
/// same surface, and the indices of the prop points that agree with it.
pub(crate) fn register(prop: &[Point], copy: &[Point]) -> Option<(Transform, Vec<usize>)> {
    // Pair up points with equal UVs through a grid over texture space.
    let cell = |uv: [f32; 2]| ((uv[0] / UV_MATCH).floor() as i64, (uv[1] / UV_MATCH).floor() as i64);
    let mut grid = rustc_hash::FxHashMap::<(i64, i64), Vec<usize>>::default();
    for (index, (_, uv)) in copy.iter().enumerate() {
        grid.entry(cell(*uv)).or_default().push(index);
    }
    let mut pairs = vec![];
    for (index, (_, uv)) in prop.iter().enumerate() {
        let (x, y) = cell(*uv);
        for neighbour in (-1..=1).flat_map(|dx| (-1..=1).map(move |dy| (x + dx, y + dy))) {
            for other in grid.get(&neighbour).into_iter().flatten() {
                let gap = [uv[0] - copy[*other].1[0], uv[1] - copy[*other].1[1]];
                if gap[0].hypot(gap[1]) < UV_MATCH {
                    pairs.push((index, *other));
                }
            }
        }
    }
    (pairs.len() >= MINIMUM_INLIERS).then_some(())?;
    let agreeing = |transform: Transform| {
        pairs
            .iter()
            .filter(|(a, b)| length(sub(apply(transform, prop[*a].0), copy[*b].0)) < INLIER)
            .copied()
            .collect::<Vec<_>>()
    };
    // Sample triangles of pairs; keep the transform most pairs agree with.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut pick = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % pairs.len() as u64) as usize
    };
    let mut best: Option<(usize, Transform)> = None;
    for _ in 0..4000 {
        let chosen = [pick(), pick(), pick()].map(|index| pairs[index]);
        let from = chosen.map(|(a, _)| prop[a].0);
        let to = chosen.map(|(_, b)| copy[b].0);
        // Congruent triangles only: a wrong pairing rarely keeps its lengths.
        let congruent = [(0, 1), (0, 2), (1, 2)]
            .iter()
            .all(|(i, j)| (length(sub(from[*i], from[*j])) - length(sub(to[*i], to[*j]))).abs() < INLIER);
        if !congruent {
            continue;
        }
        let Some(transform) = from_triangles(from, to) else { continue };
        let score = agreeing(transform).len();
        if best.as_ref().is_none_or(|(most, _)| score > *most) {
            best = Some((score, transform));
        }
    }
    let (_, mut transform) = best?;
    for _ in 0..3 {
        let inliers = agreeing(transform);
        let points = inliers.iter().map(|(a, b)| (prop[*a].0, copy[*b].0)).collect::<Vec<_>>();
        transform = fit(&points)?;
    }
    let mut inliers = agreeing(transform).into_iter().map(|(a, _)| a).collect::<Vec<_>>();
    inliers.sort_unstable();
    inliers.dedup();
    (inliers.len() >= MINIMUM_INLIERS).then_some((transform, inliers))
}

/// `weight` of a correction applied about `pivot`: its rotation and the shift
/// it gives the pivot both scale with the weight.
pub(crate) fn blended(correction: Transform, pivot: [f32; 3], weight: f32) -> Transform {
    let sign = if correction.rotation[3] < 0.0 { -1.0 } else { 1.0 };
    let mixed: Quat = std::array::from_fn(|c| {
        correction.rotation[c] * sign * weight + if c == 3 { 1.0 - weight } else { 0.0 }
    });
    let size = mixed.iter().map(|c| c * c).sum::<f32>().sqrt().max(1.0e-12);
    let rotation = mixed.map(|c| c / size);
    let shift = sub(apply(correction, pivot), pivot);
    let turned = quat_rotate(rotation, pivot);
    Transform {
        rotation,
        translation: std::array::from_fn(|c| pivot[c] - turned[c] + shift[c] * weight),
        scale: 1.0,
    }
}
