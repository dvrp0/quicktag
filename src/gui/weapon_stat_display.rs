//! Package-authored weapon-stat presentation expressions (808028FB).
//! This VM is independent of gameplay property evaluation and renderer bytecode.
use super::*;

#[derive(Clone, Debug)]
struct DisplayProgram {
    code: Vec<u8>,
    constants: Vec<f32>,
    inputs: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub(super) struct StatDisplayPrograms(Vec<DisplayProgram>);

fn checked_table(
    data: &[u8],
    header: usize,
    stride: usize,
    class: u32,
) -> Option<std::ops::Range<usize>> {
    let range = table_range(data, header, stride)?;
    if !range.is_empty() && read_u32(data, range.start.checked_sub(8)?)? != class {
        return None;
    }
    Some(range)
}

impl StatDisplayPrograms {
    pub(super) fn load() -> Option<Self> {
        let pm = package_manager();
        let candidates = pm.get_all_by_reference(0x8080_28f7);
        let [(tag, _)] = candidates.as_slice() else {
            return None;
        };
        Self::parse(&pm.read_tag(*tag).ok()?)
    }

    fn parse(data: &[u8]) -> Option<Self> {
        let rows = checked_table(data, 8, 0x70, 0x8080_28fb)?;
        let mut programs = Vec::new();
        for row in (rows.start..rows.end).step_by(0x70) {
            let code = checked_table(data, row + 0x28, 1, 0x8080_0009)?;
            let constants = checked_table(data, row + 0x38, 16, 0x8080_0090)?;
            let inputs = checked_table(data, row + 0x58, 12, 0x8080_2e3f)?;
            let constants = data[constants]
                .chunks_exact(16)
                .map(|row| {
                    let scalar = read_u32(row, 0)?;
                    // These display expressions are scalar splats, not general vectors.
                    (1..4)
                        .all(|i| read_u32(row, i * 4) == Some(scalar))
                        .then_some(f32::from_bits(scalar))
                })
                .collect::<Option<Vec<_>>>()?;
            let inputs = data[inputs]
                .chunks_exact(12)
                .map(|row| {
                    // Display source groups are one-based; property destinations are zero-based.
                    (read_u32(row, 8)? == 0xff)
                        .then_some((read_u32(row, 0)?.checked_sub(1)?, read_u32(row, 4)?))
                })
                .collect::<Option<Vec<_>>>()?;
            programs.push(DisplayProgram {
                code: data[code].to_vec(),
                constants,
                inputs,
            });
        }
        Some(Self(programs))
    }

    pub(super) fn debug_dump(&self) -> String {
        self.0
            .iter()
            .enumerate()
            .map(|(index, program)| {
                format!(
                    "  DISPLAY {index:2} inputs={:x?} constants={:?} code={:02x?}\n",
                    program.inputs, program.constants, program.code
                )
            })
            .collect()
    }

    /// Firepower from damage per projectile and the precision bonus. A pellet
    /// weapon uses the program authored for its pellet count; any other weapon,
    /// or a pellet count without its own program, uses the plain product.
    pub(super) fn firepower(&self, damage: f32, bonus: f32, pellets: Option<f32>) -> Option<f32> {
        let mut programs = self
            .0
            .iter()
            .filter(|program| program.inputs.as_slice() == [(2, 0x25), (2, 0x29)]);
        let program = pellets
            .and_then(|pellets| {
                programs
                    .clone()
                    .find(|program| program.constants.len() == 2 && program.constants[1] == pellets)
            })
            .or_else(|| programs.find(|program| program.constants.len() == 1))?;
        program.evaluate(&[damage, bonus])
    }

    /// Resolve only unambiguous single-property displays. Variant-dependent rows
    /// require their authored UI selection binding, never a first-match fallback.
    pub(super) fn single_property(&self, property: (u32, u32), value: f32) -> Option<f32> {
        let mut matching = self
            .0
            .iter()
            .filter(|program| program.inputs.as_slice() == [property]);
        let program = matching.next()?;
        if matching.next().is_some() {
            return None;
        }
        program.evaluate(&[value])
    }
}

impl DisplayProgram {
    fn evaluate(&self, inputs: &[f32]) -> Option<f32> {
        let mut code = self.code.iter().copied();
        let mut stack = Vec::<f32>::new();
        let mut output = None;
        while let Some(op) = code.next() {
            let result = match op {
                0x42 => *self.constants.get(code.next()? as usize)?,
                0x4a => *inputs.get(code.next()? as usize)?,
                0x4c => {
                    if code.next()? != 0 || output.is_some() {
                        return None;
                    }
                    output = stack.pop();
                    continue;
                }
                0x18 => stack.pop()?.abs(),
                0x15 => {
                    let c = stack.pop()?;
                    let b = stack.pop()?;
                    stack.pop()? * b + c
                }
                1..=4 | 8 | 9 => {
                    let b = stack.pop()?;
                    let a = stack.pop()?;
                    match op {
                        1 => a + b,
                        2 => a - b,
                        3 => a * b,
                        4 => a / b,
                        8 => a.min(b),
                        9 => a.max(b),
                        _ => unreachable!(),
                    }
                }
                _ => return None,
            };
            if !result.is_finite() {
                return None;
            }
            stack.push(result);
        }
        if !stack.is_empty() {
            return None;
        }
        output
    }
}
