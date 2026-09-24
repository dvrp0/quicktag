//! Authored gameplay-property assignments following investment curves.
//! Operands are constants, curve channels, or previously assigned properties.
use super::*;

#[derive(Clone, Debug, Default)]
pub(super) struct PropertyProgram(Vec<[u32; 24]>);

impl PropertyProgram {
    pub(super) fn parse(data: &[u8]) -> Option<Self> {
        let array = arrays(data).into_iter().find(|a| a.class == 0x8080_3f96)?;
        let mut rows = Vec::with_capacity(array.count);
        for index in 0..array.count {
            let start = array.start.checked_add(index.checked_mul(0x60)?)?;
            let mut row = [0; 24];
            for (word, value) in row.iter_mut().enumerate() {
                *value = read_u32(data, start + word * 4)?;
            }
            rows.push(row);
        }
        rows.sort_by_key(|row| row[23]);
        Some(Self(rows))
    }

    pub(super) fn evaluate(&self, curves: &[Option<Vec<f32>>]) -> FxHashMap<(u32, u32), f32> {
        let mut properties = FxHashMap::default();
        for row in &self.0 {
            let operand = |offset: usize| -> Option<f32> {
                let arg = &row[offset..offset + 6];
                if arg[5] != f32::MAX.to_bits() {
                    return Some(f32::from_bits(arg[5]));
                }
                if (0x3c00..0x4000).contains(&arg[4]) {
                    let address = (arg[4] - 0x3c00) as usize;
                    return curves
                        .get(address / 32)?
                        .as_ref()?
                        .get(address % 32)
                        .copied();
                }
                if arg[2] == u32::MAX {
                    return properties.get(&(arg[0], arg[1])).copied();
                }
                None
            };
            let a = operand(5);
            let b = operand(11);
            let result = match row[4] & 0xff {
                0 => a.zip(b).map(|(a, b)| a + b),
                1 => a.zip(b).map(|(a, b)| a - b),
                2 => a.zip(b).map(|(a, b)| a * b),
                3 => a.zip(b).and_then(|(a, b)| (b != 0.0).then_some(a / b)),
                4 => a,
                // Integer ammo properties use the authored round operation.
                7 => a.map(f32::round),
                _ => None,
            }
            .filter(|value| value.is_finite());
            let key = (row[0], row[1]);
            if let Some(value) = result {
                properties.insert(key, value);
            } else {
                // Never retain an earlier value when a later assignment failed.
                properties.remove(&key);
            }
        }
        properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_authored_property_updates_in_order() {
        let mut copy = [0; 24];
        copy[0] = 2;
        copy[1] = 70;
        copy[4] = 4;
        copy[9] = 0x3c00;
        copy[10] = f32::MAX.to_bits();
        let mut multiply = copy;
        multiply[4] = 0x102;
        multiply[5] = 2;
        multiply[6] = 70;
        multiply[7] = u32::MAX;
        multiply[9] = 0;
        multiply[16] = 0.89f32.to_bits();
        let program = PropertyProgram(vec![copy, multiply]);
        let values = program.evaluate(&[Some(vec![0.65])]);
        assert!((values[&(2, 70)] - 0.5785).abs() < 1e-6);
        assert!(!program.evaluate(&[None]).contains_key(&(2, 70)));
    }
}
