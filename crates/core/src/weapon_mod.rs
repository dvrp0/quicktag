//! Lossless weapon-mod investment ratings. Values are rating points, not physical units.
use anyhow::{Context, Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeaponModRating {
    pub stat_id: u32,
    pub delta: i32,
    pub extra: [u8; 32],
}

pub fn decode_weapon_mod_ratings(data: &[u8]) -> Result<Vec<WeaponModRating>> {
    let read32 = |o: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            data.get(o..o.checked_add(4).context("offset overflow")?)
                .context("truncated u32")?
                .try_into()?,
        ))
    };
    let read64 = |o: usize| -> Result<u64> {
        Ok(u64::from_le_bytes(
            data.get(o..o.checked_add(8).context("offset overflow")?)
                .context("truncated u64")?
                .try_into()?,
        ))
    };
    let components = data
        .chunks_exact(4)
        .enumerate()
        .filter(|(_, b)| *b == 0x80809245u32.to_le_bytes())
        .map(|(i, _)| i * 4)
        .collect::<Vec<_>>();
    ensure!(
        components.len() == 1,
        "expected one rating component, found {}",
        components.len()
    );
    let header = components[0] + 12;
    let count = usize::try_from(read64(header)?)?;
    if count == 0 {
        return Ok(vec![]);
    }
    let pointer = header + 8;
    let start = i64::try_from(pointer)?
        .checked_add(read64(pointer)? as i64)
        .and_then(|p| p.checked_add(16))
        .context("rating pointer overflow")?;
    let start = usize::try_from(start)?;
    let marker = start
        .checked_sub(20)
        .context("invalid rating array marker")?;
    ensure!(
        read32(marker)? == 0x8080BFCD && read32(marker + 12)? == 0x8080924B,
        "wrong rating array class"
    );
    ensure!(
        read64(marker + 4)? == count as u64,
        "rating array count mismatch"
    );
    let end = start
        .checked_add(count.checked_mul(40).context("rating count overflow")?)
        .context("rating range overflow")?;
    let rows = data.get(start..end).context("truncated rating array")?;
    rows.chunks_exact(40)
        .map(|row| {
            Ok(WeaponModRating {
                stat_id: u32::from_le_bytes(row[..4].try_into()?),
                delta: i32::from_le_bytes(row[4..8].try_into()?),
                extra: row[8..].try_into()?,
            })
        })
        .collect()
}
