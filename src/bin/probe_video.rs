use std::{path::PathBuf, sync::Arc};

use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

fn main() {
    let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages"));
    let manager = Arc::new(
        PackageManager::new(
            packages,
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .unwrap(),
    );
    tiger_pkg::initialize_package_manager(&manager);
    let videos = manager.get_all_by_type(27, Some(1));
    println!("videos={}", videos.len());
    for (tag, entry) in videos.iter().rev().take(10) {
        let data = manager.read_tag(*tag).unwrap();
        println!(
            "{tag} pkg={} size={} ref={:08X} head={:02X?}",
            tag.pkg_id(),
            data.len(),
            entry.reference,
            &data[..data.len().min(64)]
        );
        let mut offset = 0usize;
        let mut seen = std::collections::BTreeSet::new();
        while offset + 0x20 <= data.len() {
            let kind = &data[offset..offset + 4];
            let size =
                u32::from_be_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
            let payload_offset = data[offset + 9] as usize;
            let padding =
                u16::from_be_bytes(data[offset + 10..offset + 12].try_into().unwrap()) as usize;
            let payload_kind = data[offset + 15] & 3;
            let payload_start = offset + 8 + payload_offset;
            let payload_len = size.saturating_sub(payload_offset + padding);
            if payload_kind == 0
                && (kind == b"@SFV" || kind == b"@SFA")
                && seen.insert(kind.to_vec())
            {
                println!(
                    "  {} stream len={} head={:02X?}",
                    String::from_utf8_lossy(kind),
                    payload_len,
                    &data[payload_start..(payload_start + payload_len.min(64))]
                );
            }
            let next = offset + 8 + size;
            if next <= offset || next > data.len() {
                break;
            }
            offset = next;
        }
    }
}
