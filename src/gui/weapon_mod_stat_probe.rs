//! Package probe for Korean Prestige weapon-mod definitions.
//!
//! Kept under `gear::tests` so it can inspect the private catalog join without
//! changing production decoding. Run with `cargo test weapon_mod_stat_probe
//! -- --ignored --nocapture`.

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use quicktag_strings::localized::{LocalizedLanguage, create_stringmap_for_language};
use tiger_pkg::{GameVersion, MarathonVersion, PackageManager, TagHash};

use super::*;

const STAT_GROUP_COMPONENT: u32 = 0x8080_92B2;
const ARRAY_COMPONENT: u32 = 0x8080_BFCD;
const STAT_RATING_ARRAY: u32 = 0x8080_924B;

#[test]
#[ignore = "requires current Marathon packages"]
fn probes_baseline_weapon_examples() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let view = GearView::new(Arc::new(create_stringmap_for_language(
        LocalizedLanguage::English,
    )?));
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = WeaponStatResolver::load(&cache);
    for item in view.items.iter().filter(|item| {
        item.item_type.as_deref() == Some("Weapon")
            && [
                "Repeater HPR",
                "CE Tactical Sidearm",
                "CE Tactical Sidearms",
                "V11 Punch",
                "D54 Battle Pistol",
                "Conquest LMG",
                "M77 Assault Rifle",
                "V75 SCAR",
            ]
            .contains(&item.name.as_str())
    }) {
        if let Some(tag) = item.definition_tag {
            println!("OWNER {} {tag}", item.name);
            println!("BASELINE {:?}", resolver.extract(tag));
            resolver.dump_mod_curve_evidence(tag);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages; diagnostic folding-stock catalog probe"]
fn probes_folding_stock_investment_items() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let view = GearView::new(Arc::new(create_stringmap_for_language(
        LocalizedLanguage::English,
    )?));
    for item in &view.items {
        if item.definition_tag == Some(TagHash(0x80B6_FABE)) || item.name.contains("D54") || item.internal_name.as_deref().is_some_and(|name| {
            name.contains("folding_stock") || name.contains("pistol_light_01")
        }) {
            eprintln!("FOLD_ITEM name={:?} display={} definition={:?} type={:?} rarity={:?} internal={:?}", item.name, item.display_tag, item.definition_tag, item.item_type, item.rarity, item.internal_name);
            if let Some(tag) = item.definition_tag {
                if let Ok(data) = pm.read_tag(tag) {
                    eprintln!("  ratings={:?}", quicktag_core::weapon_mod::decode_weapon_mod_ratings(&data).ok().map(|rows|rows.into_iter().map(|r|(r.stat_id,r.delta)).collect::<Vec<_>>()));
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn reproduces_verified_vanilla_weapon_stats() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = WeaponStatResolver::load(&cache);
    let mut count = 0;
    let mut check = |tag: u32, expected: &[(&str, f32, f32)]| {
        let stats = resolver
            .extract(TagHash(tag))
            .expect("authored weapon baseline");
        for &(name, expected, tolerance) in expected {
            let actual = match name {
                "hip" => stats.hip_fire_spread_degrees,
                "crouch" => stats.crouch_spread_bonus.map(|v| v * 100.0),
                "movement" => stats.movement_accuracy_loss.map(|v| v * 100.0),
                "recoil" => stats.recoil.map(|v| v * 100.0),
                "magazine" => stats.magazine,
                "volt" => stats.volt_drain_percent,
                "rpm" => stats.rounds_per_minute,
                "reload" => stats.reload_seconds,
                "equip" => stats.equip_seconds,
                "ads" => stats.aim_seconds,
                "zoom" => stats.zoom,
                _ => panic!("unknown test metric"),
            }
            .unwrap_or_else(|| panic!("{tag:08X}: missing {name}"));
            assert!(
                (actual - expected).abs() <= tolerance,
                "{tag:08X} {name}: {actual} != {expected}"
            );
            count += 1;
        }
    };
    check(
        0x80B6EE28,
        &[
            ("hip", 2.65, 0.005),
            ("crouch", 60.0, 0.001),
            ("recoil", 57.8, 0.051),
            ("magazine", 9.0, 0.001),
        ],
    );
    check(0x80B6EE56, &[("crouch", 90.0, 0.001), ("movement", 100.0, 0.001)]);
    check(0x80B6EE55, &[("crouch", 90.0, 0.001), ("volt", 4.5, 0.001), ("movement", 100.0, 0.001)]);
    check(
        0x80B6EE68,
        &[("rpm", 1140.0, 0.001), ("reload", 2.69, 0.005)],
    );
    check(0x80B6EE09, &[("rpm", 540.0, 0.001), ("zoom", 1.25, 0.001)]);
    check(0x80B6EDE1, &[("zoom", 1.25, 0.001)]);
    check(
        0x80B6EDE5,
        &[
            ("rpm", 120.0, 0.001),
            ("hip", 1.89, 0.001),
            ("crouch", 80.0, 0.001),
            ("equip", 0.89, 0.0051),
            ("ads", 0.36, 0.005),
            ("recoil", 91.0, 0.001),
            ("volt", 2.5, 0.001),
            ("zoom", 1.25, 0.001),
        ],
    );
    assert_eq!(count, 22);
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn audits_vanilla_weapon_property_coverage() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let view = GearView::new(Arc::new(create_stringmap_for_language(
        LocalizedLanguage::English,
    )?));
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = WeaponStatResolver::load(&cache);
    let mut count = 0;
    for item in view
        .items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon"))
    {
        let Some(tag) = item.definition_tag else {
            continue;
        };
        if quicktag_core::weapon_mod::decode_weapon_mod_ratings(&pm.read_tag(tag)?).is_err() {
            continue;
        }
        let stats = resolver
            .extract(tag)
            .with_context(|| format!("{} {tag}: missing baseline", item.name))?;
        println!("BASELINE-AUDIT {} {tag} {stats:?}", item.name);
        count += 1;
    }
    assert!(count > 20, "unexpected weapon catalog size: {count}");
    println!("BASELINE-AUDIT total={count}");
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn reproduces_seven_additional_weapon_mod_examples() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let view = GearView::new(Arc::new(create_stringmap_for_language(
        LocalizedLanguage::English,
    )?));
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = WeaponStatResolver::load(&cache);
    let metadata =
        pm.read_tag(quicktag_core::implant::ImplantStatResolver::load(&pm)?.semantic_table)?;
    // These labels are test oracles only. Production has no item/name cases.
    // The supplied Rodeo display ID has B7 where the package uses B6.
    let cases: &[(u32, &[(&str, f32, f32)])] = &[
        (
            0x80B6EAB7,
            &[
                ("Hipfire Spread", -4.10, 0.0051),
                ("Recoil", -80.0, 0.051),
                ("Charge Time", -0.55, 0.0051),
            ],
        ),
        (
            0x80B6EB16,
            &[
                ("Rate of Fire", 60.0, 0.501),
                ("Recoil", -13.2, 0.051),
                ("Magazine", 38.0, 0.001),
            ],
        ),
        (
            0x80B6EB58,
            &[
                ("Rate of Fire", 57.0, 0.501),
                ("Recoil", -9.0, 0.051),
                ("Aim Assist", 3.10, 0.0051),
                ("Range", 1.0, 0.501),
                ("Spread Angle", -1.1, 0.051),
            ],
        ),
        (
            0x80B6EB32,
            &[
                ("Firepower", -6.3, 0.051),
                ("Damage", -4.5, 0.051),
                ("Hipfire Spread", -1.36, 0.0051),
                ("Weight", -22.5, 0.051),
            ],
        ),
        (
            0x80B6EAC4,
            &[
                ("Rate of Fire", 300.0, 0.501),
                ("Reload time", -0.36, 0.0051),
                ("Volt Drain", -1.1, 0.051),
            ],
        ),
        (
            0x80B6EB7D,
            &[
                ("Firepower", -8.9, 0.051),
                ("Damage", -5.0, 0.051),
                ("Precision", -0.10, 0.0051),
                ("ADS time", -0.07, 0.0051),
                ("Recoil", -11.6, 0.051),
                ("Range", 10.0, 0.501),
                ("Zoom", 1.1, 0.051),
            ],
        ),
        (
            0x80B6EB4C,
            &[
                ("Hipfire Spread", -0.16, 0.0051),
                ("Aim Assist", 1.0, 0.0051),
                ("Range", 2.0, 0.501),
            ],
        ),
    ];
    let mut verified = 0;
    for &(display, expected) in cases {
        let item = view
            .items
            .iter()
            .find(|i| i.display_tag == TagHash(display))
            .context("missing mod")?;
        let mut matched = false;
        for weapon in view.items.iter().filter(|w| {
            w.item_type.as_deref() == Some("Weapon") && item.compatible_weapons.contains(&w.name)
        }) {
            if let Some(tag) = weapon.definition_tag {
                if let Ok(changes) =
                    resolver.mod_stat_changes(tag, item.definition_tag.unwrap(), &metadata)
                {
                    assert_eq!(
                        changes.len(),
                        expected.len(),
                        "{} on {}: {changes:?}",
                        item.name,
                        weapon.name
                    );
                    for &(name, value, tolerance) in expected {
                        let change = changes
                            .iter()
                            .find(|change| change.name == name)
                            .with_context(|| format!("{} missing {name}", item.name))?;
                        assert!(
                            (change.delta() - value).abs() <= tolerance,
                            "{} {name}: {} vs {value}",
                            item.name,
                            change.delta()
                        );
                        println!(
                            "MATCH {} {name}: {} {}",
                            item.name,
                            change.delta(),
                            change.unit
                        );
                        verified += 1;
                    }
                    matched = true;
                    break;
                }
            }
        }
        assert!(matched, "{} has no evaluated weapon", item.name);
    }
    assert_eq!(verified, 28);
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn reproduces_twelve_labeled_weapon_mod_combinations() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = super::super::weapon_stats::WeaponStatResolver::load(&cache);
    let metadata =
        pm.read_tag(quicktag_core::implant::ImplantStatResolver::load(&pm)?.semantic_table)?;
    let cases: &[(u32, u32, &[(&str, f32, f32)])] = &[
        (0x80B6EE2D, 0x80B6F43F, &[("Reload time", -0.24, 0.0051)]),
        (0x80B6EDE6, 0x80B6F43F, &[("Reload time", -0.42, 0.0051)]),
        (
            0x80B6EDE6,
            0x80B6F54D,
            &[
                ("Recoil", -30.0, 0.051),
                ("Range", 20.0, 0.501),
                ("Magazine", 12.0, 0.001),
            ],
        ),
        (
            0x80B6EEC4,
            0x80B6F54D,
            &[
                ("Recoil", -22.8, 0.051),
                ("Range", 3.0, 0.501),
                ("Magazine", 17.0, 0.001),
            ],
        ),
        (
            0x80B6EDE6,
            0x80B6FAC9,
            &[
                ("ADS spread", -0.05, 0.0051),
                ("ADS time", -0.20, 0.0051),
                ("Range", 30.0, 0.501),
                ("Zoom", 0.5, 0.051),
            ],
        ),
        (
            0x80B6EEC4,
            0x80B6FAC9,
            &[
                ("ADS spread", -0.04, 0.0051),
                ("ADS time", -0.05, 0.0051),
                ("Range", 5.0, 0.501),
                ("Zoom", 0.7, 0.051),
            ],
        ),
        (
            0x80B6EDE6,
            0x80B6FAC1,
            &[
                ("ADS spread", -0.08, 0.0051),
                ("ADS time", -0.30, 0.0051),
                ("Range", 60.0, 0.501),
                ("Zoom", 0.5, 0.051),
            ],
        ),
        (
            0x80B6EEC4,
            0x80B6FAC1,
            &[
                ("ADS spread", -0.06, 0.0051),
                ("ADS time", -0.07, 0.0051),
                ("Range", 10.0, 0.501),
                ("Zoom", 0.7, 0.051),
            ],
        ),
        (0x80B6EDE6, 0x80B6F556, &[("Magazine", 20.0, 0.001)]),
        (0x80B6EEC4, 0x80B6F556, &[("Magazine", 28.0, 0.001)]),
        (
            0x80B6EE89,
            0x80B6F4E2,
            &[("Equip time", -0.45, 0.0051), ("Weight", -16.0, 0.051)],
        ),
        (
            0x80B6EE88,
            0x80B6F4E2,
            &[("Equip time", -0.34, 0.0051), ("Weight", -16.0, 0.051)],
        ),
    ];
    let mut verified = 0;
    for &(weapon, modification, expected) in cases {
        let changes =
            resolver.mod_stat_changes(TagHash(weapon), TagHash(modification), &metadata)?;
        assert_eq!(changes.len(), expected.len(), "unexpected stat changes");
        for &(name, value, tolerance) in expected {
            let change = changes
                .iter()
                .find(|s| s.name == name)
                .context("missing expected metric")?;
            println!(
                "MATCH {weapon:08X}+{modification:08X} {name} rating {} {}->{} value {}->{} delta={} {} label={value}",
                change.rating_id,
                change.base_rating,
                change.modified_rating,
                change.before,
                change.after,
                change.delta(),
                change.unit
            );
            assert!(
                (change.delta() - value).abs() <= tolerance,
                "{weapon:08X} {modification:08X} {name}: {} vs {value}",
                change.delta()
            );
            verified += 1;
        }
    }
    assert_eq!(verified, 30);
    Ok(())
}

#[test]
#[ignore = "requires local Marathon packages"]
fn probes_labeled_weapon_mod_curves() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        r"D:\SteamLibrary\steamapps\common\Marathon\packages",
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let ko = GearView::new_for_language(
        Arc::new(create_stringmap_for_language(LocalizedLanguage::Korean)?),
        LocalizedLanguage::Korean,
    );
    for item in ko.items.iter().filter(|i| {
        [
            "횃불벌레",
            "안정 탄약",
            "광범위 광학 장치",
            "드럼 탄창",
            "신속사격 손잡이",
        ]
        .contains(&i.name.as_str())
    }) {
        if let Some(tag) = item.definition_tag {
            println!(
                "MOD {} {:?} {tag} {:?}",
                item.name,
                item.rarity,
                quicktag_core::weapon_mod::decode_weapon_mod_ratings(&pm.read_tag(tag)?)?
                    .iter()
                    .filter(|r| r.delta != 0)
                    .map(|r| (r.stat_id, r.delta))
                    .collect::<Vec<_>>()
            );
        }
    }
    let en = GearView::new(Arc::new(create_stringmap_for_language(
        LocalizedLanguage::English,
    )?));
    let cache = quicktag_scanner::load_tag_cache();
    let resolver = super::super::weapon_stats::WeaponStatResolver::load(&cache);
    for item in en.items.iter().filter(|i| {
        i.item_type.as_deref() == Some("Weapon")
            && [
                "Twin Tap HBR",
                "Impact H-AR",
                "Bully SMG",
                "Misriah 2442",
                "WSTR Combat Shotgun",
            ]
            .contains(&i.name.as_str())
    }) {
        if let Some(tag) = item.definition_tag {
            println!("WEAPON {} {tag}", item.name);
            resolver.dump_mod_curve_evidence(tag);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn audits_weapon_mod_rating_coverage() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into()),
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let view = GearView::new_for_language(
        Arc::new(create_stringmap_for_language(LocalizedLanguage::English)?),
        LocalizedLanguage::English,
    );
    let mut counts = std::collections::BTreeMap::<String, (usize, usize, usize)>::new();
    let mut rating_counts = std::collections::BTreeMap::<u32, usize>::new();
    for item in view
        .items
        .iter()
        .filter(|i| i.item_type.as_deref() == Some("Weapon Mod"))
    {
        let tag = item.definition_tag.context("mod missing definition")?;
        let rows = quicktag_core::weapon_mod::decode_weapon_mod_ratings(&pm.read_tag(tag)?)?;
        let nonzero = rows.iter().filter(|r| r.delta != 0).collect::<Vec<_>>();
        for rating in &nonzero {
            *rating_counts.entry(rating.stat_id).or_default() += 1;
        }
        let entry = counts
            .entry(item.mod_category.clone().unwrap_or_default())
            .or_default();
        entry.0 += 1;
        entry.1 += usize::from(!nonzero.is_empty());
        entry.2 += usize::from(rows.iter().any(|r| r.extra != [0; 32]));
        if item.mod_category.as_deref() == Some("Chip") && !nonzero.is_empty() {
            println!(
                "CHIP {} {tag} {:?}",
                item.name,
                nonzero
                    .iter()
                    .map(|r| (r.stat_id, r.delta))
                    .collect::<Vec<_>>()
            );
        }
    }
    println!("COVERAGE category=(total,nonzero,nonzero_tail) {counts:?}");
    println!("RATING_COUNTS {rating_counts:?}");
    for (tag, expected) in [
        (0x80B6FAE5, vec![(15, 20)]),
        (0x80B6F523, vec![(18, 90), (19, 50)]),
        (0x80B6F590, vec![(16, 15), (17, 30), (25, 30)]),
        (0x80B6F576, vec![(12, -10), (17, 90), (21, 90)]),
    ] {
        let rows =
            quicktag_core::weapon_mod::decode_weapon_mod_ratings(&pm.read_tag(TagHash(tag))?)?;
        assert_eq!(
            rows.iter()
                .filter(|r| r.delta != 0)
                .map(|r| (r.stat_id, r.delta))
                .collect::<Vec<_>>(),
            expected
        );
    }
    for item in view.items.iter().filter(|i| {
        i.item_type.as_deref() == Some("Weapon")
            && ["Conquest", "Retaliator", "Copperhead", "V22"]
                .iter()
                .any(|n| i.name.contains(n))
    }) {
        println!(
            "WEAPON {} {:?}",
            item.name,
            item.definition_tag.map(|t| format!("{t}"))
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires current Marathon packages"]
fn gear_exposes_every_weapon_mod_rating_record() -> Result<()> {
    let pm = Arc::new(PackageManager::new(
        std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into()),
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();
    let mut view = GearView::new_for_language(
        Arc::new(create_stringmap_for_language(LocalizedLanguage::English)?),
        LocalizedLanguage::English,
    );
    view.reconcile_weapon_skin_models(&quicktag_scanner::load_tag_cache());

    let modifications = view
        .items
        .iter()
        .filter(|item| item.item_type.as_deref() == Some("Weapon Mod"))
        .collect::<Vec<_>>();
    assert_eq!(modifications.len(), 515);
    assert_eq!(view.weapon_mod_stats.len(), modifications.len());
    for item in modifications {
        let details = view
            .weapon_mod_stats
            .get(&item.display_tag)
            .with_context(|| format!("missing GUI stats for {}", item.name))?;
        let expected = quicktag_core::weapon_mod::decode_weapon_mod_ratings(
            &pm.read_tag(item.definition_tag.context("mod missing definition")?)?,
        )?
        .into_iter()
        .filter(|rating| rating.delta != 0)
        .map(|rating| (rating.stat_id, rating.delta))
        .collect::<Vec<_>>();
        let actual = details
            .ratings
            .iter()
            .map(|rating| (rating.rating_id, rating.value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{}", item.name);
        assert!(
            details
                .weapons
                .iter()
                .flat_map(|weapon| &weapon.changes)
                .all(|change| actual.iter().any(|(rating, _)| *rating == change.rating_id))
        );
    }
    Ok(())
}

fn u32_at(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn word_hits(data: &[u8], needle: u32) -> Vec<usize> {
    data.chunks_exact(4)
        .enumerate()
        .filter_map(|(index, bytes)| {
            (u32::from_le_bytes(bytes.try_into().ok()?) == needle).then_some(index * 4)
        })
        .collect()
}

fn nearby_words(data: &[u8], offset: usize) -> Vec<(usize, u32)> {
    let start = offset.saturating_sub(0x10) & !3;
    let end = (offset + 0x30).min(data.len() & !3);
    (start..end)
        .step_by(4)
        .filter_map(|at| Some((at, u32_at(data, at)?)))
        .collect()
}

fn array_headers(data: &[u8]) -> Vec<(usize, usize, u32, usize)> {
    data.chunks_exact(4)
        .enumerate()
        .filter_map(|(index, bytes)| {
            let offset = index * 4;
            if u32::from_le_bytes(bytes.try_into().ok()?) != ARRAY_COMPONENT {
                return None;
            }
            let count =
                u64::from_le_bytes(data.get(offset + 4..offset + 12)?.try_into().ok()?) as usize;
            let class = u32_at(data, offset + 12)?;
            Some((offset, count, class, offset + 20))
        })
        .collect()
}

fn rating_rows(data: &[u8]) -> Vec<(usize, u32, i32)> {
    array_headers(data)
        .into_iter()
        .filter(|(_, _, class, _)| *class == STAT_RATING_ARRAY)
        .flat_map(|(_, count, _, start)| {
            (0..count).filter_map(move |index| {
                let row = start + index * 0x28;
                Some((row, u32_at(data, row)?, u32_at(data, row + 4)? as i32))
            })
        })
        .collect()
}

#[test]
#[ignore = "requires current Marathon packages; diagnostic weapon-mod stat probe"]
fn probes_korean_prestige_weapon_mod_stat_definitions() -> Result<()> {
    let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages"));
    let pm = Arc::new(PackageManager::new(
        packages,
        GameVersion::Marathon(MarathonVersion::Marathon),
        None,
    )?);
    tiger_pkg::initialize_package_manager(&pm);
    quicktag_core::classes::initialize_reference_names();

    let strings = Arc::new(create_stringmap_for_language(LocalizedLanguage::Korean)?);
    let view = GearView::new_for_language(strings, LocalizedLanguage::Korean);
    let resolver = quicktag_core::implant::ImplantStatResolver::load(&pm)?;

    let targets = [
        ("회로 보호막", "Retaliator LMG", "Shield"),
        ("무한 벨트", "Conquest LMG", "Magazine"),
        ("과충전 렌즈", "V22 Volt Thrower", "Muzzle"),
        ("3중 총열", "Copperhead RF", "Muzzle"),
    ];

    for (name, weapon, slot) in targets {
        let matches = view
            .items
            .iter()
            .filter(|item| item.name == name)
            .collect::<Vec<_>>();
        eprintln!(
            "TARGET name={name:?} expected_weapon={weapon:?} expected_slot={slot:?} matches={}",
            matches.len()
        );
        assert!(!matches.is_empty(), "missing Korean target {name}");

        for item in matches {
            let Some(definition_tag) = item.definition_tag else {
                eprintln!("  item display={} has no definition", item.display_tag);
                continue;
            };
            let data = pm.read_tag(definition_tag)?;
            let stats = match resolver.resolve(&data) {
                Ok(decoded) => format!(
                    "ok bindings={:?} stats={:?}",
                    decoded.bindings,
                    decoded
                        .stats
                        .iter()
                        .map(|stat| (stat.semantic_id, stat.raw_name_hash, stat.value))
                        .collect::<Vec<_>>()
                ),
                Err(error) => format!("ERR {error:#}"),
            };
            let stat_components = word_hits(&data, STAT_GROUP_COMPONENT);
            let array_components = word_hits(&data, ARRAY_COMPONENT);
            eprintln!(
                "  display={} definition={} size=0x{:X} rarity={:?} type={:?} subcategory={:?} mod_category={:?} family={:?} internal_hash={:?} internal_name={:?} compatible={:?}",
                item.display_tag,
                definition_tag,
                data.len(),
                item.rarity,
                item.item_type,
                item.subcategory,
                item.mod_category,
                item.mod_family,
                item.internal_hash.map(|hash| format!("{hash:08X}")),
                item.internal_name,
                item.compatible_weapons,
            );
            eprintln!(
                "    stat_group_components={stat_components:?} arrays={array_components:?} resolver={stats}"
            );
            for offset in stat_components
                .iter()
                .chain(array_components.iter())
                .take(4)
            {
                eprintln!("    words@0x{offset:X}={:?}", nearby_words(&data, *offset));
            }
            eprintln!(
                "    array_headers={:?}",
                array_headers(&data)
                    .into_iter()
                    .map(|(offset, count, class, start)| {
                        let words = (0..count.min(8))
                            .filter_map(|index| u32_at(&data, start + index * 4))
                            .collect::<Vec<_>>();
                        (
                            format!("0x{offset:X}"),
                            count,
                            format!("0x{class:08X}"),
                            words,
                        )
                    })
                    .collect::<Vec<_>>()
            );
            eprintln!("    stat_rating_rows={:?}", rating_rows(&data));
        }
    }

    for prefix in [
        "weapon_mods.shields.lmg.",
        "weapon_mods.magazines.lmg.",
        "weapon_mods.muzzles.dampener.",
        "weapon_mods.muzzles.base.",
    ] {
        eprintln!("FAMILY prefix={prefix}");
        for item in view.items.iter().filter(|item| {
            item.internal_name
                .as_deref()
                .is_some_and(|path| path.starts_with(prefix))
        }) {
            eprintln!(
                "  {} display={} definition={:?} rarity={:?} category={:?} family={:?} compatible={:?}",
                item.name,
                item.display_tag,
                item.definition_tag,
                item.rarity,
                item.mod_category,
                item.mod_family,
                item.compatible_weapons,
            );
        }
    }
    Ok(())
}
