#[test]
#[ignore = "requires installed Marathon packages and GPU"]
fn sniper_barrels_keep_base_model_lighting() {
    use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};
    let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
        .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into());
    let manager = Arc::new(
        PackageManager::new(
            packages,
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .expect("packages"),
    );
    tiger_pkg::initialize_package_manager(&manager);
    quicktag_core::classes::initialize_reference_names();
    let graph = Arc::new(quicktag_scanner::load_tag_cache());
    let strings = Arc::new(quicktag_strings::localized::create_stringmap().expect("strings"));
    let mut gear = super::super::gear::GearView::new(strings);
    gear.reconcile_weapon_skin_models(&graph);
    let catalog = gear.model_weapon_catalog();
    let target = TagHash(0x80AA0AC7);
    let owner = weapon_index_for_model(&catalog, target).expect("sniper owner");
    let weapon = &catalog.weapons[owner];
    eprintln!(
        "sniper={} owner={} socket={:?}",
        weapon.name, weapon.owner_tag, weapon.socket_owner
    );
    let barrels = weapon
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.name.to_ascii_lowercase().contains("barrel"))
        .flat_map(|(slot_index, slot)| {
            slot.mods
                .iter()
                .enumerate()
                .map(move |(mod_index, item)| (slot_index, mod_index, item.clone()))
        })
        .unique_by(|(_, _, item)| item.model_tag)
        .collect_vec();
    assert!(barrels.len() >= 2, "need multiple barrel variants");
    let state = crate::create_headless_render_state().expect("GPU");
    let mut view = ModelsView::new(graph, TextureCache::new(state));
    view.set_weapon_catalog(catalog);
    view.load_model(target);
    let base_frame = view
        .preview_environment
        .light_model_frame
        .expect("default-mod lighting frame");
    let base_transform = ModelLightTransform::new(
        &view.preview_environment,
        view.preview.as_ref().unwrap().wireframe().unwrap(),
    );
    let source = base_transform.to_model(light_source_position(&view.preview_environment));
    let mut lengths = vec![];
    for (slot, index, item) in barrels {
        view.selected_mods[slot] = Some(index);
        view.selected_mod_unique_ids[slot] = Some(0.5);
        view.rebuild_model_preview();
        let wireframe = view.preview.as_ref().unwrap().wireframe().unwrap();
        let assembled = ModelCameraFrame::from_wireframe(wireframe);
        lengths.push(assembled.radius.to_bits());
        let transform = ModelLightTransform::new(&view.preview_environment, wireframe);
        assert_eq!(view.preview_environment.light_model_frame, Some(base_frame));
        assert_eq!(transform.scale, base_transform.scale);
        assert_eq!(transform.center, base_transform.center);
        assert_eq!(
            transform.to_model(light_source_position(&view.preview_environment)),
            source
        );
        eprintln!(
            "barrel={} tag={} rarity={} assembled_extent={} lighting_extent={} center={:?}",
            item.name,
            item.model_tag,
            item.rarity_code,
            assembled.radius,
            transform.scale,
            transform.center
        );
    }
    lengths.sort_unstable();
    lengths.dedup();
    assert!(
        lengths.len() >= 2,
        "barrel variants must exercise differing geometry bounds"
    );
    std::mem::forget(view);
}
