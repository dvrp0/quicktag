#[test]
#[ignore = "requires installed Marathon packages"]
fn every_runner_skin_uses_its_authored_pattern() {
    use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};
    let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
        .unwrap_or_else(|_| r"D:\SteamLibrary\steamapps\common\Marathon\packages".into());
    let manager = Arc::new(
        PackageManager::new(
            packages,
            GameVersion::Marathon(MarathonVersion::Marathon),
            None,
        )
        .unwrap(),
    );
    tiger_pkg::initialize_package_manager(&manager);
    quicktag_core::classes::initialize_reference_names();
    let cache = quicktag_scanner::load_tag_cache();
    let strings = Arc::new(quicktag_strings::localized::create_stringmap().unwrap());
    let mut gear = super::super::gear::GearView::new(strings);
    gear.reconcile_weapon_skin_models(&cache);
    let catalog = gear.model_weapon_catalog();
    let models = runner_model_index(&cache, &catalog);
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/runner-shells.json")).unwrap();
    let expected = fixtures
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["pattern"].as_str().unwrap().to_owned())
        .sorted()
        .collect_vec();
    let actual = catalog
        .runner_skins
        .iter()
        .map(|skin| skin.model_tag.to_string())
        .sorted()
        .collect_vec();
    assert_eq!(
        actual, expected,
        "Runner catalog changed; audit new authored roots"
    );
    assert_eq!(
        models.len(),
        catalog.runner_skins.len(),
        "missing or aliased shell roots"
    );
    for skin in &catalog.runner_skins {
        assert_eq!(
            package_manager()
                .get_entry(skin.model_tag)
                .unwrap()
                .reference,
            0x8080BAAD
        );
        let model = &models[&skin.model_tag];
        assert_eq!(model.pattern, skin.model_tag);
        assert!(is_model_catalog_reference(0x8080BAAD));
        assert!(!is_model_catalog_reference(0x8080BADB));
        assert_eq!(
            runner_skin_for_model(&catalog, skin.model_tag)
                .unwrap()
                .name,
            skin.name
        );
        for nested in &model.nested_patterns {
            assert!(
                !models.contains_key(nested),
                "{} absorbed another skin root {nested}",
                skin.name
            );
            assert_eq!(
                runner_model_root(&models, *nested),
                Some(skin.model_tag),
                "{} nested Pattern does not route to its shell root",
                skin.name
            );
        }
    }
}
