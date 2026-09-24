use super::*;

#[test]
#[ignore = "requires installed Marathon packages"]
fn all_runner_shells_preserve_authored_geometry() {
    super::super::tests::init_goliath_test_package_manager();
    let cache = Arc::new(quicktag_scanner::load_tag_cache());
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/runner-shells.json")).unwrap();
    let mut report = vec![];
    for fixture in fixtures.as_array().unwrap() {
        let tag = TagHash(u32::from_str_radix(fixture["pattern"].as_str().unwrap(), 16).unwrap());
        let shell = RunnerShellAssembly::resolve(&cache, tag).expect("authored shell root");
        assert_eq!(shell.pattern, tag, "{}", fixture["name"]);
        let mut actual = shell
            .geometry()
            .iter()
            .map(ToString::to_string)
            .collect_vec();
        actual.sort();
        let expected = fixture["geometry"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tag| tag.as_str().unwrap().to_owned())
            .collect_vec();
        assert_eq!(
            actual, expected,
            "{} {tag}: geometry ownership changed",
            fixture["name"]
        );
        let preview = shell.load(cache.clone()).expect("assembled preview");
        let GeometryPreviewKind::Model(model) = preview.kind else {
            panic!("shell model")
        };
        let wireframe = model.wireframe.expect("shell mesh");
        assert!(!wireframe.indices.is_empty(), "{tag}: empty shell");
        let mut start = 0;
        for geometry in shell.geometry() {
            let entry = package_manager().get_entry(geometry).unwrap();
            let (_, authored) = parse_model_wireframe(geometry, &entry).expect("authored submesh");
            let end = start + authored.vertices.len();
            assert_eq!(
                &wireframe.vertices[start..end],
                &authored.vertices,
                "{tag}: submesh {geometry} moved during assembly"
            );
            start = end;
        }
        assert_eq!(
            start,
            wireframe.vertices.len(),
            "{tag}: unexpected vertices"
        );
        report.push(serde_json::json!({"pattern":tag.to_string(),"name":fixture["name"],"shell":fixture["shell"],"geometry":actual,"vertices":start,"indices":wireframe.indices.len(),"min":wireframe.min,"max":wireframe.max}));
        eprintln!(
            "shell {tag} {}: {} parts, {start} vertices",
            fixture["name"],
            shell.parts.len()
        );
    }
    std::fs::write(
        "target/runner-shell-audit.json",
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
