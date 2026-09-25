use super::*;
use crate::geometry::{WireframeMaterialRange, WireframeMaterialTextures};

fn panel(scale: f32, offset: [f32; 3]) -> WireframePreview {
    let positions = [
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5],
    ];
    let point = |p: [f32; 3]| std::array::from_fn(|axis| p[axis] * scale + offset[axis]);
    WireframePreview {
        source: "lighting scale test".into(),
        position_format: "f32x3",
        uv_format: None,
        vertices: positions.into_iter().map(point).collect(),
        rigid_indices: None,
        normals: Some(vec![[0.0, 1.0, 0.0]; 4]),
        procedural_positions: None,
        procedural_normals: None,
        tangents: None,
        uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        normal_format: None,
        tangent_format: None,
        indices: vec![0, 1, 2, 0, 2, 3],
        material_ranges: vec![WireframeMaterialRange {
            index_start: 0,
            index_count: 6,
            raw_lod_category: Some(0),
            render_stage: Some(0),
            technique: None,
            gear_dye_change_color_index: None,
            authored_source: None,
            procedural_scale: 1.0,
            texture: None,
            textures: WireframeMaterialTextures {
                solid_color: Some([0.4, 0.3, 0.2, 1.0]),
                solid_surface: Some([0.7, 0.0]),
                ..Default::default()
            },
        }],
        authored_inputs: vec![],
        authored_shadow_ranges: vec![],
        min: point([-0.5, 0.0, -0.5]),
        max: point([0.5, 0.0, 0.5]),
        vertex_count_total: 4,
        index_count_total: 6,
    }
}

#[test]
fn lighting_tracks_base_bounds_and_gizmo_round_trips() {
    let environment = ModelEnvironment::default();
    let source = light_source_position(&environment);
    for scale in [0.125, 1.0, 8.0, 64.0] {
        let offset = [4.0, -8.0, 2.0];
        let mesh = panel(scale, offset);
        let transform = ModelLightTransform::new(&environment, &mesh);
        assert_eq!(transform.scale, scale);
        assert_eq!(transform.center, offset);
        let actual = transform.to_model(source);
        let returned = transform.to_rig(actual);
        for axis in 0..3 {
            assert!((returned[axis] - source[axis]).abs() < 0.00001);
        }
    }
    let mut longer = panel(1.0, [0.0; 3]);
    longer.max[0] = 2.5; // Longer attachment, without changing the camera frame.
    let transform = ModelLightTransform::new(&environment, &longer);
    assert_eq!(transform.scale, 3.0);
    assert_eq!(transform.center, [1.0, 0.0, 0.0]);
    let locked = ModelLightTransform::new(
        &ModelEnvironment {
            light_model_frame: Some(ModelCameraFrame::from_wireframe(&panel(1.0, [0.0; 3]))),
            ..environment
        },
        &longer,
    );
    assert_eq!(locked.scale, 1.0);
    assert_eq!(locked.center, [0.0; 3]);
    let fixed = ModelLightTransform::new(
        &ModelEnvironment {
            light_scale_with_model: false,
            ..environment
        },
        &longer,
    );
    assert_eq!(fixed.to_model(source), source);
    assert_eq!(fixed.scale, 1.0);
}

#[test]
fn uniformly_scaled_models_preserve_gpu_lighting() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("GPU adapter");
    let mut limits = wgpu::Limits::default();
    limits.max_sampled_textures_per_shader_stage = 18;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .expect("GPU device");
    let target_format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = eframe::egui_wgpu::Renderer::new(&device, target_format, Default::default());
    let state = eframe::egui_wgpu::RenderState {
        adapter,
        available_adapters: vec![],
        device,
        queue,
        target_format,
        renderer: Arc::new(eframe::egui::mutex::RwLock::new(renderer)),
    };
    let textures = TextureCache::new(state.clone());
    let environment = ModelEnvironment {
        light_orbit_center: [0.0; 3],
        light_target: [0.0; 3],
        light_orbit_position: [0.0, 1.0, 0.0],
        light_orbit_radius: 2.0,
        light_range: 4.0,
        sun_intensity: 2.0,
        ambient_intensity: 0.0,
        specular_ibl_intensity: 0.0,
        ssao_strength: 0.0,
        vertex_ao_strength: 0.0,
        tone_mapping: false,
        fxaa: false,
        brightness: 1.0,
        ..ModelEnvironment::default()
    };
    let render = |scale, offset| {
        let mesh = panel(scale, offset);
        let mut gpu = GpuModelPreview::create(&state.device, &mesh, None).expect("panel GPU");
        for draw in &mut gpu.draws {
            draw.pipeline.rasterizer = 1; // Two-sided fixture isolates lighting from winding.
        }
        let gpu = Arc::new(gpu);
        let callback = ModelPaintCallback::new(
            gpu,
            &textures,
            &mesh,
            None,
            None,
            0.0,
            0.0,
            1.0,
            egui::Vec2::ZERO,
            false,
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(256.0, 256.0)),
            1.0,
            environment,
        );
        assert_eq!(callback.scene.light_position[3], scale);
        assert_eq!(callback.scene.light_parameters[1], 4.0 * scale);
        let bytes = callback
            .export_image_bytes(&state, [256, 256], image::ImageFormat::Png)
            .expect("render panel");
        image::load_from_memory(&bytes).expect("PNG").to_rgba8()
    };
    let baseline = render(1.0, [0.0; 3]);
    assert!(
        baseline.pixels().filter(|p| p[3] != 0).count() > 10000,
        "empty panel capture"
    );
    assert!(
        baseline.get_pixel(128, 128)[0] > 20,
        "panel must receive direct lighting"
    );
    for scale in [0.125, 8.0] {
        let actual = render(scale, [4.0, -8.0, 2.0]);
        let max_delta = baseline
            .as_raw()
            .iter()
            .zip(actual.as_raw())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(
            max_delta <= 1,
            "scale {scale} changed lighting by {max_delta} levels"
        );
    }
    std::mem::forget(textures);
    std::mem::forget(state);
}
