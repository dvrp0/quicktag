mod asset_cache;
mod geometry;
mod gui;
mod material;
mod panic_handler;
mod render;
mod texture;
mod util;

use std::sync::Arc;

use clap::Parser;
use eframe::egui_wgpu::WgpuConfiguration;
use eframe::wgpu;
use eframe::{egui::ViewportBuilder, egui_wgpu::WgpuSetupCreateNew};
use env_logger::Env;
use game_detector::InstalledGame;
use log::info;
use tiger_pkg::{DestinyVersion, GameVersion, PackageManager, Version, package_manager};

use crate::gui::QuickTagApp;

#[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None, disable_version_flag(true))]
struct Args {
    /// Path to packages directory
    packages_path: Option<String>,

    /// Game version for the specified packages directory
    #[arg(short, value_enum)]
    version: Option<GameVersion>,

    /// Export all resolved Head/Torso/Leg implant icons and exit
    #[arg(long)]
    export_implant_icons: bool,

    /// Output directory for --export-implant-icons
    #[arg(long, default_value = "./implant_icons")]
    implant_icons_output: std::path::PathBuf,

    /// Decode one texture tag (hex) to a PNG at --render-output and exit
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    export_texture: Option<tiger_pkg::TagHash>,

    /// Write one shader tag's compiled payload to --render-output and exit
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    export_shader: Option<tiger_pkg::TagHash>,

    /// Print one technique's stages, TFX bytecode and evaluated constants and exit
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    dump_technique: Option<tiger_pkg::TagHash>,

    /// TFX time in seconds for --render-model and --dump-technique
    #[arg(long, default_value_t = 0.0)]
    render_time: f32,

    /// Render one model tag (hex, e.g. 80B15334) to a PNG and exit
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    render_model: Option<tiger_pkg::TagHash>,

    /// Output file for --render-model
    #[arg(long, default_value = "./model.png")]
    render_output: std::path::PathBuf,

    /// Camera yaw in degrees for --render-model
    #[arg(long, default_value_t = -24.0, allow_negative_numbers = true)]
    render_yaw: f32,

    /// Camera pitch in degrees for --render-model
    #[arg(long, default_value_t = 14.0, allow_negative_numbers = true)]
    render_pitch: f32,

    /// Key light, sky ambient and ground ambient colours for --render-model,
    /// as nine linear values R,G,B,R,G,B,R,G,B
    #[arg(long, value_delimiter = ',', default_values_t = [1.0; 9])]
    render_lighting: Vec<f32>,

    /// Output of --render-model: final, albedo, normals, properties (metal,
    /// AO, roughness) or emissive
    #[arg(long, default_value = "final")]
    render_view: String,

    /// Magnification of the fitted frame for --render-model
    #[arg(long, default_value_t = 1.0)]
    render_zoom: f32,

    /// Point of the fitted frame to centre on, as X,Y in -1..1 (Y up)
    #[arg(long, value_delimiter = ',', default_values_t = [0.0, 0.0], allow_negative_numbers = true)]
    render_focus: Vec<f32>,
}

fn parse_tag(value: &str) -> Result<tiger_pkg::TagHash, std::num::ParseIntError> {
    u32::from_str_radix(value.trim_start_matches("0x"), 16).map(tiger_pkg::TagHash)
}

fn main() -> eframe::Result<()> {
    panic_handler::install_hook(None);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let _rt_guard = rt.enter();

    env_logger::Builder::from_env(
        Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=error,naga=warn"),
    )
    .init();
    let args = Args::parse();

    let packages_path = if let Some(packages_path) = args.packages_path {
        packages_path
    } else if let Some(path) = find_d2_packages_path() {
        let mut path = std::path::PathBuf::from(path);
        path.push("packages");
        path.to_str().unwrap().to_string()
    } else {
        panic!("Could not find Destiny 2 packages directory");
    };

    info!(
        "Initializing package manager for version {:?} at '{}'",
        args.version, packages_path
    );
    let pm = PackageManager::new(
        packages_path,
        args.version
            .unwrap_or(GameVersion::Destiny(DestinyVersion::Destiny2TheEdgeOfFate)),
        None,
    )
    .unwrap();

    tiger_pkg::initialize_package_manager(&Arc::new(pm));

    quicktag_core::classes::initialize_reference_names();

    if args.export_implant_icons {
        let render_state = create_headless_render_state()
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        let exported = crate::gui::GearView::export_implant_icons(
            &args.implant_icons_output,
            &render_state,
        )
        .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        println!(
            "Exported {exported} implant icons to {}",
            args.implant_icons_output.display()
        );
        return Ok(());
    }

    if let Some(tag) = args.export_shader {
        let payload = package_manager()
            .get_entry(tag)
            .ok_or_else(|| std::io::Error::other("missing shader tag"))
            .and_then(|entry| {
                package_manager()
                    .read_tag(tiger_pkg::TagHash(entry.reference))
                    .map_err(std::io::Error::other)
            })
            .and_then(|payload| std::fs::write(&args.render_output, payload))
            .map_err(|error| eframe::Error::AppCreation(Box::new(error)));
        payload?;
        println!("Exported shader {tag} to {}", args.render_output.display());
        return Ok(());
    }

    if let Some(tag) = args.dump_technique {
        let technique = crate::render::technique::TechniqueDescriptor::load(tag)
            .ok_or_else(|| eframe::Error::AppCreation(Box::new(std::io::Error::other("not a technique"))))?;
        let mut inputs = crate::render::tfx::TfxRuntimeInputs::default();
        inputs.apply_marathon_global_defaults();
        inputs.time_seconds = args.render_time;
        for stage in &technique.stages {
            println!(
                "STAGE {:?} shader={:?} cb_slot={:?} constants={} inline_rows={}",
                stage.stage,
                stage.shader,
                stage.constant_buffer_slot,
                stage.constants.len(),
                stage.inline_constants.len()
            );
            for (index, constant) in stage.constants.iter().enumerate() {
                println!("  CONST {index} = {constant:?}");
            }
            for resource in &stage.resources {
                println!("  TEXTURE slot={} {:?}", resource.slot, resource.resolved);
            }
            let execution = stage.execute(&inputs);
            for step in &execution.trace {
                println!("  OP {:04} {:02X} {} {}", step.byte_offset, step.opcode, step.operation, step.detail);
            }
            for (target, value) in &execution.outputs {
                println!("  OUT {target} = {value:?}");
            }
            for (row, value) in stage.runtime_state(&inputs).constant_registers.iter().enumerate() {
                println!("  ROW {row} = {value:?}");
            }
        }
        return Ok(());
    }

    if let Some(tag) = args.export_texture {
        let render_state = create_headless_render_state()
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        let image = crate::texture::Texture::load(&render_state, tag, false)
            .and_then(|texture| texture.to_image(&render_state, 0))
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        image
            .save(&args.render_output)
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        println!("Exported {tag} ({}x{}) to {}", image.width(), image.height(), args.render_output.display());
        return Ok(());
    }

    if let Some(tag) = args.render_model {
        let render_state = create_headless_render_state()
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        crate::gui::ModelsView::export_model_image(
            &render_state,
            tag,
            args.render_yaw,
            args.render_pitch,
            args.render_zoom,
            [args.render_focus[0], *args.render_focus.get(1).unwrap_or(&0.0)],
            args.render_time,
            std::array::from_fn(|index| args.render_lighting.get(index).copied().unwrap_or(1.0)),
            &args.render_view,
            &args.render_output,
        )
        .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        println!("Rendered {tag} to {}", args.render_output.display());
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: ViewportBuilder::default()
            .with_title(format!(
                "Quicktag - {} ({:?})",
                package_manager().version.name(),
                package_manager().platform
            ))
            .with_icon(
                eframe::icon_data::from_png_bytes(include_bytes!("../quicktag.png"))
                    .expect("Failed to load icon"),
            ),
        persist_window: true,
        wgpu_options: WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN,
                    ..Default::default()
                },
                device_descriptor: Arc::new(|_adapter| {
                    let mut required_limits = wgpu::Limits::default();
                    // The material ABI binds sixteen 2D inputs plus a local
                    // coating cube; the scene contributes one more cube.
                    required_limits.max_sampled_textures_per_shader_stage = 18;
                    required_limits.max_storage_buffers_per_shader_stage = 12;
                    required_limits.max_color_attachment_bytes_per_sample = 64;
                    wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                            | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
                            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D
                            | wgpu::Features::TEXTURE_BINDING_ARRAY
                            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
                        required_limits,
                        // SAFETY: authored modules are embedded, hash-pinned and
                        // validated before entering Vulkan passthrough.
                        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                        ..Default::default()
                    }
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "Quicktag",
        native_options,
        Box::new(|cc| Ok(Box::new(QuickTagApp::new(cc)))),
    )
}

fn create_headless_render_state() -> Result<eframe::egui_wgpu::RenderState, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|error| format!("Could not find a GPU adapter: {error}"))?;
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_sampled_textures_per_shader_stage = 18;
    required_limits.max_storage_buffers_per_shader_stage = 12;
    required_limits.max_color_attachment_bytes_per_sample = 64;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
        required_limits,
        // SAFETY: same validated authored-program contract as the UI device.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .map_err(|error| format!("Could not create a GPU device: {error}"))?;
    let target_format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = eframe::egui_wgpu::Renderer::new(
        &device,
        target_format,
        eframe::egui_wgpu::RendererOptions::default(),
    );
    Ok(eframe::egui_wgpu::RenderState {
        adapter,
        available_adapters: vec![],
        device,
        queue,
        target_format,
        renderer: Arc::new(eframe::egui::mutex::RwLock::new(renderer)),
    })
}

fn find_d2_packages_path() -> Option<String> {
    let mut installations = game_detector::find_all_games();
    installations.retain(|i| match i {
        InstalledGame::Steam(a) => a.appid == 1085660,
        InstalledGame::EpicGames(m) => m.display_name == "Destiny 2",
        InstalledGame::MicrosoftStore(p) => p.app_name == "Destiny2PCbasegame",
        _ => false,
    });

    info!("Found {} Destiny 2 installations", installations.len());

    // Sort installations, weighting Steam > Epic > Microsoft Store
    installations.sort_by_cached_key(|i| match i {
        InstalledGame::Steam(_) => 0,
        InstalledGame::EpicGames(_) => 1,
        InstalledGame::MicrosoftStore(_) => 2,
        _ => 3,
    });

    match installations.first() {
        Some(InstalledGame::Steam(a)) => Some(a.game_path.clone()),
        Some(InstalledGame::EpicGames(m)) => Some(m.install_location.clone()),
        Some(InstalledGame::MicrosoftStore(p)) => Some(p.path.clone()),
        _ => None,
    }
}
