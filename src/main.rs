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
                    backends: wgpu::Backends::PRIMARY,
                    ..Default::default()
                },
                device_descriptor: Arc::new(|_adapter| {
                    let mut required_limits = wgpu::Limits::default();
                    // The material ABI binds sixteen 2D inputs plus a local
                    // coating cube; the scene contributes one more cube.
                    required_limits.max_sampled_textures_per_shader_stage = 18;
                    wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D
                            | wgpu::Features::TEXTURE_BINDING_ARRAY
                            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                        required_limits,
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
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|error| format!("Could not find a GPU adapter: {error}"))?;
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_sampled_textures_per_shader_stage = 18;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        required_limits,
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
