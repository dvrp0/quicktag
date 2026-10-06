mod animation;
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

    /// Print every runner and weapon skin with its model root tag and exit
    #[arg(long)]
    list_skins: bool,

    /// Print decoded stat inputs of weapons/mods whose name contains one of these (comma separated) and exit
    #[arg(long, value_delimiter = ',')]
    dump_weapon_stats: Vec<String>,

    /// Weapon mod definition tags (hex, comma separated) applied by --dump-weapon-stats
    #[arg(long, value_delimiter = ',', value_parser = parse_tag)]
    dump_weapon_mods: Vec<tiger_pkg::TagHash>,

    /// Match --dump-weapon-stats names in Korean instead of English; also the language of --export-gear-json
    #[arg(long)]
    dump_korean: bool,

    /// Write every Gear record with its weapon, mod and implant stats to this JSON file and exit
    #[arg(long, value_name = "FILE")]
    export_gear_json: Option<std::path::PathBuf>,

    /// Write the raw bytes of these tags (hex, comma separated) as <TAG>.bin into the --render-output directory and exit
    #[arg(long, value_delimiter = ',', value_parser = parse_tag)]
    dump_tags: Vec<tiger_pkg::TagHash>,

    /// Print every authoring path embedded in a tag, with how many tags hold it, and exit
    #[arg(long)]
    dump_paths: bool,

    /// Print every tag that references this tag (hex) and exit
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    dump_referrers: Option<tiger_pkg::TagHash>,

    /// Print every full-body runner animation clip with its frame count and name and exit
    #[arg(long)]
    list_clips: bool,

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
    /// AO, roughness), emissive, or fallback (no original shaders)
    #[arg(long, default_value = "final")]
    render_view: String,

    /// With --render-model: print the clips the packages tie to the model's skeleton
    #[arg(long)]
    list_model_clips: bool,

    /// Runner codename (thief, stealth, ...) for --list-model-clips, as the Models view derives it
    #[arg(long)]
    runner_codename: Option<String>,

    /// Pose --render-model with one animation clip tag (runner skins)
    #[arg(long, value_name = "TAG", value_parser = parse_tag)]
    render_clip: Option<tiger_pkg::TagHash>,

    /// Clip frame for --render-clip; negative renders the skeleton's bind pose
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    render_frame: f32,

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

    if args.list_skins {
        return crate::gui::GearView::print_skin_catalog()
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))));
    }

    if let Some(target) = args.dump_referrers {
        let cache = quicktag_scanner::load_tag_cache();
        for (tag, scan) in cache.hashes.iter() {
            if scan.file_hashes.iter().any(|item| item.hash == target) {
                let reference = package_manager().get_entry(*tag).map(|entry| entry.reference);
                println!("REFERRER {tag} reference={reference:08X?}");
            }
        }
        return Ok(());
    }

    if args.dump_paths {
        // Authoring paths embedded in tags, as the scanner's raw strings hold them.
        let cache = quicktag_scanner::load_tag_cache();
        let mut paths = std::collections::BTreeMap::<String, Vec<tiger_pkg::TagHash>>::new();
        for (tag, scan) in cache.hashes.iter() {
            for text in &scan.raw_strings {
                if text.starts_with("content") && text.contains(['\\', '/']) {
                    paths.entry(text.replace('/', "\\")).or_default().push(*tag);
                }
            }
        }
        for (path, tags) in &paths {
            println!("{path}\t{}\t{}", tags.len(), tags[0]);
        }
        return Ok(());
    }

    if !args.dump_tags.is_empty() {
        // A value that is not a tag is taken as a reference class: every tag of it is written.
        let tags = args
            .dump_tags
            .iter()
            .flat_map(|tag| match package_manager().get_entry(*tag) {
                Some(_) => vec![*tag],
                None => package_manager()
                    .get_all_by_reference(tag.0)
                    .into_iter()
                    .map(|(tag, _)| tag)
                    .collect(),
            })
            .collect::<Vec<_>>();
        for tag in &tags {
            let result = package_manager()
                .read_tag(*tag)
                .map_err(std::io::Error::other)
                .and_then(|data| std::fs::write(args.render_output.join(format!("{tag}.bin")), data));
            let reference = package_manager().get_entry(*tag).map(|entry| entry.reference);
            println!("TAG {tag} reference={reference:08X?} {result:?}");
            // Wide (Tag64) references, which the 32-bit hash scan cannot see.
            let data = package_manager().read_tag(*tag).unwrap_or_default();
            for offset in (0..data.len().saturating_sub(7)).step_by(4) {
                let wide = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                if let Some(entry) = package_manager().lookup.tag64_entries.get(&wide) {
                    let target = entry.hash32;
                    let class = package_manager().get_entry(target).map(|entry| entry.reference);
                    println!("  TAG64 {offset:#x} -> {target} reference={class:08X?}");
                }
            }
        }
        return Ok(());
    }

    if !args.dump_weapon_stats.is_empty() {
        let language = if args.dump_korean {
            quicktag_strings::localized::LocalizedLanguage::Korean
        } else {
            quicktag_strings::localized::LocalizedLanguage::English
        };
        return crate::gui::GearView::dump_weapon_stats(
            &args.dump_weapon_stats,
            &args.dump_weapon_mods,
            language,
        )
        .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))));
    }

    if let Some(output) = &args.export_gear_json {
        let language = if args.dump_korean {
            quicktag_strings::localized::LocalizedLanguage::Korean
        } else {
            quicktag_strings::localized::LocalizedLanguage::English
        };
        let exported = crate::gui::GearView::export_json(output, language)
            .map_err(|error| eframe::Error::AppCreation(Box::new(std::io::Error::other(error))))?;
        println!("Exported {exported} gear records to {}", output.display());
        return Ok(());
    }

    if args.list_clips {
        let names = animation::runner_clip_names();
        for group in animation::runner_clip_groups() {
            for clip in &group.clips {
                println!("RUNNERCLIP	{}	{}	{}	{:08X}	{}	{:08X}", group.codename.as_deref().unwrap_or("-"), clip.tag, clip.frames, clip.name_hash, clip.slots, clip.rig);
            }
        }
        for clip in animation::runner_clips() {
            let name = names.get(&clip.name_hash).cloned().unwrap_or_else(|| format!("{:08X}", clip.name_hash));
            println!("CLIP	{}	{}	{name}", clip.tag, clip.frames);
        }
        return Ok(());
    }

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
        let format = crate::texture::Texture::load_desc(tag).map(|desc| desc.format).ok();
        println!("Exported {tag} ({}x{}, {format:?}) to {}", image.width(), image.height(), args.render_output.display());
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
            args.render_clip.map(|clip| (clip, args.render_frame)),
            args.list_model_clips,
            args.runner_codename.as_deref(),
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
