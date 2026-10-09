mod audio;
mod audio_events;
mod audio_list;
mod common;
mod external_file;
mod gear;
mod hexview;
mod implant_stats;
mod item_effect;
mod perk_data;
mod model_renderer;
mod modellist;
mod named_tags;
mod packages;
mod profile_texture;
mod prop_handoff;
mod raw_strings;
mod signatures;
mod space_usage;
mod sticker_texture;
mod strings;
mod style;
mod tag;
mod texturelist;
mod video;
mod weapon_stats;

use std::cell::RefCell;
use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use eframe::egui::{CornerRadius, PointerButton, RichText, TextEdit, Widget};
use eframe::egui_wgpu::RenderState;
use eframe::{
    egui::{self},
    emath::Align2,
    epaint::{Color32, Vec2},
};
use egui_notify::Toasts;
use lazy_static::lazy_static;
use log::info;
use notify::Watcher;
use parking_lot::{Mutex, RwLock};
use poll_promise::Promise;
use quicktag_core::util::fnv1;
use quicktag_scanner::context::ScannerContext;
use quicktag_scanner::signatures::SIGNATURES_HASH;
use quicktag_scanner::{ScanStatus, TagCache, load_tag_cache, scanner_progress};
use quicktag_strings::localized::{
    LocalizedLanguage, RawStringHashCache, StringCache, create_stringmap,
    create_stringmap_for_language, supported_languages,
};
use rustc_hash::{FxHashMap, FxHashSet};
use strings::StringViewVariant;
use tiger_pkg::{TagHash, package_manager};

pub(crate) use self::gear::GearView;
pub(crate) use self::modellist::ModelsView;
use self::named_tags::NamedTagView;
use self::packages::PackagesView;
use self::raw_strings::RawStringsView;
use self::strings::StringsView;
use self::tag::TagView;
use self::texturelist::TexturesView;
use self::video::VideoView;
use crate::gui::external_file::ExternalFileScanView;
use crate::gui::signatures::SignaturesView;
use crate::gui::space_usage::SpaceUsageView;
use crate::gui::tag::TagHistory;
use crate::texture::cache::TextureCache;

#[derive(PartialEq)]
pub enum Panel {
    Tag,
    NamedTags,
    Packages,
    Textures,
    Models,
    Gear,
    Audio,
    Video,
    AudioEvents,
    Strings,
    SpaceUsage,
    ExternalFile,
}

#[derive(PartialEq)]
pub enum StringsPanel {
    Localized,
    Raw,
    Hashes,
    Signatures,
}

lazy_static! {
    pub static ref TOASTS: Arc<Mutex<Toasts>> = Arc::new(Mutex::new(Toasts::new()));
    pub static ref CACHE: RwLock<Arc<TagCache>> = RwLock::new(Arc::new(TagCache::default()));
    pub static ref RAW_STRING_HASH_LOOKUP: RwLock<Option<Arc<RawStringHashCache>>> =
        RwLock::new(None);
}

const GAME_LANGUAGE_STORAGE_KEY: &str = "quicktag_game_language";

struct LocalizedBundle {
    language: LocalizedLanguage,
    strings: Arc<StringCache>,
    gear: GearView,
}

fn load_localized_bundle(language: LocalizedLanguage) -> Result<LocalizedBundle, String> {
    let strings = Arc::new(
        create_stringmap_for_language(language)
            .map_err(|error| format!("Failed to load {} strings: {error:#}", language.label()))?,
    );
    if strings.is_empty() {
        return Err(format!(
            "The game packages did not contain any {} strings",
            language.label()
        ));
    }

    let gear = GearView::new_for_language(strings.clone(), language);
    if matches!(
        package_manager().version,
        tiger_pkg::GameVersion::Marathon(_)
    ) && let Some(error) = gear.load_error()
    {
        return Err(error.to_owned());
    }

    Ok(LocalizedBundle {
        language,
        strings,
        gear,
    })
}

fn add_localization_font_fallback(fonts: &mut egui::FontDefinitions) {
    let mut candidates = Vec::<PathBuf>::new();

    #[cfg(target_os = "windows")]
    {
        let windows = std::env::var_os("WINDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let font_dir = windows.join("Fonts");
        for filename in [
            "NotoSansCJKkr-Regular.otf",
            "malgun.ttf",
            "meiryo.ttc",
            "msyh.ttc",
            "msjh.ttc",
        ] {
            candidates.push(font_dir.join(filename));
        }
    }

    #[cfg(target_os = "linux")]
    candidates.extend([
        PathBuf::from("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        PathBuf::from("/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf"),
        PathBuf::from("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]);

    #[cfg(target_os = "macos")]
    candidates.extend([
        PathBuf::from("/System/Library/Fonts/PingFang.ttc"),
        PathBuf::from("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        PathBuf::from("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    ]);

    let Some((path, data)) = candidates
        .into_iter()
        .find_map(|path| std::fs::read(&path).ok().map(|data| (path, data)))
    else {
        log::warn!("No system CJK font fallback found; some localized glyphs may be unavailable");
        return;
    };

    const FONT_NAME: &str = "quicktag_localization_fallback";
    fonts.font_data.insert(
        FONT_NAME.to_owned(),
        Arc::new(egui::FontData::from_owned(data)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push(FONT_NAME.to_owned());
    }
    info!("Loaded localization font fallback from {}", path.display());
}

pub fn get_string_for_hash(hash: u32) -> Option<String> {
    let lookup = RAW_STRING_HASH_LOOKUP.read();
    let lookup = lookup.as_ref()?;
    lookup.get(&hash)?.first().cloned().map(|(s, _)| s)
}

pub struct QuickTagApp {
    scanner_context: ScannerContext,
    cache_load: Option<Promise<TagCache>>,
    reload_cache: bool,
    cache: Arc<TagCache>,
    current_wordlist_hash: u64,
    current_signatures_hash: u64,
    tag_history: Rc<RefCell<TagHistory>>,
    strings: Arc<StringCache>,
    raw_strings: Arc<RawStringHashCache>,
    language: LocalizedLanguage,
    loading_language: Option<LocalizedLanguage>,
    language_load: Option<Promise<Result<LocalizedBundle, String>>>,
    localized_string_cache: FxHashMap<LocalizedLanguage, Arc<StringCache>>,
    localized_gear_cache: FxHashMap<LocalizedLanguage, GearView>,

    texture_cache: TextureCache,

    tag_input: String,
    tag_split: bool,
    /// (pkg id, entry index)
    tag_split_input: (String, String),

    open_panel: Panel,
    strings_panel: StringsPanel,

    tag_view: Option<TagView>,
    external_file_view: Option<ExternalFileScanView>,

    named_tags_view: NamedTagView,
    packages_view: PackagesView,
    textures_view: TexturesView,
    models_view: ModelsView,
    gear_view: GearView,
    audio_view: audio_list::AudioView,
    video_view: VideoView,
    audio_events_view: audio_events::AudioEventView,
    strings_view: StringsView,
    raw_strings_view: RawStringsView,
    raw_string_hashes_view: StringsView,
    signatures_view: SignaturesView,
    space_usage_view: SpaceUsageView,

    _schemafile_watcher: notify::RecommendedWatcher,
    schemafile_update_rx: Receiver<Result<notify::Event, notify::Error>>,

    pub wgpu_state: RenderState,
}

impl QuickTagApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx
            .options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);
        let mut fonts = egui::FontDefinitions::default();
        match package_manager().version {
            tiger_pkg::GameVersion::Destiny(_) => {
                fonts.font_data.insert(
                    "Destiny_Keys".into(),
                    Arc::new(egui::FontData::from_static(include_bytes!(
                        "../../Destiny_Keys.otf"
                    ))),
                );

                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(1, "Destiny_Keys".to_owned());
            }
            tiger_pkg::GameVersion::Marathon(_) => {
                fonts.font_data.insert(
                    "goliath_symbols_pc".into(),
                    Arc::new(egui::FontData::from_static(include_bytes!(
                        "../../goliath_symbols_pc.otf"
                    ))),
                );

                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(1, "goliath_symbols_pc".to_owned());
            }
        }

        add_localization_font_fallback(&mut fonts);

        cc.egui_ctx.set_fonts(fonts);

        let requested_language = cc
            .storage
            .and_then(|storage| storage.get_string(GAME_LANGUAGE_STORAGE_KEY))
            .as_deref()
            .and_then(LocalizedLanguage::from_code)
            .filter(|language| supported_languages().contains(language))
            .unwrap_or_default();

        let initial_bundle = load_localized_bundle(requested_language).or_else(|error| {
            log::error!(
                "Could not restore {} localization: {error}",
                requested_language.label()
            );
            load_localized_bundle(LocalizedLanguage::English)
        });
        let (language, strings, gear_view) = match initial_bundle {
            Ok(bundle) => (bundle.language, bundle.strings, bundle.gear),
            Err(error) => {
                log::error!("Could not load localized strings: {error}");
                let strings = Arc::new(create_stringmap().unwrap_or_default());
                (
                    LocalizedLanguage::English,
                    strings.clone(),
                    GearView::new(strings),
                )
            }
        };
        let mut localized_string_cache = FxHashMap::default();
        localized_string_cache.insert(language, strings.clone());
        let texture_cache = TextureCache::new(cc.wgpu_render_state.clone().unwrap());
        let mut models_view = ModelsView::new(Default::default(), texture_cache.clone());
        models_view.set_weapon_catalog(gear_view.model_weapon_catalog());

        let (tx, rx) = std::sync::mpsc::channel();
        let mut schemafile_watcher = notify::recommended_watcher(tx).unwrap();
        let version_schema_path = quicktag_core::classes::version_schemafile_path();
        if !Path::new(&version_schema_path).exists() {
            std::fs::File::create(&version_schema_path).expect("Failed to create schema file");
        }
        schemafile_watcher
            .watch(
                Path::new(&version_schema_path),
                notify::RecursiveMode::NonRecursive,
            )
            .unwrap();
        if !Path::new("schema.txt").exists() {
            std::fs::File::create("schema.txt").expect("Failed to create schema file");
        }
        schemafile_watcher
            .watch(Path::new("schema.txt"), notify::RecursiveMode::NonRecursive)
            .unwrap();

        quicktag_core::classes::load_schemafile();
        quicktag_scanner::signatures::load_sigfile();

        QuickTagApp {
            scanner_context: ScannerContext::create(&package_manager())
                .expect("Failed to create scanner context"),
            cache_load: None,
            reload_cache: true,
            tag_history: Rc::new(RefCell::new(TagHistory::default())),
            cache: Default::default(),
            current_wordlist_hash: 0,
            current_signatures_hash: 0,
            tag_view: None,
            external_file_view: None,
            tag_input: String::new(),
            tag_split: false,
            tag_split_input: (String::new(), String::new()),

            texture_cache: texture_cache.clone(),

            open_panel: Panel::Tag,
            strings_panel: StringsPanel::Localized,

            named_tags_view: NamedTagView::new(),
            packages_view: PackagesView::new(texture_cache.clone()),
            textures_view: TexturesView::new(texture_cache.clone()),
            models_view,
            gear_view,
            audio_view: audio_list::AudioView::new(),
            video_view: VideoView::new(),
            audio_events_view: audio_events::AudioEventView::new(),
            strings_view: StringsView::new(
                strings.clone(),
                Default::default(),
                StringViewVariant::LocalizedStrings,
            ),
            raw_strings_view: RawStringsView::new(Default::default()),
            raw_string_hashes_view: StringsView::new(
                Arc::new(Default::default()),
                Default::default(),
                StringViewVariant::RawWordlist,
            ),
            signatures_view: SignaturesView::new(Default::default()),
            space_usage_view: SpaceUsageView::new(),

            strings,
            raw_strings: Default::default(),
            language,
            loading_language: None,
            language_load: None,
            localized_string_cache,
            localized_gear_cache: Default::default(),

            _schemafile_watcher: schemafile_watcher,
            schemafile_update_rx: rx,

            wgpu_state: cc.wgpu_render_state.clone().unwrap(),
        }
    }
}

impl eframe::App for QuickTagApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(GAME_LANGUAGE_STORAGE_KEY, self.language.code().to_owned());
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.finish_language_load(ctx);

        if self.reload_cache {
            self.cache_load = Some(Promise::spawn_thread("load_cache", move || {
                load_tag_cache()
            }));
            self.reload_cache = false;
        }

        if let Ok(Ok(event)) = self.schemafile_update_rx.try_recv()
            && matches!(
                event.kind,
                notify::EventKind::Modify(notify::event::ModifyKind::Data(_))
            )
        {
            quicktag_core::classes::load_schemafile();
            crate::geometry::invalidate_cached_models();
            info!("Reloaded schema file");
        }

        ctx.set_style(style::style());
        let mut is_loading_cache = false;
        if let Some(cache_promise) = self.cache_load.as_ref()
            && cache_promise.poll().is_pending()
        {
            {
                let painter = ctx.layer_painter(egui::LayerId::background());
                painter.rect_filled(
                    egui::Rect::EVERYTHING,
                    CornerRadius::default(),
                    Color32::from_black_alpha(127),
                );
            }
            egui::Window::new("Loading cache")
                .collapsible(false)
                .resizable(false)
                .title_bar(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    let progress = if let ScanStatus::Scanning {
                        current_package,
                        total_packages,
                    } = scanner_progress()
                    {
                        current_package as f32 / total_packages as f32
                    } else {
                        0.9999
                    };

                    ui.add(
                        egui::ProgressBar::new(progress)
                            .animate(true)
                            .text(scanner_progress().to_string()),
                    );
                    ctx.request_repaint();
                });

            // 

            is_loading_cache = true;
        }

        if self
            .cache_load
            .as_ref()
            .map(|v| v.poll().is_ready())
            .unwrap_or_default()
        {
            let c = self.cache_load.take().unwrap();
            let cache = c.try_take().unwrap_or_default();
            self.cache = Arc::new(cache);
            *CACHE.write() = self.cache.clone();

            self.strings_view = StringsView::new(
                self.strings.clone(),
                self.cache.clone(),
                StringViewVariant::LocalizedStrings,
            );
            self.raw_strings_view = RawStringsView::new(self.cache.clone());

            let mut new_rsh_cache = RawStringHashCache::default();
            for s in self
                .cache
                .hashes
                .iter()
                .flat_map(|(_, sc)| sc.raw_strings.iter().cloned())
            {
                let h = fnv1(s.as_bytes());
                let entry = new_rsh_cache.entry(h).or_default();
                if entry.iter().any(|(s2, _)| s2 == &s) {
                    continue;
                }

                entry.push((s, false));
            }

            let mut wordlist_hasher = DefaultHasher::new();
            quicktag_strings::wordlist::load_wordlist(|s, h| {
                wordlist_hasher.write(s.as_bytes());
                let entry = new_rsh_cache.entry(h).or_default();
                if entry.iter().any(|(s2, _)| s2 == s) {
                    return;
                }

                entry.push((s.to_string(), true));
            });
            self.current_wordlist_hash = wordlist_hasher.finish();
            self.current_signatures_hash =
                SIGNATURES_HASH.load(std::sync::atomic::Ordering::Relaxed);

            let mut filtered_wordlist_hashes: StringCache = Default::default();
            let found_hashes: FxHashSet<u32> = self
                .cache
                .hashes
                .iter()
                .flat_map(|(_, scan)| scan.wordlist_hashes.iter().map(|h| h.hash))
                .collect();
            for hash in found_hashes {
                if let Some(strings) = new_rsh_cache.get(&hash) {
                    filtered_wordlist_hashes
                        .insert(hash, strings.iter().map(|(s, _)| s.clone()).collect());
                }
            }
            // for (tag, _) in self
            //     .cache
            //     .hashes
            //     .iter()
            //     .filter(|(_, scan)| scan.wordlist_hashes.iter().any(|c| c.hash == *hash))
            // {
            //     self.string_selected_entries.push((
            //         *tag,
            //         label,
            //         TagType::from_type_subtype(e.file_type, e.file_subtype),
            //     ));
            // }

            self.raw_string_hashes_view = StringsView::new(
                Arc::new(filtered_wordlist_hashes),
                self.cache.clone(),
                StringViewVariant::RawWordlist,
            );

            self.signatures_view = SignaturesView::new(self.cache.clone());
            self.gear_view.reconcile_weapon_skin_models(&self.cache);
            self.models_view
                .set_weapon_catalog(self.gear_view.model_weapon_catalog());
            self.models_view.set_cache(self.cache.clone());

            // // Dump all raw strings to a csv file
            // if let Ok(mut f) = std::fs::File::create("raw_strings.csv") {
            //     writeln!(f, "hash|string|is_wordlist").unwrap();
            //     for (hash, strings) in new_rsh_cache.iter() {
            //         for (string, is_wordlist) in strings {
            //             writeln!(f, "{:08X}|{}|{}", hash, string, is_wordlist).unwrap();
            //         }
            //     }
            // }

            self.raw_strings = Arc::new(new_rsh_cache);
            *RAW_STRING_HASH_LOOKUP.write() = Some(Arc::clone(&self.raw_strings));

            {
                self.audio_events_view = audio_events::AudioEventView::new();
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_enabled_ui(!is_loading_cache, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("Scan file").clicked() {
                            if let Ok(Some(selected_file)) = native_dialog::FileDialog::new()
                                .add_filter("All files", &["*"])
                                .show_open_single_file()
                            {
                                let filename = selected_file
                                    .file_name()
                                    .unwrap()
                                    .to_string_lossy()
                                    .to_string();
                                let data = std::fs::read(&selected_file).unwrap();
                                self.external_file_view = Some(ExternalFileScanView::new(
                                    filename,
                                    &self.scanner_context,
                                    &data,
                                ));

                                self.open_panel = Panel::ExternalFile;
                            }

                            ui.close();
                        }

                        if ui.button("Regenerate Cache").clicked() {
                            self.regenerate_cache();
                            ui.close();
                        }
                    });

                    ui.separator();
                    ui.label("Game text:");
                    let mut requested_language = None;
                    ui.add_enabled_ui(self.language_load.is_none(), |ui| {
                        egui::ComboBox::from_id_salt("game_localization_language")
                            .selected_text(self.language.label())
                            .width(176.0)
                            .show_ui(ui, |ui| {
                                for &language in supported_languages() {
                                    if ui
                                        .selectable_label(
                                            self.language == language,
                                            language.label(),
                                        )
                                        .clicked()
                                    {
                                        requested_language = Some(language);
                                        ui.close();
                                    }
                                }
                            });
                    });
                    if let Some(language) = self.loading_language {
                        ui.spinner();
                        ui.weak(format!("Loading {}…", language.label()));
                    }
                    if let Some(language) = requested_language {
                        self.request_language(language, ctx);
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                        if self.current_signatures_hash != self.cache.signatures_hash && self.current_wordlist_hash != self.cache.wordlist_hash {
                            if ui.button(RichText::new("Signatures and wordlist changed, click here to regenerate your cache").color(Color32::ORANGE)).clicked() {
                                self.regenerate_cache();
                            }
                        } else {
                            if self.current_signatures_hash != self.cache.signatures_hash && ui.button(RichText::new("Signatures changed, click here to regenerate your cache").color(Color32::GREEN)).clicked() {
                                self.regenerate_cache();
                            }
                            if self.current_wordlist_hash != self.cache.wordlist_hash && ui.button(RichText::new("Wordlist changed, click here to regenerate your cache").color(Color32::YELLOW)).clicked() {
                                self.regenerate_cache();
                            }
                        }
                    });

                });
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Tag:");
                    let mut submitted = false;

                    if self.tag_split {
                        submitted |= TextEdit::singleline(&mut self.tag_split_input.0)
                            .hint_text("PKG ID")
                            .desired_width(64.)
                            .ui(ui)
                            .lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter));

                        submitted |= TextEdit::singleline(&mut self.tag_split_input.1)
                            .hint_text("Index")
                            .desired_width(64.)
                            .ui(ui)
                            .lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    } else {
                        submitted |= TextEdit::singleline(&mut self.tag_input)
                            .hint_text("32/64-bit hex tag")
                            .desired_width(128. + 8.)
                            .ui(ui)
                            .lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    }

                    if ui.button("Open").clicked() || submitted {
                        let tag_input_trimmed = self.tag_input.trim();
                        let tag = if self.tag_split {
                            let pkg_id = self.tag_split_input.0.trim();
                            let entry_index = self.tag_split_input.1.trim();

                            if pkg_id.is_empty() || entry_index.is_empty() {
                                TagHash::NONE
                            } else {
                                let pkg_id: u16 =
                                    u16::from_str_radix(pkg_id, 16).unwrap_or_default();
                                let entry_index = str::parse(entry_index).unwrap_or_default();
                                TagHash::new(pkg_id, entry_index)
                            }
                        } else if tag_input_trimmed.len() >= 16 {
                            let hash =
                                u64::from_str_radix(tag_input_trimmed, 16).unwrap_or_default();
                            if let Some(t) = package_manager()
                                .lookup
                                .tag64_entries
                                .get(&u64::from_be(hash))
                            {
                                t.hash32
                            } else {
                                TagHash::NONE
                            }
                        } else if tag_input_trimmed.len() > 8
                            && tag_input_trimmed.chars().all(char::is_numeric)
                        {
                            let hash = tag_input_trimmed.parse().unwrap_or_default();
                            TagHash(hash)
                        } else {
                            let hash =
                                u32::from_str_radix(tag_input_trimmed, 16).unwrap_or_default();

                            let hash = TagHash(hash);
                            if hash.is_valid() {
                                hash
                            } else {
                                // Try old format hash
                                let s = TagHash(hash.0.swap_bytes());
                                if s.is_valid() {
                                    TOASTS.lock().warning("Old-style flipped hashes (eg. from Alkahest/Charm) are deprecated.");
                                }
                                s
                            }
                        };

                        self.open_tag(tag, true);
                    }

                    ui.checkbox(&mut self.tag_split, "Split pkg/entry");
                });

                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.open_panel, Panel::Tag, "Tag");
                    ui.selectable_value(&mut self.open_panel, Panel::NamedTags, "Named tags");
                    ui.selectable_value(&mut self.open_panel, Panel::Packages, "Packages");
                    ui.selectable_value(&mut self.open_panel, Panel::Textures, "Textures");
                    ui.selectable_value(&mut self.open_panel, Panel::Models, "Models");
                    ui.selectable_value(&mut self.open_panel, Panel::Gear, "Gear");
                    ui.selectable_value(&mut self.open_panel, Panel::Audio, "Audio");
                    ui.selectable_value(&mut self.open_panel, Panel::Video, "Video");
                    ui.selectable_value(&mut self.open_panel, Panel::AudioEvents, "Wwise Events");
                    ui.selectable_value(&mut self.open_panel, Panel::Strings, "Strings");
                    ui.selectable_value(&mut self.open_panel, Panel::SpaceUsage, "Space Usage");
                    if let Some(external_file_view) = &self.external_file_view {
                        ui.selectable_value(
                            &mut self.open_panel,
                            Panel::ExternalFile,
                            format!("File {}", external_file_view.filename),
                        );
                    }
                });

                ui.separator();

                if self.open_panel == Panel::Strings {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.strings_panel, StringsPanel::Localized, "Localized");
                        ui.selectable_value(&mut self.strings_panel, StringsPanel::Raw, "Raw Strings");
                        ui.selectable_value(&mut self.strings_panel, StringsPanel::Hashes, "Hashes");
                        ui.selectable_value(&mut self.strings_panel, StringsPanel::Signatures, "Signatures");
                    });
                    ui.separator();
                }

                let action = match self.open_panel {
                    Panel::Tag => {
                        if let Some(tagview) = &mut self.tag_view {
                            tagview.view(ctx, ui)
                        } else {
                            ui.label("No tag loaded");
                            None
                        }
                    }
                    Panel::NamedTags => self.named_tags_view.view(ctx, ui),
                    Panel::Packages => self.packages_view.view(ctx, ui),
                    Panel::Textures => self.textures_view.view(ctx, ui),
                    Panel::Models => self.models_view.view(ctx, ui),
                    Panel::Gear => self.gear_view.view(ctx, ui, &self.texture_cache),
                    Panel::Audio => self.audio_view.view(ctx, ui),
                    Panel::Video => self.video_view.view(ctx, ui),
                    Panel::AudioEvents => self.audio_events_view.view(ctx, ui),
                    Panel::Strings => match self.strings_panel {
                        StringsPanel::Localized => self.strings_view.view(ctx, ui),
                        StringsPanel::Raw => self.raw_strings_view.view(ctx, ui),
                        StringsPanel::Hashes => self.raw_string_hashes_view.view(ctx, ui),
                        StringsPanel::Signatures => self.signatures_view.view(ctx, ui),
                    },
                    Panel::SpaceUsage => self.space_usage_view.view(ctx, ui),
                    Panel::ExternalFile => {
                        if let Some(external_file_view) = &mut self.external_file_view {
                            external_file_view.view(ctx, ui, &self.texture_cache)
                        } else {
                            self.open_panel = Panel::Tag;
                            None
                        }
                    }
                };

                if self.open_panel == Panel::Tag && action.is_none() {
                    if ui.input(|i| i.pointer.button_pressed(PointerButton::Extra1)) {
                        let t = self.tag_history.borrow_mut().back();
                        if let Some(t) = t {
                            self.open_tag(t, false);
                        }
                    }

                    if ui.input(|i| i.pointer.button_pressed(PointerButton::Extra2)) {
                        let t = self.tag_history.borrow_mut().forward();
                        if let Some(t) = t {
                            self.open_tag(t, false);
                        }
                    }
                }

                if let Some(action) = action {
                    match action {
                        ViewAction::OpenTag(t) => self.open_tag(t, true),
                        ViewAction::ShowTexture(t) => {
                            self.textures_view.show_texture(t);
                            self.open_panel = Panel::Textures;
                        }
                        ViewAction::ShowModel(t) => {
                            self.models_view.show_model(t);
                            self.open_panel = Panel::Models;
                        }
                    }
                }
            });
        });

        TOASTS.lock().show(ctx);

        // Redraw the window while we're loading textures. This prevents loading textures from seeming "stuck"
        if self.texture_cache.is_loading_textures() {
            ctx.request_repaint();
        }
    }
}

impl QuickTagApp {
    fn request_language(&mut self, language: LocalizedLanguage, ctx: &egui::Context) {
        if language == self.language || self.language_load.is_some() {
            return;
        }

        if let Some(strings) = self.localized_string_cache.get(&language).cloned()
            && let Some(gear) = self.localized_gear_cache.remove(&language)
        {
            self.apply_localized_bundle(LocalizedBundle {
                language,
                strings,
                gear,
            });
            TOASTS
                .lock()
                .success(format!("Game text changed to {}", language.label()));
            return;
        }

        self.loading_language = Some(language);
        self.language_load = Some(Promise::spawn_thread("load_localization", move || {
            load_localized_bundle(language)
        }));
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    fn finish_language_load(&mut self, ctx: &egui::Context) {
        let Some(promise) = self.language_load.as_ref() else {
            return;
        };
        if promise.poll().is_pending() {
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        }

        let result = self
            .language_load
            .take()
            .and_then(|promise| promise.try_take().ok());
        self.loading_language = None;

        match result {
            Some(Ok(bundle)) => {
                let language = bundle.language;
                self.apply_localized_bundle(bundle);
                TOASTS
                    .lock()
                    .success(format!("Game text changed to {}", language.label()));
            }
            Some(Err(error)) => {
                log::error!("Failed to change game text language: {error}");
                TOASTS.lock().error(error);
            }
            None => {
                log::error!("Localization loader completed without a result");
                TOASTS
                    .lock()
                    .error("Localization loader completed without a result");
            }
        }
    }

    fn apply_localized_bundle(&mut self, mut bundle: LocalizedBundle) {
        bundle.gear.inherit_ui_state(&self.gear_view);

        let previous_language = self.language;
        let previous_gear = std::mem::replace(&mut self.gear_view, bundle.gear);
        self.gear_view.reconcile_weapon_skin_models(&self.cache);
        self.models_view
            .set_weapon_catalog(self.gear_view.model_weapon_catalog());
        self.localized_gear_cache
            .insert(previous_language, previous_gear);

        self.language = bundle.language;
        self.strings = bundle.strings.clone();
        self.localized_string_cache
            .insert(bundle.language, bundle.strings.clone());
        self.strings_view.set_strings(bundle.strings.clone());
        if let Some(tag_view) = &mut self.tag_view {
            tag_view.set_string_cache(bundle.strings);
        }
    }

    fn open_tag(&mut self, tag: TagHash, push_history: bool) {
        let new_view = TagView::create(
            self.cache.clone(),
            self.tag_history.clone(),
            self.strings.clone(),
            self.raw_strings.clone(),
            tag,
            self.wgpu_state.clone(),
            self.texture_cache.clone(),
        );
        if new_view.is_some() {
            self.tag_view = new_view;
            self.open_panel = Panel::Tag;
        } else if package_manager().get_entry(tag).is_some() {
            TOASTS.lock().warning(format!(
                "Could not find tag '{}' ({tag}) in cache\nThis usually means it has no references",
                self.tag_input
            ));
        } else {
            TOASTS
                .lock()
                .error(format!("Could not find tag '{}' ({tag})", self.tag_input));
        }

        if push_history {
            self.tag_history.borrow_mut().push(tag);
        }
    }

    fn regenerate_cache(&mut self) {
        if let Err(e) = std::fs::remove_file(quicktag_scanner::cache_path()) {
            log::error!("Failed to remove cache file: {}", e);
        } else {
            self.tag_view = None;
            self.open_panel = Panel::Tag;

            self.reload_cache = true;
        }
    }
}

pub enum ViewAction {
    OpenTag(TagHash),
    ShowTexture(TagHash),
    ShowModel(TagHash),
}

pub trait View {
    fn view(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> Option<ViewAction>;
}
