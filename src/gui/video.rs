use crate::gui::common::tag_context;
use crate::gui::{View, ViewAction, audio};
use crate::util::format_file_size;
use eframe::egui;
use eframe::egui::{ColorImage, Key, TextureHandle, TextureOptions};
use eframe::wgpu::naga::FastIndexMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use tiger_pkg::manager::PackagePath;
use tiger_pkg::version::EngineVersion;
use tiger_pkg::{TagHash, package_manager};

#[derive(Clone, Copy)]
struct VideoEntry {
    tag: TagHash,
    size: usize,
}

struct PackageVideos {
    entries: Vec<VideoEntry>,
}

impl PackageVideos {
    fn by_pkg_id(id: u16) -> Self {
        let (file_type, file_subtype) = criware_usm_type();
        let entries = package_manager()
            .get_all_by_type(file_type, Some(file_subtype))
            .iter()
            .filter(|(tag, _)| tag.pkg_id() == id)
            .map(|(tag, entry)| VideoEntry {
                tag: *tag,
                size: entry.file_size as usize,
            })
            .collect();
        Self { entries }
    }
}

fn criware_usm_type() -> (u8, u8) {
    match package_manager().version.engine_version() {
        EngineVersion::TigerD2v1 => (27, 0),
        EngineVersion::TigerD2v2 | EngineVersion::TigerGoliath => (27, 1),
        _ => (u8::MAX, u8::MAX),
    }
}

struct PlaybackControl {
    stopped: AtomicBool,
    paused: AtomicBool,
}

impl PlaybackControl {
    fn new(paused: bool) -> Self {
        Self {
            stopped: AtomicBool::new(false),
            paused: AtomicBool::new(paused),
        }
    }
}

struct DecodedFrame {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    index: u64,
}

enum DecoderEvent {
    Metadata {
        width: usize,
        height: usize,
        fps: f32,
        total_frames: u64,
    },
    Frame(DecodedFrame),
    Finished,
    Error(String),
}

struct VideoPlayback {
    tag: TagHash,
    receiver: mpsc::Receiver<DecoderEvent>,
    control: Arc<PlaybackControl>,
    texture: Option<TextureHandle>,
    dimensions: Option<(usize, usize)>,
    fps: Option<f32>,
    total_frames: Option<u64>,
    frame_index: u64,
    finished: bool,
    error: Option<String>,
    audio_ready: Arc<Mutex<Option<Vec<u8>>>>,
    audio_wav: Option<Arc<Vec<u8>>>,
    audio_started: bool,
    has_frame: bool,
}

impl VideoPlayback {
    fn spawn(tag: TagHash) -> Self {
        Self::spawn_at(tag, 0, false, None)
    }

    fn spawn_at(
        tag: TagHash,
        start_frame: u64,
        start_paused: bool,
        cached_audio: Option<Arc<Vec<u8>>>,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(2);
        let control = Arc::new(PlaybackControl::new(start_paused));
        let audio_ready = Arc::new(Mutex::new(None));
        let worker_control = control.clone();
        let worker_audio = audio_ready.clone();
        let decode_audio_track = cached_audio.is_none();
        thread::Builder::new()
            .name(format!("video-{tag}"))
            .spawn(move || {
                decode_video(
                    tag,
                    sender,
                    worker_control,
                    worker_audio,
                    start_frame,
                    start_paused,
                    decode_audio_track,
                )
            })
            .expect("failed to spawn video decoder");

        Self {
            tag,
            receiver,
            control,
            texture: None,
            dimensions: None,
            fps: None,
            total_frames: None,
            frame_index: start_frame,
            finished: false,
            error: None,
            audio_ready,
            audio_wav: cached_audio,
            audio_started: false,
            has_frame: false,
        }
    }

    fn stop(&self) {
        self.control.stopped.store(true, Ordering::Release);
    }

    fn is_paused(&self) -> bool {
        self.control.paused.load(Ordering::Acquire)
    }

    fn set_paused(&mut self, paused: bool) {
        self.control.paused.store(paused, Ordering::Release);
        if paused {
            self.stop_audio();
        } else {
            self.audio_started = false;
        }
    }

    fn stop_audio(&mut self) {
        audio::AudioPlayer::instance().stop();
        self.audio_started = false;
    }

    fn restart_audio(&mut self) {
        self.stop_audio();
    }

    fn position_seconds(&self) -> f64 {
        self.fps
            .filter(|fps| *fps > 0.0)
            .map(|fps| self.frame_index as f64 / fps as f64)
            .unwrap_or(0.0)
    }

    fn duration_seconds(&self) -> Option<f64> {
        match (self.total_frames, self.fps) {
            (Some(frames), Some(fps)) if frames > 0 && fps > 0.0 => {
                Some(frames as f64 / fps as f64)
            }
            _ => None,
        }
    }

    fn frame_for_seconds(&self, seconds: f64) -> Option<u64> {
        let fps = self.fps?;
        let total = self.total_frames?;
        if total == 0 || fps <= 0.0 {
            return None;
        }
        let frame = (seconds.max(0.0) * fps as f64).round() as u64;
        Some(frame.min(total - 1))
    }

    fn start_audio(&mut self, volume: f32) {
        let Some(wav) = &self.audio_wav else {
            return;
        };
        if volume <= 0.0001 || self.is_paused() || self.finished || !self.has_frame {
            return;
        }
        let Some(segment) = make_pcm_wav_segment(wav, self.position_seconds(), volume) else {
            return;
        };
        self.audio_started = audio::AudioPlayer::instance().play_video_wav(segment);
    }

    fn update(&mut self, ctx: &egui::Context, volume: f32, muted: bool) {
        if self.audio_wav.is_none() {
            if let Some(wav) = self.audio_ready.lock().unwrap().take() {
                self.audio_wav = Some(Arc::new(wav));
            }
        }

        while let Ok(event) = self.receiver.try_recv() {
            match event {
                DecoderEvent::Metadata {
                    width,
                    height,
                    fps,
                    total_frames,
                } => {
                    self.dimensions = Some((width, height));
                    self.fps = Some(fps);
                    self.total_frames = Some(total_frames);
                    if total_frames > 0 {
                        self.frame_index = self.frame_index.min(total_frames - 1);
                    }
                }
                DecoderEvent::Frame(frame) => {
                    let image = ColorImage::from_rgba_unmultiplied(
                        [frame.width, frame.height],
                        &frame.pixels,
                    );
                    if let Some(texture) = &mut self.texture {
                        texture.set(image, TextureOptions::LINEAR);
                    } else {
                        self.texture = Some(ctx.load_texture(
                            format!("video-{}", self.tag),
                            image,
                            TextureOptions::LINEAR,
                        ));
                    }
                    self.frame_index = frame.index;
                    self.has_frame = true;
                }
                DecoderEvent::Finished => {
                    self.finished = true;
                    self.stop_audio();
                }
                DecoderEvent::Error(error) => {
                    self.error = Some(error);
                    self.stop_audio();
                }
            }
        }

        if muted || volume <= 0.0001 {
            if self.audio_started {
                self.stop_audio();
            }
        } else if !self.audio_started && !self.is_paused() && !self.finished && self.has_frame {
            self.start_audio(volume);
        }
    }
}

impl Drop for VideoPlayback {
    fn drop(&mut self) {
        self.stop();
        if self.audio_started {
            self.stop_audio();
        }
    }
}

pub struct VideoView {
    selected_package: u16,
    selected_videos: Option<PackageVideos>,
    packages: FastIndexMap<u16, PackagePath>,
    current_row: usize,
    playback: Option<VideoPlayback>,
    volume: f32,
    volume_drag: Option<f32>,
    muted: bool,
    loop_playback: bool,
    fullscreen: bool,
    scrub_position: Option<f64>,
}

impl VideoView {
    pub fn new() -> Self {
        let (file_type, file_subtype) = criware_usm_type();
        let video_package_ids: std::collections::HashSet<u16> = package_manager()
            .get_all_by_type(file_type, Some(file_subtype))
            .iter()
            .map(|(tag, _)| tag.pkg_id())
            .collect();
        let mut packages: Vec<_> = package_manager()
            .package_paths
            .iter()
            .filter(|(id, _)| video_package_ids.contains(id))
            .map(|(id, path)| (*id, path.clone()))
            .collect();
        packages.sort_by_cached_key(|(_, path)| format!("{}_{}", path.name, path.id));

        Self {
            selected_package: u16::MAX,
            selected_videos: None,
            packages: packages.into_iter().collect(),
            current_row: 0,
            playback: None,
            volume: 1.0,
            volume_drag: None,
            muted: false,
            loop_playback: false,
            fullscreen: false,
            scrub_position: None,
        }
    }

    fn select_package(&mut self, id: u16) {
        self.stop_playback();
        self.selected_package = id;
        self.selected_videos = Some(PackageVideos::by_pkg_id(id));
        self.current_row = 0;
        self.scrub_position = None;
    }

    fn select_video(&mut self, row: usize) {
        let Some(tag) = self
            .selected_videos
            .as_ref()
            .and_then(|videos| videos.entries.get(row))
            .map(|entry| entry.tag)
        else {
            return;
        };
        self.stop_playback();
        self.current_row = row;
        self.scrub_position = None;
        self.playback = Some(VideoPlayback::spawn(tag));
    }

    fn select_relative_video(&mut self, delta: i32) {
        let len = self
            .selected_videos
            .as_ref()
            .map(|videos| videos.entries.len())
            .unwrap_or(0);
        if len == 0 {
            return;
        }
        let row = (self.current_row as i32 + delta).clamp(0, len as i32 - 1) as usize;
        if row != self.current_row || self.playback.is_none() {
            self.select_video(row);
        }
    }

    fn restart_at_frame(&mut self, frame: u64, paused: bool) {
        let Some(mut old) = self.playback.take() else {
            return;
        };
        let tag = old.tag;
        let cached_audio = old.audio_wav.clone();
        old.stop();
        old.stop_audio();
        self.scrub_position = None;
        self.playback = Some(VideoPlayback::spawn_at(tag, frame, paused, cached_audio));
    }

    fn seek_to_seconds(&mut self, seconds: f64) {
        let Some(playback) = &self.playback else {
            return;
        };
        let Some(frame) = playback.frame_for_seconds(seconds) else {
            return;
        };
        let paused = playback.is_paused();
        self.restart_at_frame(frame, paused);
    }

    fn seek_relative(&mut self, seconds: f64) {
        let Some(position) = self.playback.as_ref().map(VideoPlayback::position_seconds) else {
            return;
        };
        self.seek_to_seconds(position + seconds);
    }

    fn step_frame(&mut self, delta: i64) {
        let Some(playback) = &self.playback else {
            return;
        };
        let total = playback.total_frames.unwrap_or(0);
        if total == 0 {
            return;
        }
        let frame = (playback.frame_index as i64 + delta).clamp(0, total as i64 - 1) as u64;
        self.restart_at_frame(frame, true);
    }

    fn toggle_pause(&mut self) {
        let Some(finished) = self.playback.as_ref().map(|playback| playback.finished) else {
            return;
        };
        if finished {
            self.restart_at_frame(0, false);
            return;
        }
        if let Some(playback) = &mut self.playback {
            let paused = playback.is_paused();
            playback.set_paused(!paused);
        }
    }

    fn set_muted(&mut self, muted: bool) {
        if self.muted == muted {
            return;
        }
        self.muted = muted;
        if let Some(playback) = &mut self.playback {
            playback.restart_audio();
        }
    }

    fn set_volume(&mut self, volume: f32) {
        let volume = volume.clamp(0.0, 1.0);
        if (self.volume - volume).abs() < f32::EPSILON {
            return;
        }
        self.volume = volume;
        if let Some(playback) = &mut self.playback {
            playback.restart_audio();
        }
    }

    fn set_fullscreen(&mut self, ctx: &egui::Context, fullscreen: bool) {
        if self.fullscreen == fullscreen {
            return;
        }
        self.fullscreen = fullscreen;
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(fullscreen));
    }

    fn stop_playback(&mut self) {
        if let Some(playback) = self.playback.take() {
            playback.stop();
            audio::AudioPlayer::instance().stop();
        }
        self.scrub_position = None;
    }
}

impl View for VideoView {
    fn view(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> Option<ViewAction> {
        if self.fullscreen
            && !ctx.wants_keyboard_input()
            && ctx.input(|input| input.key_pressed(Key::Escape))
        {
            self.set_fullscreen(ctx, false);
        }

        if !self.fullscreen {
            egui::SidePanel::left("video_packages_panel")
                .resizable(true)
                .min_width(230.0)
                .show_inside(ui, |ui| {
                    ui.heading("Packages");
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let package_rows: Vec<_> = self
                            .packages
                            .iter()
                            .map(|(id, path)| (*id, path.clone()))
                            .collect();
                        for (id, path) in package_rows {
                            let name = format!("{id:04x}: {}_{}", path.name, path.id);
                            if ui
                                .selectable_label(self.selected_package == id, name)
                                .clicked()
                            {
                                self.select_package(id);
                            }
                        }
                    });
                });

            egui::SidePanel::left("video_entries_panel")
                .resizable(true)
                .default_width(220.0)
                .show_inside(ui, |ui| {
                    ui.heading("Videos");
                    let entries = self
                        .selected_videos
                        .as_ref()
                        .map(|videos| videos.entries.clone())
                        .unwrap_or_default();

                    let mut requested_row = None;
                    if !entries.is_empty() && !ctx.wants_keyboard_input() {
                        if ui.input(|input| input.key_pressed(Key::ArrowDown)) {
                            requested_row = Some((self.current_row + 1).min(entries.len() - 1));
                        }
                        if ui.input(|input| input.key_pressed(Key::ArrowUp)) {
                            requested_row = Some(self.current_row.saturating_sub(1));
                        }
                    }

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (row, entry) in entries.iter().enumerate() {
                            let response = ui.selectable_label(
                                row == self.current_row && self.playback.is_some(),
                                format!("{}  {}", entry.tag, format_file_size(entry.size)),
                            );
                            response.context_menu(|ui| tag_context(ui, entry.tag));
                            if response.clicked() {
                                requested_row = Some(row);
                            }
                        }
                    });
                    if let Some(row) = requested_row {
                        self.select_video(row);
                    }
                });
        }

        let mut toggle_pause = false;
        let mut stop = false;
        let mut restart = false;
        let mut previous = false;
        let mut next = false;
        let mut seek_to = None;
        let mut seek_delta = None;
        let mut step_delta = None;
        let mut requested_muted = self.muted;
        let mut requested_loop = self.loop_playback;
        let mut requested_fullscreen = self.fullscreen;
        let mut scrub_position = self.scrub_position;
        let mut volume_drag = self.volume_drag;
        let mut committed_volume = None;
        let mut displayed_volume = volume_drag.unwrap_or(self.volume);
        let entries_len = self
            .selected_videos
            .as_ref()
            .map(|videos| videos.entries.len())
            .unwrap_or(0);
        let current_row = self.current_row;
        let playback_volume = self.volume;
        let playback_muted = self.muted;
        let fullscreen_now = self.fullscreen;

        if let Some(playback) = &mut self.playback {
            playback.update(ctx, playback_volume, playback_muted);

            if !fullscreen_now {
                egui::TopBottomPanel::top("video_info_panel")
                    .resizable(false)
                    .show_inside(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.monospace(playback.tag.to_string());
                            ui.separator();
                            if let (Some((width, height)), Some(fps)) =
                                (playback.dimensions, playback.fps)
                            {
                                let frame_text = match playback.total_frames {
                                    Some(total) if total > 0 => {
                                        format!("frame {}/{}", playback.frame_index + 1, total)
                                    }
                                    _ => format!("frame {}", playback.frame_index + 1),
                                };
                                ui.label(format!("{width}×{height} · {fps:.3} fps · {frame_text}"));
                            } else {
                                ui.label("Reading stream metadata…");
                            }
                        });
                    });
            }

            egui::TopBottomPanel::bottom("video_controls_panel")
                .resizable(false)
                .show_inside(ui, |ui| {
                    ui.add_space(3.0);
                    let current = if playback.finished {
                        playback
                            .duration_seconds()
                            .unwrap_or_else(|| playback.position_seconds())
                    } else {
                        playback.position_seconds()
                    };
                    let duration = playback.duration_seconds();
                    let shown_position = scrub_position.unwrap_or(current);

                    ui.horizontal(|ui| {
                        ui.monospace(format_timestamp(shown_position));
                        if let Some(duration) = duration {
                            let mut slider_position = shown_position.clamp(0.0, duration);
                            let width = (ui.available_width() - 90.0).max(80.0);
                            let response = ui.add_sized(
                                [width, 18.0],
                                egui::Slider::new(&mut slider_position, 0.0..=duration)
                                    .show_value(false),
                            );
                            if response.changed() {
                                scrub_position = Some(slider_position);
                            }
                            let released = ui.input(|input| input.pointer.any_released());
                            if scrub_position.is_some()
                                && (released || (response.changed() && !response.dragged()))
                            {
                                seek_to = scrub_position.take();
                            }
                            response.on_hover_text("Seek");
                            ui.monospace(format_timestamp(duration));
                        } else {
                            let mut unavailable_progress = 0.0f64;
                            ui.add_enabled(
                                false,
                                egui::Slider::new(&mut unavailable_progress, 0.0..=1.0)
                                    .show_value(false),
                            );
                            ui.monospace("--:--");
                        }
                    });

                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(current_row > 0, egui::Button::new("⏮"))
                            .on_hover_text("Previous video")
                            .clicked()
                        {
                            previous = true;
                        }
                        if ui
                            .button(if playback.finished {
                                "▶ Replay"
                            } else if playback.is_paused() {
                                "▶ Play"
                            } else {
                                "⏸ Pause"
                            })
                            .on_hover_text("Play / pause  (Space)")
                            .clicked()
                        {
                            toggle_pause = true;
                        }
                        if ui.button("■").on_hover_text("Stop").clicked() {
                            stop = true;
                        }
                        if ui
                            .add_enabled(
                                entries_len > 0 && current_row + 1 < entries_len,
                                egui::Button::new("⏭"),
                            )
                            .on_hover_text("Next video")
                            .clicked()
                        {
                            next = true;
                        }
                        if ui.button("↻").on_hover_text("Restart  (Home)").clicked() {
                            restart = true;
                        }

                        ui.separator();
                        if ui
                            .button(if requested_muted || displayed_volume <= 0.0001 {
                                "Unmute"
                            } else {
                                "Mute"
                            })
                            .on_hover_text("Mute / unmute  (M)")
                            .clicked()
                        {
                            requested_muted = !requested_muted;
                        }

                        let volume_response = ui.add_sized(
                            [110.0, 18.0],
                            egui::Slider::new(&mut displayed_volume, 0.0..=1.0).show_value(false),
                        );
                        if volume_response.changed() {
                            volume_drag = Some(displayed_volume);
                            if !volume_response.dragged() {
                                committed_volume = volume_drag.take();
                            }
                        }
                        if volume_drag.is_some() && ui.input(|input| input.pointer.any_released()) {
                            committed_volume = volume_drag.take();
                        }
                        volume_response.on_hover_text(format!(
                            "Volume: {}%",
                            (displayed_volume * 100.0).round() as u32
                        ));

                        ui.separator();
                        ui.toggle_value(&mut requested_loop, "Loop")
                            .on_hover_text("Loop this video");

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button(if requested_fullscreen {
                                    "Exit fullscreen"
                                } else {
                                    "Fullscreen"
                                })
                                .on_hover_text("Toggle fullscreen  (F)")
                                .clicked()
                            {
                                requested_fullscreen = !requested_fullscreen;
                            }
                        });
                    });
                    ui.add_space(3.0);
                });

            if playback.error.is_some() {
                let error = playback.error.as_deref().unwrap_or_default();
                ui.centered_and_justified(|ui| {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                });
            } else if let Some(texture) = &playback.texture {
                let available = ui.available_size();
                let source = texture.size_vec2();
                let scale = (available.x / source.x)
                    .min(available.y / source.y)
                    .max(0.0);
                let size = source * scale;
                ui.centered_and_justified(|ui| {
                    let response = ui.add(
                        egui::Image::new(texture)
                            .fit_to_exact_size(size)
                            .sense(egui::Sense::click()),
                    );
                    if response.double_clicked() {
                        requested_fullscreen = !requested_fullscreen;
                    } else if response.clicked() {
                        toggle_pause = true;
                    }
                });
                if !playback.finished && !playback.is_paused() {
                    ctx.request_repaint_after(Duration::from_millis(8));
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        ui.label("Reading USM and starting decoder…");
                    });
                });
                ctx.request_repaint_after(Duration::from_millis(16));
            }

            if playback.finished && requested_loop {
                restart = true;
            }

            if !ctx.wants_keyboard_input() {
                let input = ctx.input(|input| {
                    (
                        input.key_pressed(Key::Space),
                        input.key_pressed(Key::M),
                        input.key_pressed(Key::F),
                        input.key_pressed(Key::Home),
                        input.key_pressed(Key::ArrowLeft),
                        input.key_pressed(Key::ArrowRight),
                        input.modifiers.shift,
                        input.key_pressed(Key::Escape),
                    )
                });
                if input.0 {
                    toggle_pause = true;
                }
                if input.1 {
                    requested_muted = !requested_muted;
                }
                if input.2 {
                    requested_fullscreen = !requested_fullscreen;
                }
                if input.3 {
                    restart = true;
                }
                if input.4 {
                    if input.6 {
                        step_delta = Some(-1);
                    } else {
                        seek_delta = Some(-5.0);
                    }
                }
                if input.5 {
                    if input.6 {
                        step_delta = Some(1);
                    } else {
                        seek_delta = Some(5.0);
                    }
                }
                if input.7 && requested_fullscreen {
                    requested_fullscreen = false;
                }
            }
        } else {
            ui.centered_and_justified(|ui| ui.label("Select a video to decode"));
        }

        self.scrub_position = scrub_position;
        self.volume_drag = volume_drag;
        self.loop_playback = requested_loop;

        if requested_muted != self.muted {
            self.set_muted(requested_muted);
        }
        if let Some(volume) = committed_volume {
            self.set_volume(volume);
        }
        if requested_fullscreen != self.fullscreen {
            self.set_fullscreen(ctx, requested_fullscreen);
        }

        if stop {
            self.stop_playback();
        } else if previous {
            self.select_relative_video(-1);
        } else if next {
            self.select_relative_video(1);
        } else if restart {
            self.restart_at_frame(0, false);
        } else if let Some(frame_delta) = step_delta {
            self.step_frame(frame_delta);
        } else if let Some(seconds) = seek_delta {
            self.seek_relative(seconds);
        } else if let Some(seconds) = seek_to {
            self.seek_to_seconds(seconds);
        } else if toggle_pause {
            self.toggle_pause();
        }

        None
    }
}

fn format_timestamp(seconds: f64) -> String {
    let total = seconds.max(0.0).floor() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn decode_video(
    tag: TagHash,
    sender: mpsc::SyncSender<DecoderEvent>,
    control: Arc<PlaybackControl>,
    audio_ready: Arc<Mutex<Option<Vec<u8>>>>,
    start_frame: u64,
    start_paused: bool,
    decode_audio_track: bool,
) {
    if let Err(error) = decode_video_inner(
        tag,
        &sender,
        &control,
        audio_ready,
        start_frame,
        start_paused,
        decode_audio_track,
    ) {
        let _ = sender.send(DecoderEvent::Error(format!(
            "Video decode failed: {error:#}"
        )));
    }
}

fn decode_video_inner(
    tag: TagHash,
    sender: &mpsc::SyncSender<DecoderEvent>,
    control: &Arc<PlaybackControl>,
    audio_ready: Arc<Mutex<Option<Vec<u8>>>>,
    start_frame: u64,
    start_paused: bool,
    decode_audio_track: bool,
) -> anyhow::Result<()> {
    let usm = Arc::new(package_manager().read_tag(tag)?);
    let stream_info = find_mpeg2_stream_info(&usm)?;
    let total_frames = count_mpeg2_frames(&usm)?;
    if sender
        .send(DecoderEvent::Metadata {
            width: stream_info.width,
            height: stream_info.height,
            fps: stream_info.fps,
            total_frames,
        })
        .is_err()
    {
        return Ok(());
    }

    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mpegvideo",
        "-i",
        "pipe:0",
        "-an",
    ]);
    if start_frame > 0 {
        let trim_filter = format!("trim=start_frame={start_frame},setpts=PTS-STARTPTS");
        command.args(["-vf", trim_filter.as_str()]);
    }
    command.args([
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
    ]);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().map_err(|error| {
        anyhow::anyhow!("could not start ffmpeg ({error}). Install ffmpeg and add it to PATH")
    })?;
    let mut stdin = child.stdin.take().unwrap();

    if decode_audio_track {
        let audio_usm = usm.clone();
        let audio_control = control.clone();
        let audio_output = audio_ready.clone();
        thread::spawn(move || decode_audio(audio_usm, audio_control, audio_output));
    }

    let feeder_control = control.clone();
    let feeder = thread::spawn(move || {
        for chunk in UsmChunks::new(&usm) {
            let Ok(chunk) = chunk else { break };
            if feeder_control.stopped.load(Ordering::Acquire) {
                break;
            }
            if chunk.kind == *b"@SFV"
                && chunk.payload_kind == 0
                && stdin.write_all(chunk.payload).is_err()
            {
                break;
            }
        }
    });

    let mut stdout = child.stdout.take().unwrap();
    let frame_size = stream_info.width * stream_info.height * 4;
    let mut frame_index = start_frame;
    let frame_interval = Duration::from_secs_f64(1.0 / stream_info.fps as f64);
    let mut next_frame_at = Instant::now();
    let mut first_visible_frame = true;

    loop {
        if control.stopped.load(Ordering::Acquire) {
            let _ = child.kill();
            break;
        }

        let showing_initial_paused_frame = first_visible_frame && start_paused;
        let mut resumed_from_pause = false;
        if !showing_initial_paused_frame {
            while control.paused.load(Ordering::Acquire) {
                resumed_from_pause = true;
                if control.stopped.load(Ordering::Acquire) {
                    let _ = child.kill();
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            if control.stopped.load(Ordering::Acquire) {
                break;
            }
        }
        if resumed_from_pause {
            next_frame_at = Instant::now();
        }

        let mut pixels = vec![0u8; frame_size];
        match stdout.read_exact(&mut pixels) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }

        if !first_visible_frame {
            while Instant::now() < next_frame_at && !control.stopped.load(Ordering::Acquire) {
                thread::sleep(
                    next_frame_at
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(8)),
                );
            }
        }

        if sender
            .send(DecoderEvent::Frame(DecodedFrame {
                pixels,
                width: stream_info.width,
                height: stream_info.height,
                index: frame_index,
            }))
            .is_err()
        {
            let _ = child.kill();
            break;
        }

        if first_visible_frame {
            next_frame_at = Instant::now() + frame_interval;
        } else {
            next_frame_at += frame_interval;
        }
        first_visible_frame = false;
        frame_index += 1;
    }

    let _ = feeder.join();
    if !control.stopped.load(Ordering::Acquire) {
        let status = child.wait()?;
        if status.success() {
            let _ = sender.send(DecoderEvent::Finished);
        } else {
            let mut error = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut error);
            }
            anyhow::bail!("ffmpeg exited with {status}: {}", error.trim());
        }
    }
    Ok(())
}

fn decode_audio(
    usm: Arc<Vec<u8>>,
    control: Arc<PlaybackControl>,
    output: Arc<Mutex<Option<Vec<u8>>>>,
) {
    // Let FFmpeg demux the USM itself instead of concatenating every @SFA payload.
    // @SFA can contain multiple stream indices and codec-specific header/extradata;
    // flattening all payloads into one byte stream corrupts multi-track files and
    // bypasses important USM audio metadata.
    let mut command = Command::new("ffmpeg");
    command.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "usm",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
    ]);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }

    let Ok(mut child) = command.spawn() else {
        eprintln!("video audio: failed to start ffmpeg");
        return;
    };
    let Some(mut stdin) = child.stdin.take() else {
        return;
    };

    let writer_control = control.clone();
    let writer = thread::spawn(move || {
        if !writer_control.stopped.load(Ordering::Acquire) {
            let _ = stdin.write_all(&usm);
        }
        // Drop stdin here so ffmpeg sees EOF.
    });

    let Some(mut stdout) = child.stdout.take() else {
        let _ = writer.join();
        return;
    };
    let mut stderr = child.stderr.take();
    let mut wav = Vec::new();
    let read_ok = stdout.read_to_end(&mut wav).is_ok();
    let _ = writer.join();
    let status = child.wait();

    if control.stopped.load(Ordering::Acquire) {
        return;
    }

    match status {
        Ok(status) if status.success() && read_ok && !wav.is_empty() => {
            *output.lock().unwrap() = Some(wav);
        }
        Ok(status) => {
            let mut message = String::new();
            if let Some(mut stderr) = stderr.take() {
                let _ = stderr.read_to_string(&mut message);
            }
            eprintln!(
                "video audio: ffmpeg failed with {status}: {}",
                message.trim()
            );
        }
        Err(error) => {
            eprintln!("video audio: failed waiting for ffmpeg: {error}");
        }
    }
}

fn count_mpeg2_frames(usm: &[u8]) -> anyhow::Result<u64> {
    let mut count = 0u64;
    let mut window = [0u8; 4];
    let mut filled = 0usize;

    for chunk in UsmChunks::new(usm) {
        let chunk = chunk?;
        if chunk.kind != *b"@SFV" || chunk.payload_kind != 0 {
            continue;
        }
        for &byte in chunk.payload {
            if filled < 4 {
                window[filled] = byte;
                filled += 1;
            } else {
                window.copy_within(1..4, 0);
                window[3] = byte;
            }
            if filled == 4 && window == [0x00, 0x00, 0x01, 0x00] {
                count += 1;
            }
        }
    }

    Ok(count)
}

fn make_pcm_wav_segment(wav: &[u8], start_seconds: f64, volume: f32) -> Option<Vec<u8>> {
    if wav.len() < 12 || &wav[..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }

    let mut offset = 12usize;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut audio_format = None;
    let mut data = None;

    while offset.checked_add(8)? <= wav.len() {
        let id = &wav[offset..offset + 4];
        let size = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into().ok()?) as usize;
        let chunk_start = offset.checked_add(8)?;
        let chunk_end = match chunk_start.checked_add(size) {
            Some(end) if end <= wav.len() => end,
            // ffmpeg writes 0xffffffff for the data chunk size when WAV is streamed to a pipe.
            // In that case the actual data extends to the end of the captured byte buffer.
            _ if id == b"data" => wav.len(),
            _ => return None,
        };

        if id == b"fmt " && size >= 16 && chunk_start.checked_add(16)? <= chunk_end {
            audio_format = Some(u16::from_le_bytes(
                wav[chunk_start..chunk_start + 2].try_into().ok()?,
            ));
            channels = Some(u16::from_le_bytes(
                wav[chunk_start + 2..chunk_start + 4].try_into().ok()?,
            ));
            sample_rate = Some(u32::from_le_bytes(
                wav[chunk_start + 4..chunk_start + 8].try_into().ok()?,
            ));
            bits_per_sample = Some(u16::from_le_bytes(
                wav[chunk_start + 14..chunk_start + 16].try_into().ok()?,
            ));
        } else if id == b"data" {
            data = Some(&wav[chunk_start..chunk_end]);
        }

        offset = chunk_end.checked_add(size & 1)?;
    }

    let channels = channels?;
    let sample_rate = sample_rate?;
    let bits_per_sample = bits_per_sample?;
    let audio_format = audio_format?;
    let data = data?;
    if audio_format != 1 || channels == 0 || bits_per_sample != 16 {
        return None;
    }

    let block_align = channels as usize * 2;
    let start_frame = (start_seconds.max(0.0) * sample_rate as f64).floor() as usize;
    let start_byte =
        start_frame.saturating_mul(block_align).min(data.len()) / block_align * block_align;
    let source = &data[start_byte..];
    if source.is_empty() {
        return None;
    }

    let volume = volume.clamp(0.0, 1.0);
    let mut pcm = Vec::with_capacity(source.len());
    if volume >= 0.9999 {
        pcm.extend_from_slice(source);
    } else {
        for sample in source.chunks_exact(2) {
            let value = i16::from_le_bytes([sample[0], sample[1]]);
            let scaled = (value as f32 * volume)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            pcm.extend_from_slice(&scaled.to_le_bytes());
        }
    }

    let data_len = u32::try_from(pcm.len()).ok()?;
    let byte_rate = sample_rate.checked_mul(block_align as u32)?;
    let riff_size = 36u32.checked_add(data_len)?;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&pcm);
    Some(out)
}

#[derive(Clone, Copy)]
struct Mpeg2StreamInfo {
    width: usize,
    height: usize,
    fps: f32,
}

fn find_mpeg2_stream_info(usm: &[u8]) -> anyhow::Result<Mpeg2StreamInfo> {
    for chunk in UsmChunks::new(usm) {
        let chunk = chunk?;
        if chunk.kind != *b"@SFV" || chunk.payload_kind != 0 {
            continue;
        }
        if let Some(header) = chunk
            .payload
            .windows(8)
            .find(|bytes| bytes[..4] == *b"\0\0\x01\xB3")
        {
            let width = ((header[4] as usize) << 4) | ((header[5] as usize) >> 4);
            let height = (((header[5] as usize) & 0x0f) << 8) | header[6] as usize;
            let fps = match header[7] & 0x0f {
                1 => 24_000.0 / 1_001.0,
                2 => 24.0,
                3 => 25.0,
                4 => 30_000.0 / 1_001.0,
                5 => 30.0,
                6 => 50.0,
                7 => 60_000.0 / 1_001.0,
                8 => 60.0,
                code => anyhow::bail!("unsupported MPEG-2 frame-rate code {code}"),
            };
            anyhow::ensure!(width > 0 && height > 0, "invalid MPEG-2 dimensions");
            return Ok(Mpeg2StreamInfo { width, height, fps });
        }
    }
    anyhow::bail!("USM contains no MPEG-2 video sequence header")
}

struct UsmChunk<'a> {
    kind: [u8; 4],
    payload_kind: u8,
    payload: &'a [u8],
}

struct UsmChunks<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> UsmChunks<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for UsmChunks<'a> {
    type Item = anyhow::Result<UsmChunk<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.data.len() {
            return None;
        }
        if self.data.len().saturating_sub(self.offset) < 0x20 {
            self.offset = self.data.len();
            return Some(Err(anyhow::anyhow!("truncated USM chunk header")));
        }
        let header = &self.data[self.offset..self.offset + 0x20];
        let kind = header[..4].try_into().unwrap();
        let size_after_prefix = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
        let payload_offset = header[9] as usize;
        let padding = u16::from_be_bytes(header[10..12].try_into().unwrap()) as usize;
        let chunk_end = match self.offset.checked_add(8 + size_after_prefix) {
            Some(end) if end <= self.data.len() => end,
            _ => {
                self.offset = self.data.len();
                return Some(Err(anyhow::anyhow!("USM chunk exceeds file bounds")));
            }
        };
        let payload_start = self.offset + 8 + payload_offset;
        let payload_end = match chunk_end.checked_sub(padding) {
            Some(end) if payload_start <= end => end,
            _ => {
                self.offset = self.data.len();
                return Some(Err(anyhow::anyhow!("invalid USM payload bounds")));
            }
        };
        self.offset = chunk_end;
        Some(Ok(UsmChunk {
            kind,
            payload_kind: header[15] & 3,
            payload: &self.data[payload_start..payload_end],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(kind: [u8; 4], payload_kind: u8, payload: &[u8], padding: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(kind);
        bytes.extend(((0x18 + payload.len() + padding) as u32).to_be_bytes());
        bytes.push(0);
        bytes.push(0x18);
        bytes.extend((padding as u16).to_be_bytes());
        bytes.extend([0, 0, 0, payload_kind]);
        bytes.extend([0; 16]);
        bytes.extend(payload);
        bytes.resize(bytes.len() + padding, 0);
        bytes
    }

    #[test]
    fn parses_usm_chunk_payload_without_padding() {
        let bytes = chunk(*b"@SFV", 0, b"video", 7);
        let parsed = UsmChunks::new(&bytes).next().unwrap().unwrap();
        assert_eq!(parsed.kind, *b"@SFV");
        assert_eq!(parsed.payload_kind, 0);
        assert_eq!(parsed.payload, b"video");
    }

    #[test]
    fn reads_mpeg2_dimensions_and_frame_rate() {
        let bytes = chunk(*b"@SFV", 0, &[0, 0, 1, 0xB3, 0x78, 0x04, 0x38, 0x14], 0);
        let info = find_mpeg2_stream_info(&bytes).unwrap();
        assert_eq!((info.width, info.height), (1920, 1080));
        assert!((info.fps - 29.970).abs() < 0.001);
    }

    #[test]
    fn rejects_truncated_usm_chunks() {
        let mut bytes = chunk(*b"@SFV", 0, b"video", 0);
        bytes.pop();
        assert!(UsmChunks::new(&bytes).next().unwrap().is_err());
    }

    #[test]
    fn counts_picture_start_codes_across_video_chunks() {
        let mut bytes = chunk(*b"@SFV", 0, &[0xaa, 0x00, 0x00], 0);
        bytes.extend(chunk(
            *b"@SFV",
            0,
            &[0x01, 0x00, 0xbb, 0x00, 0x00, 0x01, 0x00],
            0,
        ));
        assert_eq!(count_mpeg2_frames(&bytes).unwrap(), 2);
    }

    #[test]
    fn slices_ffmpeg_streaming_pcm_wav_and_scales_volume() {
        let samples = [1000i16, -1000, 2000, -2000];
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&u32::MAX.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&4u32.to_le_bytes());
        wav.extend_from_slice(&8u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&u32::MAX.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        let sliced = make_pcm_wav_segment(&wav, 0.5, 0.5).unwrap();
        assert_eq!(&sliced[..4], b"RIFF");
        assert_eq!(&sliced[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(sliced[40..44].try_into().unwrap()), 4);
        let first = i16::from_le_bytes(sliced[44..46].try_into().unwrap());
        let second = i16::from_le_bytes(sliced[46..48].try_into().unwrap());
        assert_eq!((first, second), (1000, -1000));
    }

    #[test]
    #[ignore = "requires installed Marathon packages and ffmpeg"]
    fn decodes_first_frame_from_live_marathon_usm() {
        use std::path::PathBuf;
        use tiger_pkg::{GameVersion, MarathonVersion, PackageManager};

        let packages = std::env::var("QUICKTAG_MARATHON_PACKAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(r"D:\SteamLibrary\steamapps\common\Marathon\packages")
            });
        let manager = Arc::new(
            PackageManager::new(
                packages,
                GameVersion::Marathon(MarathonVersion::Marathon),
                None,
            )
            .unwrap(),
        );
        tiger_pkg::initialize_package_manager(&manager);
        let tag = manager
            .get_all_by_type(27, Some(1))
            .iter()
            .min_by_key(|(_, entry)| entry.file_size)
            .map(|(tag, _)| *tag)
            .unwrap();
        let (sender, receiver) = mpsc::sync_channel(2);
        let control = Arc::new(PlaybackControl::new(false));
        let receiver_control = control.clone();
        let receiver_thread = thread::spawn(move || {
            while let Ok(event) = receiver.recv() {
                if let DecoderEvent::Frame(frame) = event {
                    receiver_control.stopped.store(true, Ordering::Release);
                    return (frame.width, frame.height, frame.pixels.len());
                }
            }
            panic!("decoder produced no frame");
        });

        decode_video_inner(
            tag,
            &sender,
            &control,
            Arc::new(Mutex::new(None)),
            0,
            false,
            true,
        )
        .unwrap();
        let (width, height, byte_count) = receiver_thread.join().unwrap();
        assert_eq!(byte_count, width * height * 4);
    }
}
