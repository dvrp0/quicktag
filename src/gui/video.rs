use crate::gui::common::tag_context;
use crate::gui::{View, ViewAction};
use crate::util::format_file_size;
use eframe::egui;
use eframe::egui::{ColorImage, Key, TextureHandle, TextureOptions};
use eframe::wgpu::naga::FastIndexMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
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
    fn new() -> Self {
        Self {
            stopped: AtomicBool::new(false),
            paused: AtomicBool::new(false),
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
    frame_index: u64,
    finished: bool,
    error: Option<String>,
}

impl VideoPlayback {
    fn spawn(tag: TagHash) -> Self {
        let (sender, receiver) = mpsc::sync_channel(2);
        let control = Arc::new(PlaybackControl::new());
        let worker_control = control.clone();
        thread::Builder::new()
            .name(format!("video-{tag}"))
            .spawn(move || decode_video(tag, sender, worker_control))
            .expect("failed to spawn video decoder");

        Self {
            tag,
            receiver,
            control,
            texture: None,
            dimensions: None,
            fps: None,
            frame_index: 0,
            finished: false,
            error: None,
        }
    }

    fn stop(&self) {
        self.control.stopped.store(true, Ordering::Release);
    }

    fn is_paused(&self) -> bool {
        self.control.paused.load(Ordering::Acquire)
    }

    fn toggle_pause(&self) {
        self.control.paused.fetch_xor(true, Ordering::AcqRel);
    }

    fn update(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                DecoderEvent::Metadata { width, height, fps } => {
                    self.dimensions = Some((width, height));
                    self.fps = Some(fps);
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
                }
                DecoderEvent::Finished => self.finished = true,
                DecoderEvent::Error(error) => self.error = Some(error),
            }
        }
    }
}

impl Drop for VideoPlayback {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct VideoView {
    selected_package: u16,
    selected_videos: Option<PackageVideos>,
    packages: FastIndexMap<u16, PackagePath>,
    current_row: usize,
    playback: Option<VideoPlayback>,
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
        }
    }

    fn select_package(&mut self, id: u16) {
        self.stop_playback();
        self.selected_package = id;
        self.selected_videos = Some(PackageVideos::by_pkg_id(id));
        self.current_row = 0;
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
        self.playback = Some(VideoPlayback::spawn(tag));
    }

    fn stop_playback(&mut self) {
        if let Some(playback) = self.playback.take() {
            playback.stop();
        }
    }
}

impl View for VideoView {
    fn view(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) -> Option<ViewAction> {
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
                if !entries.is_empty() {
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

        if let Some(playback) = &mut self.playback {
            playback.update(ctx);
            let mut restart = false;
            ui.horizontal(|ui| {
                if ui
                    .button(if playback.is_paused() {
                        "▶ Play"
                    } else {
                        "⏸ Pause"
                    })
                    .clicked()
                {
                    playback.toggle_pause();
                }
                restart = ui.button("↻ Restart").clicked();
                ui.separator();
                ui.monospace(playback.tag.to_string());
                if let (Some((width, height)), Some(fps)) = (playback.dimensions, playback.fps) {
                    ui.label(format!(
                        "{width}×{height} · {fps:.3} fps · frame {}",
                        playback.frame_index
                    ));
                }
            });

            ui.separator();
            if let Some(error) = &playback.error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            } else if let Some(texture) = &playback.texture {
                let available = ui.available_size();
                let source = texture.size_vec2();
                let scale = (available.x / source.x)
                    .min(available.y / source.y)
                    .min(1.0)
                    .max(0.0);
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(source * scale));
                });
                if !playback.finished && !playback.is_paused() {
                    ctx.request_repaint_after(Duration::from_millis(8));
                }
            } else {
                ui.spinner();
                ui.label("Reading USM and starting decoder…");
                ctx.request_repaint_after(Duration::from_millis(16));
            }

            if restart {
                let tag = playback.tag;
                self.stop_playback();
                self.playback = Some(VideoPlayback::spawn(tag));
            }
        } else {
            ui.centered_and_justified(|ui| ui.label("Select a video to decode"));
        }

        None
    }
}

fn decode_video(
    tag: TagHash,
    sender: mpsc::SyncSender<DecoderEvent>,
    control: Arc<PlaybackControl>,
) {
    if let Err(error) = decode_video_inner(tag, &sender, &control) {
        let _ = sender.send(DecoderEvent::Error(format!(
            "Video decode failed: {error:#}"
        )));
    }
}

fn decode_video_inner(
    tag: TagHash,
    sender: &mpsc::SyncSender<DecoderEvent>,
    control: &Arc<PlaybackControl>,
) -> anyhow::Result<()> {
    let usm = package_manager().read_tag(tag)?;
    let stream_info = find_mpeg2_stream_info(&usm)?;
    if sender
        .send(DecoderEvent::Metadata {
            width: stream_info.width,
            height: stream_info.height,
            fps: stream_info.fps,
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
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
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
    let mut child = command.spawn().map_err(|error| {
        anyhow::anyhow!("could not start ffmpeg ({error}). Install ffmpeg and add it to PATH")
    })?;
    let mut stdin = child.stdin.take().unwrap();
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
    let mut frame_index = 0u64;
    let frame_interval = Duration::from_secs_f64(1.0 / stream_info.fps as f64);
    let mut next_frame_at = Instant::now();
    loop {
        if control.stopped.load(Ordering::Acquire) {
            let _ = child.kill();
            break;
        }
        while control.paused.load(Ordering::Acquire) {
            if control.stopped.load(Ordering::Acquire) {
                let _ = child.kill();
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        if control.stopped.load(Ordering::Acquire) {
            break;
        }

        let mut pixels = vec![0u8; frame_size];
        match stdout.read_exact(&mut pixels) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        while Instant::now() < next_frame_at && !control.stopped.load(Ordering::Acquire) {
            thread::sleep(
                next_frame_at
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(8)),
            );
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
        frame_index += 1;
        next_frame_at = Instant::now() + frame_interval;
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
        let control = Arc::new(PlaybackControl::new());
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

        decode_video_inner(tag, &sender, &control).unwrap();
        let (width, height, byte_count) = receiver_thread.join().unwrap();
        assert_eq!(byte_count, width * height * 4);
    }
}
