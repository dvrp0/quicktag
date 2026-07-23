use std::{sync::Arc, u32};

use eframe::egui::{self, Color32, RichText, TextEdit, Widget};
use quicktag_core::tagtypes::TagType;
use quicktag_scanner::{
    TagCache,
    signatures::{SIGNATURE_LIST, Signature},
};
use tiger_pkg::{TagHash, package_manager};

use super::{View, ViewAction, common::ResponseExt, tag::format_tag_entry};

pub struct SignaturesView {
    cache: Arc<TagCache>,
    signatures: Vec<(Signature, String)>,
    signatures_filtered: Vec<(Signature, String)>,

    selected_signature: Signature,
    selected_entries: Vec<(TagHash, String, TagType)>,
    filter: String,
    update_search: bool,
    search_by_hash: bool,
    exact_match: bool,
}

impl SignaturesView {
    pub fn new(cache: Arc<TagCache>) -> Self {
        let mut signatures: Vec<(Signature, String)> = SIGNATURE_LIST
            .load()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        signatures.sort_by_key(|(_, s)| s.clone());
        Self {
            cache,
            signatures_filtered: signatures.clone(),
            signatures,
            selected_signature: Signature::U32(u32::MAX),
            selected_entries: vec![],
            filter: String::new(),
            update_search: true,
            search_by_hash: false,
            exact_match: false,
        }
    }

    fn reset_search(&mut self) {
        self.signatures_filtered = self.signatures.clone();
    }

    fn filter_strings(&mut self) {
        if !self.update_search {
            return;
        }

        if self.filter.is_empty() {
            self.reset_search();
        } else {
            self.filter_signatures();
        }

        self.update_search = false;
    }

    fn filter_signatures(&mut self) {
        let match_b = self.filter.to_lowercase();

        self.signatures_filtered = self
            .signatures
            .iter()
            .filter(|(_, s)| {
                let match_a = s.to_lowercase();

                if self.exact_match {
                    match_a == match_b
                } else {
                    match_a.contains(&match_b)
                }
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
    }
}

impl View for SignaturesView {
    fn view(
        &mut self,
        _ctx: &eframe::egui::Context,
        ui: &mut eframe::egui::Ui,
    ) -> Option<super::ViewAction> {
        self.filter_strings();
        egui::SidePanel::left("signatures_left_panel")
            .resizable(true)
            .min_width(384.0)
            .show_inside(ui, |ui| {
                ui.separator();
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    let mut text_edit =
                        TextEdit::singleline(&mut self.filter).hint_text("Enter search term");

                    let input_response = text_edit.ui(ui);

                    self.update_search |= input_response.changed();
                    self.update_search |=
                        ui.checkbox(&mut self.exact_match, "Exact match").changed();
                });

                let string_height = {
                    let s = ui.spacing();
                    s.interact_size.y
                };

                egui::ScrollArea::vertical()
                    .max_width(ui.available_width() * 0.70)
                    .auto_shrink([false, false])
                    .show_rows(
                        ui,
                        string_height,
                        self.signatures_filtered.len(),
                        |ui, range| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            for (sig, string) in &self.signatures_filtered[range] {
                                let response = ui.selectable_value(
                                    &mut self.selected_signature,
                                    sig.clone(),
                                    RichText::new(format!("'{}' {:X?}", string, sig,))
                                        .color(Color32::GREEN),
                                );

                                if response.clicked() {
                                    self.selected_entries.clear();
                                    for (tag, _) in self.cache.hashes.iter().filter(|(_, scan)| {
                                        scan.signatures.iter().any(|s| s.hash == *sig)
                                    }) {
                                        if let Some(e) = package_manager().get_entry(*tag) {
                                            let label = format_tag_entry(*tag, Some(&e));

                                            self.selected_entries.push((
                                                *tag,
                                                label,
                                                TagType::from_type_subtype(
                                                    e.file_type,
                                                    e.file_subtype,
                                                ),
                                            ));
                                        }
                                    }
                                }
                            }
                        },
                    );
            });

        egui::CentralPanel::default()
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_width(f32::INFINITY)
                    .show(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                        if self.selected_signature == Signature::U32(u32::MAX) {
                            ui.label(RichText::new("No string selected").italics());
                        } else {
                            for (tag, label, tag_type) in &self.selected_entries {
                                if ui
                                    .add(egui::Button::selectable(
                                        false,
                                        RichText::new(label).color(tag_type.display_color()),
                                    ))
                                    .tag_context(*tag)
                                    .clicked()
                                {
                                    return Some(ViewAction::OpenTag(*tag));
                                }
                            }
                        }
                        None
                    })
                    .inner
            })
            .inner
    }
}
