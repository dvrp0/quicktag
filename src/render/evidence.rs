use tiger_pkg::TagHash;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceLevel {
    Unknown,
    Heuristic,
    Probable,
    StronglyCorrelated,
    Confirmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    pub tag: TagHash,
    pub offset: u64,
    pub size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub evidence: EvidenceLevel,
    pub source_spans: Vec<SourceSpan>,
    pub technique: Option<TagHash>,
    pub shader_stage: Option<&'static str>,
    pub notes: Vec<String>,
}

impl Default for ProvenanceRecord {
    fn default() -> Self {
        Self {
            evidence: EvidenceLevel::Unknown,
            source_spans: vec![],
            technique: None,
            shader_stage: None,
            notes: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProvenanceId(pub u32);

#[derive(Debug, Clone, Default)]
pub struct ProvenanceStore {
    records: Vec<ProvenanceRecord>,
}

impl ProvenanceStore {
    pub fn insert(&mut self, record: ProvenanceRecord) -> ProvenanceId {
        if let Some(index) = self.records.iter().position(|existing| existing == &record) {
            return ProvenanceId(index as u32);
        }
        let id = ProvenanceId(self.records.len() as u32);
        self.records.push(record);
        id
    }

    pub fn get(&self, id: ProvenanceId) -> Option<&ProvenanceRecord> {
        self.records.get(id.0 as usize)
    }
}
