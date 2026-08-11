use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vibrato::{Dictionary, Tokenizer};

use crate::tokenizer::{split_sentence_spans, AnnotatedToken, AnnotatedTokenSequenceBuilder};

const HOMOPHONE_GROUPS_ASSET: &str = "homophone-groups.jsonl.zst";
const HOMOPHONE_MANIFEST_ASSET: &str = "homophone-manifest.json";
const HOMOPHONE_CHECKSUMS_ASSET: &str = "HOMOPHONE-SHA256SUMS";
const NATURALNESS_POLICY_VERSION: &str = "conservative-v1";
const NATURAL_CONTENT_POS: &[&str] = &["名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞"];
const MAX_NATURAL_SURFACE_CHARACTERS: usize = 24;
const MAX_NATURAL_READING_CHARACTERS: usize = 24;

#[derive(Debug)]
pub struct HomophoneBuildOptions {
    pub inputs: Vec<PathBuf>,
    pub vibrato_dictionary: PathBuf,
    pub output_dir: PathBuf,
    pub min_group_size: usize,
    pub min_candidate_count: u64,
    pub min_natural_occurrences: u64,
    pub min_natural_sentences: u64,
    pub occurrence_shard_records: u64,
    pub vibrato_dictionary_version: String,
    pub pipeline_commit: String,
}

#[derive(Debug, Deserialize)]
struct CorpusRecord {
    source: String,
    document_id: String,
    title: String,
    url: String,
    text: String,
    #[serde(default)]
    annotations: Annotations,
}

#[derive(Debug, Default, Deserialize)]
struct Annotations {
    #[serde(default)]
    ruby: Vec<RubyAnnotation>,
}

#[derive(Debug, Deserialize)]
struct RubyAnnotation {
    start: usize,
    end: usize,
    reading: String,
}

#[derive(Debug, Default)]
struct CandidateStats {
    count: u64,
    lemma_counts: BTreeMap<String, u64>,
    pos_counts: BTreeMap<String, u64>,
    source_counts: BTreeMap<String, u64>,
    reading_source_counts: BTreeMap<String, u64>,
    document_count: u64,
    sentence_count: u64,
    last_document_key: String,
    last_sentence_id: String,
}

#[derive(Debug, Default)]
struct BuildStats {
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
    natural_candidate_tokens: u64,
    token_rejection_reasons: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct CandidateSummary {
    surface: String,
    occurrence_count: u64,
    document_count: u64,
    sentence_count: u64,
    lemma_count: usize,
    dominant_lemma: String,
    lemmas: Vec<String>,
    parts_of_speech: Vec<String>,
    source_counts: BTreeMap<String, u64>,
    reading_source_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct GroupSummary {
    schema_version: u8,
    group_id: String,
    reading: String,
    candidate_count: usize,
    total_occurrences: u64,
    candidates: Vec<CandidateSummary>,
}

#[derive(Debug, Serialize)]
struct Occurrence {
    schema_version: u8,
    group_id: String,
    reading: String,
    reading_source: String,
    surface: String,
    lemma: String,
    pos: String,
    subpos: String,
    subsubpos: String,
    source: String,
    document_id: String,
    title: String,
    url: String,
    sentence_id: String,
    sentence: String,
    sentence_start: usize,
    sentence_end: usize,
    target_start: usize,
    target_end: usize,
    token_index: usize,
    left_context: String,
    right_context: String,
}

#[derive(Debug, Serialize)]
struct AssetMetadata {
    name: String,
    records: u64,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct HomophoneManifest {
    schema_version: u8,
    format: FormatMetadata,
    selection: SelectionMetadata,
    tokenizer: TokenizerMetadata,
    corpus: CorpusMetadata,
    quality: QualityMetadata,
    assets: Vec<AssetMetadata>,
}

#[derive(Debug, Serialize)]
struct FormatMetadata {
    name: &'static str,
    record_type: &'static str,
    encoding: &'static str,
    compression: &'static str,
    offsets: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectionMetadata {
    reading_normalization: &'static str,
    surface_grouping: &'static str,
    min_group_size: usize,
    min_candidate_count: u64,
    candidate_definition: &'static str,
    naturalness: NaturalnessMetadata,
}

#[derive(Debug, Serialize)]
struct NaturalnessMetadata {
    policy: &'static str,
    min_occurrences_per_candidate: u64,
    min_sentences_per_candidate: u64,
    content_pos: Vec<&'static str>,
    sentence_filter: &'static str,
    surface_filter: &'static str,
    variant_filter: &'static str,
    lemma_filter: &'static str,
}

#[derive(Debug, Serialize)]
struct TokenizerMetadata {
    implementation: &'static str,
    implementation_version: &'static str,
    dictionary: &'static str,
    dictionary_version: String,
}

#[derive(Debug, Serialize)]
struct CorpusMetadata {
    input_assets: Vec<InputAssetMetadata>,
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
    natural_candidate_tokens: u64,
    homophone_groups: u64,
    candidate_forms: u64,
    occurrences: u64,
    occurrence_shard_records: u64,
    pipeline_commit: String,
}

#[derive(Debug, Serialize)]
struct InputAssetMetadata {
    name: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct QualityMetadata {
    raw_reading_groups: u64,
    raw_candidate_forms: u64,
    candidate_forms_after_evidence_filter: u64,
    rejected_candidate_forms: u64,
    rejected_groups: u64,
    variant_collapsed_forms: u64,
    selected_groups: u64,
    selected_candidate_forms: u64,
    selected_occurrences: u64,
    candidate_rejection_reasons: BTreeMap<String, u64>,
    group_rejection_reasons: BTreeMap<String, u64>,
    token_rejection_reasons: BTreeMap<String, u64>,
}

pub fn build_homophones(mut options: HomophoneBuildOptions) -> Result<()> {
    validate_options(&mut options)?;
    fs::create_dir_all(&options.output_dir)?;

    let dictionary =
        Dictionary::read(File::open(&options.vibrato_dictionary).with_context(|| {
            format!(
                "opening Vibrato dictionary {}",
                options.vibrato_dictionary.display()
            )
        })?)?;
    let tokenizer = Tokenizer::new(dictionary);
    let sequence_builder = AnnotatedTokenSequenceBuilder::new(&tokenizer);

    eprintln!("phase 1/3: tokenizing input records and collecting reading groups");
    let mut counts: HashMap<String, HashMap<String, CandidateStats>> = HashMap::new();
    let mut stats = BuildStats::default();
    for input in &options.inputs {
        collect_statistics(input, &sequence_builder, &mut counts, &mut stats)?;
    }

    let (groups, mut quality) = build_groups(
        counts,
        options.min_group_size,
        options.min_candidate_count,
        options.min_natural_occurrences,
        options.min_natural_sentences,
    );
    quality.token_rejection_reasons = stats.token_rejection_reasons.clone();
    let input_assets = options
        .inputs
        .iter()
        .map(|path| input_asset_metadata(path))
        .collect::<Result<Vec<_>>>()?;
    let candidate_forms = groups
        .values()
        .map(|group| group.candidate_count as u64)
        .sum::<u64>();
    let group_occurrences = groups
        .values()
        .map(|group| group.total_occurrences)
        .sum::<u64>();

    eprintln!(
        "selected {} homophone groups, {} candidate forms, {} occurrences",
        groups.len(),
        candidate_forms,
        group_occurrences
    );

    eprintln!("phase 2/3: writing group and occurrence assets");
    let groups_path = options.output_dir.join(HOMOPHONE_GROUPS_ASSET);
    write_groups(&groups_path, groups.values())?;

    let (occurrence_assets, occurrences) = write_occurrences(
        &options.inputs,
        &sequence_builder,
        &groups,
        &options.output_dir,
        options.occurrence_shard_records,
    )?;
    if occurrences != group_occurrences {
        bail!(
            "occurrence recount mismatch: group index has {}, output has {}",
            group_occurrences,
            occurrences
        );
    }

    eprintln!("phase 3/3: writing manifest and checksums");
    let mut assets = vec![asset_metadata(&groups_path, groups.len() as u64)?];
    for (path, records) in occurrence_assets {
        assets.push(asset_metadata(&path, records)?);
    }
    let manifest = HomophoneManifest {
        schema_version: 1,
        format: FormatMetadata {
            name: "Japanese homophone context corpus",
            record_type: "JSONL; one group or occurrence per line",
            encoding: "UTF-8",
            compression: "zstd for JSONL assets",
            offsets: "Unicode code-point [start, end) offsets within the source document",
        },
        selection: SelectionMetadata {
            reading_normalization: "Vibrato/IPADIC reading with katakana-to-hiragana conversion",
            surface_grouping: "exact kanji-containing surface form; lemma and IPADIC part of speech are retained",
            min_group_size: options.min_group_size,
            min_candidate_count: options.min_candidate_count,
            candidate_definition: "A surface form with an attested token occurrence",
            naturalness: NaturalnessMetadata {
                policy: NATURALNESS_POLICY_VERSION,
                min_occurrences_per_candidate: options.min_natural_occurrences,
                min_sentences_per_candidate: options.min_natural_sentences,
                content_pos: NATURAL_CONTENT_POS.to_vec(),
                sentence_filter: "reject markup, URL-like, control-heavy, and excessively long contexts",
                surface_filter: "reject OOV, affix/function, mixed ASCII/digit, and non-Japanese noise",
                variant_filter: "keep the most frequent representative when kana/okurigana-only variants share a kanji skeleton",
                lemma_filter: "require distinct dominant dictionary lemmas within each group",
            },
        },
        tokenizer: TokenizerMetadata {
            implementation: "Vibrato",
            implementation_version: "0.5.2",
            dictionary: "IPADIC for Vibrato",
            dictionary_version: options.vibrato_dictionary_version.clone(),
        },
        corpus: CorpusMetadata {
            input_assets,
            documents: stats.documents,
            sentences: stats.sentences,
            valid_tokens: stats.valid_tokens,
            natural_candidate_tokens: stats.natural_candidate_tokens,
            homophone_groups: groups.len() as u64,
            candidate_forms,
            occurrences,
            occurrence_shard_records: options.occurrence_shard_records,
            pipeline_commit: options.pipeline_commit.clone(),
        },
        quality,
        assets,
    };
    let manifest_path = options.output_dir.join(HOMOPHONE_MANIFEST_ASSET);
    write_json(&manifest_path, &manifest)?;
    write_checksums(&options.output_dir, &manifest)?;

    Ok(())
}

fn validate_options(options: &mut HomophoneBuildOptions) -> Result<()> {
    if options.inputs.is_empty() {
        bail!("at least one --input is required");
    }
    options.inputs.sort();
    for input in &options.inputs {
        if !input.is_file() {
            bail!("input does not exist: {}", input.display());
        }
    }
    if !options.vibrato_dictionary.is_file() {
        bail!(
            "Vibrato dictionary does not exist: {}",
            options.vibrato_dictionary.display()
        );
    }
    if options.min_group_size < 2 {
        bail!("--min-group-size must be at least 2");
    }
    if options.min_candidate_count == 0 {
        bail!("--min-candidate-count must be positive");
    }
    if options.min_natural_occurrences == 0 {
        bail!("--min-natural-occurrences must be positive");
    }
    if options.min_natural_sentences == 0 {
        bail!("--min-natural-sentences must be positive");
    }
    if options.occurrence_shard_records == 0 {
        bail!("--occurrence-shard-records must be positive");
    }
    Ok(())
}

fn collect_statistics(
    input: &Path,
    sequence_builder: &AnnotatedTokenSequenceBuilder<'_>,
    counts: &mut HashMap<String, HashMap<String, CandidateStats>>,
    stats: &mut BuildStats,
) -> Result<()> {
    for_each_record(input, |record| {
        stats.documents += 1;
        for (sentence_index, span) in split_sentence_spans(&record.text).into_iter().enumerate() {
            stats.sentences += 1;
            let sentence = &record.text[span.start_byte..span.end_byte];
            let sentence_start = record.text[..span.start_byte].chars().count();
            let sentence_id = format!("{}:s{:06}", record.document_id, sentence_index);
            let sentence_reason = natural_sentence_rejection_reason(sentence);
            for mut token in sequence_builder.tokenize(sentence)? {
                apply_ruby(&record, sentence, sentence_start, &mut token);
                stats.valid_tokens += 1;
                if !contains_kanji(&token.surface) {
                    continue;
                }
                if let Some(reason) =
                    sentence_reason.or_else(|| natural_token_rejection_reason(&token))
                {
                    *stats
                        .token_rejection_reasons
                        .entry(reason.to_owned())
                        .or_default() += 1;
                    continue;
                }
                stats.natural_candidate_tokens += 1;
                let candidates = counts.entry(token.reading.clone()).or_default();
                let candidate = candidates.entry(token.surface.clone()).or_default();
                observe_candidate(
                    candidate,
                    &token,
                    &record.source,
                    &record.document_id,
                    &sentence_id,
                );
            }
        }
        Ok(())
    })
}

fn observe_candidate(
    stats: &mut CandidateStats,
    token: &AnnotatedToken,
    source: &str,
    document_id: &str,
    sentence_id: &str,
) {
    stats.count += 1;
    *stats.lemma_counts.entry(token.lemma.clone()).or_default() += 1;
    let pos = format!("{}/{}/{}", token.pos, token.subpos, token.subsubpos);
    *stats.pos_counts.entry(pos).or_default() += 1;
    *stats.source_counts.entry(source.to_owned()).or_default() += 1;
    *stats
        .reading_source_counts
        .entry(token.reading_source.clone())
        .or_default() += 1;
    let document_key = format!("{source}\u{1f}{document_id}");
    if document_key != stats.last_document_key {
        stats.document_count += 1;
        stats.last_document_key = document_key;
    }
    if sentence_id != stats.last_sentence_id {
        stats.sentence_count += 1;
        stats.last_sentence_id = sentence_id.to_owned();
    }
}

fn build_groups(
    counts: HashMap<String, HashMap<String, CandidateStats>>,
    min_group_size: usize,
    min_candidate_count: u64,
    min_natural_occurrences: u64,
    min_natural_sentences: u64,
) -> (BTreeMap<String, GroupSummary>, QualityMetadata) {
    let mut groups = BTreeMap::new();
    let raw_reading_groups = counts.len() as u64;
    let raw_candidate_forms = counts.values().map(HashMap::len).sum::<usize>() as u64;
    let mut candidate_rejection_reasons = BTreeMap::new();
    let mut group_rejection_reasons = BTreeMap::new();
    let mut rejected_candidate_forms = 0;
    let mut rejected_groups = 0;
    let mut variant_collapsed_forms = 0;
    let mut candidate_forms_after_evidence_filter = 0;
    for (reading, candidates) in counts {
        let mut summaries = Vec::new();
        for (surface, stats) in candidates {
            if stats.count < min_candidate_count {
                rejected_candidate_forms += 1;
                increment_reason(
                    &mut candidate_rejection_reasons,
                    "below_min_candidate_count",
                );
                continue;
            }
            if stats.count < min_natural_occurrences {
                rejected_candidate_forms += 1;
                increment_reason(&mut candidate_rejection_reasons, "insufficient_occurrences");
                continue;
            }
            if stats.sentence_count < min_natural_sentences {
                rejected_candidate_forms += 1;
                increment_reason(&mut candidate_rejection_reasons, "insufficient_sentences");
                continue;
            }
            candidate_forms_after_evidence_filter += 1;
            let dominant_lemma = dominant_lemma(&stats.lemma_counts);
            summaries.push(CandidateSummary {
                surface,
                occurrence_count: stats.count,
                document_count: stats.document_count,
                sentence_count: stats.sentence_count,
                lemma_count: stats.lemma_counts.len(),
                dominant_lemma,
                lemmas: stats.lemma_counts.into_keys().collect(),
                parts_of_speech: stats.pos_counts.into_keys().collect(),
                source_counts: stats.source_counts,
                reading_source_counts: stats.reading_source_counts,
            });
        }
        let (collapsed, count) = collapse_orthographic_variants(summaries);
        summaries = collapsed;
        variant_collapsed_forms += count;
        summaries.sort_by(|left, right| left.surface.cmp(&right.surface));
        if summaries.len() < min_group_size {
            rejected_groups += 1;
            increment_reason(&mut group_rejection_reasons, "too_few_candidates");
            continue;
        }
        if summaries
            .iter()
            .map(|candidate| candidate.dominant_lemma.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            < min_group_size
        {
            rejected_groups += 1;
            increment_reason(&mut group_rejection_reasons, "same_dominant_lemma");
            continue;
        }
        let total_occurrences = summaries
            .iter()
            .map(|candidate| candidate.occurrence_count)
            .sum();
        groups.insert(
            reading.clone(),
            GroupSummary {
                schema_version: 1,
                group_id: format!("homophone:{reading}"),
                reading,
                candidate_count: summaries.len(),
                total_occurrences,
                candidates: summaries,
            },
        );
    }
    let selected_candidate_forms = groups
        .values()
        .map(|group| group.candidate_count as u64)
        .sum();
    let selected_occurrences = groups.values().map(|group| group.total_occurrences).sum();
    let selected_groups = groups.len() as u64;
    (
        groups,
        QualityMetadata {
            raw_reading_groups,
            raw_candidate_forms,
            candidate_forms_after_evidence_filter,
            rejected_candidate_forms,
            rejected_groups,
            variant_collapsed_forms,
            selected_groups,
            selected_candidate_forms,
            selected_occurrences,
            candidate_rejection_reasons,
            group_rejection_reasons,
            token_rejection_reasons: BTreeMap::new(),
        },
    )
}

fn increment_reason(reasons: &mut BTreeMap<String, u64>, reason: &str) {
    *reasons.entry(reason.to_owned()).or_default() += 1;
}

fn dominant_lemma(counts: &BTreeMap<String, u64>) -> String {
    counts
        .iter()
        .max_by(|(left_lemma, left_count), (right_lemma, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_lemma.cmp(left_lemma))
        })
        .map(|(lemma, _)| lemma.clone())
        .unwrap_or_default()
}

fn collapse_orthographic_variants(
    candidates: Vec<CandidateSummary>,
) -> (Vec<CandidateSummary>, u64) {
    let mut by_key: HashMap<String, Vec<CandidateSummary>> = HashMap::new();
    for candidate in candidates {
        by_key
            .entry(orthographic_key(&candidate.surface))
            .or_default()
            .push(candidate);
    }
    let mut selected = Vec::new();
    let mut collapsed = 0;
    for mut variants in by_key.into_values() {
        variants.sort_by(|left, right| {
            right
                .occurrence_count
                .cmp(&left.occurrence_count)
                .then_with(|| right.document_count.cmp(&left.document_count))
                .then_with(|| right.sentence_count.cmp(&left.sentence_count))
                .then_with(|| left.surface.cmp(&right.surface))
        });
        collapsed += variants.len().saturating_sub(1) as u64;
        selected.push(variants.remove(0));
    }
    selected.sort_by(|left, right| left.surface.cmp(&right.surface));
    (selected, collapsed)
}

fn orthographic_key(surface: &str) -> String {
    surface
        .chars()
        .filter(|character| {
            !matches!(
                character,
                '\u{3041}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}'
            )
        })
        .collect()
}

fn contains_kanji(value: &str) -> bool {
    value.chars().any(|character| {
        matches!(
            character,
            '\u{3400}'..='\u{4DBF}'
                | '\u{4E00}'..='\u{9FFF}'
                | '\u{F900}'..='\u{FAFF}'
        )
    })
}

fn natural_token_rejection_reason(token: &AnnotatedToken) -> Option<&'static str> {
    if token.pos == "UNK" {
        return Some("unknown_token");
    }
    if !NATURAL_CONTENT_POS.contains(&token.pos.as_str()) {
        return Some("non_content_pos");
    }
    if token.pos == "名詞" && matches!(token.subpos.as_str(), "数詞" | "形式名詞") {
        return Some("non_lexical_noun");
    }
    if !is_natural_surface(&token.surface) {
        return Some("surface_noise");
    }
    if token.surface.chars().count() > MAX_NATURAL_SURFACE_CHARACTERS {
        return Some("surface_too_long");
    }
    if token.reading.chars().count() > MAX_NATURAL_READING_CHARACTERS {
        return Some("reading_too_long");
    }
    None
}

fn natural_sentence_rejection_reason(sentence: &str) -> Option<&'static str> {
    if sentence.is_empty() || sentence.chars().count() > 2_000 {
        return Some("context_too_long");
    }
    for marker in [
        "http://", "https://", "www.", "[[", "]]", "{{", "}}", "<ref", "</ref>", "ISBN", "==", "|",
    ] {
        if sentence.contains(marker) {
            return Some("noisy_context");
        }
    }
    let ascii_alnum = sentence
        .chars()
        .filter(|character| character.is_ascii() && character.is_ascii_alphanumeric())
        .count();
    if ascii_alnum > 20.max(sentence.chars().count() / 4) {
        return Some("ascii_heavy_context");
    }
    let digits = sentence
        .chars()
        .filter(|character| character.is_ascii_digit())
        .count();
    if digits > 12.max(sentence.chars().count() / 5) {
        return Some("numeric_heavy_context");
    }
    None
}

fn is_natural_surface(surface: &str) -> bool {
    if surface.is_empty() || surface.trim() != surface {
        return false;
    }
    surface.chars().all(|character| {
        matches!(
            character,
            '\u{3041}'..='\u{309F}'
                | '\u{30A0}'..='\u{30FF}'
                | '\u{3400}'..='\u{4DBF}'
                | '\u{4E00}'..='\u{9FFF}'
                | '\u{F900}'..='\u{FAFF}'
                | '々'
                | '〆'
        )
    })
}

fn apply_ruby(
    record: &CorpusRecord,
    sentence: &str,
    sentence_start_chars: usize,
    token: &mut AnnotatedToken,
) {
    let token_start = sentence_start_chars + sentence[..token.start_byte].chars().count();
    let token_end = sentence_start_chars + sentence[..token.end_byte].chars().count();
    if let Some(annotation) = record
        .annotations
        .ruby
        .iter()
        .find(|annotation| annotation.start == token_start && annotation.end == token_end)
    {
        let reading = normalize_reading(&annotation.reading);
        if !reading.is_empty() {
            token.reading = reading;
            token.reading_source = "aozora_ruby".to_owned();
        }
    }
}

fn normalize_reading(reading: &str) -> String {
    reading
        .chars()
        .map(|character| match character {
            'ァ'..='ヶ' => char::from_u32(character as u32 - 0x60).unwrap_or(character),
            'ヽ' => 'ゝ',
            'ヾ' => 'ゞ',
            _ => character,
        })
        .collect()
}

fn write_groups<'a>(path: &Path, groups: impl Iterator<Item = &'a GroupSummary>) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = zstd::stream::write::Encoder::new(file, 3)?;
    for group in groups {
        write_json_line(&mut encoder, group)?;
    }
    encoder.finish()?;
    Ok(())
}

struct OccurrenceShardWriter {
    path: PathBuf,
    encoder: zstd::stream::write::Encoder<'static, File>,
    records: u64,
}

impl OccurrenceShardWriter {
    fn new(output_dir: &Path, shard_index: u64) -> Result<Self> {
        let path = output_dir.join(format!("homophone-occurrences-{shard_index:05}.jsonl.zst"));
        let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self {
            path,
            encoder: zstd::stream::write::Encoder::new(file, 3)?,
            records: 0,
        })
    }

    fn finish(self) -> Result<(PathBuf, u64)> {
        self.encoder.finish()?;
        Ok((self.path, self.records))
    }
}

fn write_occurrences(
    inputs: &[PathBuf],
    sequence_builder: &AnnotatedTokenSequenceBuilder<'_>,
    groups: &BTreeMap<String, GroupSummary>,
    output_dir: &Path,
    shard_records: u64,
) -> Result<(Vec<(PathBuf, u64)>, u64)> {
    let mut shard_index = 0;
    let mut current = Some(OccurrenceShardWriter::new(output_dir, shard_index)?);
    let mut assets = Vec::new();
    let mut occurrence_count = 0u64;

    for input in inputs {
        for_each_record(input, |record| {
            for (sentence_index, span) in split_sentence_spans(&record.text).into_iter().enumerate()
            {
                let sentence = &record.text[span.start_byte..span.end_byte];
                let sentence_start = record.text[..span.start_byte].chars().count();
                let sentence_end = sentence_start + sentence.chars().count();
                let sentence_id = format!("{}:s{:06}", record.document_id, sentence_index);
                let sentence_reason = natural_sentence_rejection_reason(sentence);
                let tokens = sequence_builder.tokenize(sentence)?;
                for (token_index, mut token) in tokens.into_iter().enumerate() {
                    apply_ruby(&record, sentence, sentence_start, &mut token);
                    if !contains_kanji(&token.surface)
                        || sentence_reason.is_some()
                        || natural_token_rejection_reason(&token).is_some()
                    {
                        continue;
                    }
                    let Some(group) = groups.get(&token.reading) else {
                        continue;
                    };
                    if !group
                        .candidates
                        .iter()
                        .any(|candidate| candidate.surface == token.surface)
                    {
                        continue;
                    }
                    if current
                        .as_ref()
                        .is_some_and(|shard| shard.records >= shard_records)
                    {
                        let finished = current.take().expect("occurrence shard exists").finish()?;
                        assets.push(finished);
                        shard_index += 1;
                        current = Some(OccurrenceShardWriter::new(output_dir, shard_index)?);
                    }
                    let target_start =
                        sentence_start + sentence[..token.start_byte].chars().count();
                    let target_end = sentence_start + sentence[..token.end_byte].chars().count();
                    let occurrence = Occurrence {
                        schema_version: 1,
                        group_id: group.group_id.clone(),
                        reading: token.reading,
                        reading_source: token.reading_source,
                        surface: token.surface,
                        lemma: token.lemma,
                        pos: token.pos,
                        subpos: token.subpos,
                        subsubpos: token.subsubpos,
                        source: record.source.clone(),
                        document_id: record.document_id.clone(),
                        title: record.title.clone(),
                        url: record.url.clone(),
                        sentence_id: sentence_id.clone(),
                        sentence: sentence.to_owned(),
                        sentence_start,
                        sentence_end,
                        target_start,
                        target_end,
                        token_index,
                        left_context: sentence[..token.start_byte].to_owned(),
                        right_context: sentence[token.end_byte..].to_owned(),
                    };
                    write_json_line(
                        &mut current.as_mut().expect("occurrence shard exists").encoder,
                        &occurrence,
                    )?;
                    current.as_mut().expect("occurrence shard exists").records += 1;
                    occurrence_count += 1;
                }
            }
            Ok(())
        })?;
    }
    if let Some(shard) = current.take() {
        if shard.records > 0 {
            assets.push(shard.finish()?);
        }
    }
    Ok((assets, occurrence_count))
}

fn for_each_record(path: &Path, mut consume: impl FnMut(CorpusRecord) -> Result<()>) -> Result<()> {
    let reader = open_input(path)?;
    for (line_number, line) in reader.lines().enumerate() {
        let line_number = line_number + 1;
        let line =
            line.with_context(|| format!("reading {} line {}", path.display(), line_number))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: CorpusRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing {} line {}", path.display(), line_number))?;
        consume(record)?;
    }
    Ok(())
}

fn open_input(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if path.extension().is_some_and(|extension| extension == "zst") {
        let decoder = zstd::stream::read::Decoder::new(BufReader::new(file))?;
        return Ok(Box::new(BufReader::new(decoder)));
    }
    Ok(Box::new(BufReader::new(file)))
}

fn write_json_line<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn asset_metadata(path: &Path, records: u64) -> Result<AssetMetadata> {
    let metadata = fs::metadata(path)?;
    Ok(AssetMetadata {
        name: path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned(),
        records,
        bytes: metadata.len(),
        sha256: sha256_file(path)?,
    })
}

fn input_asset_metadata(path: &Path) -> Result<InputAssetMetadata> {
    Ok(InputAssetMetadata {
        name: path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned(),
        bytes: fs::metadata(path)?.len(),
        sha256: sha256_file(path)?,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let bytes = std::io::Read::read(&mut file, &mut buffer)?;
        if bytes == 0 {
            break;
        }
        digest.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_checksums(output_dir: &Path, manifest: &HomophoneManifest) -> Result<()> {
    let path = output_dir.join(HOMOPHONE_CHECKSUMS_ASSET);
    let mut file = File::create(path)?;
    for asset in &manifest.assets {
        writeln!(file, "{}  {}", asset.sha256, asset.name)?;
    }
    let manifest_path = output_dir.join(HOMOPHONE_MANIFEST_ASSET);
    writeln!(
        file,
        "{}  {}",
        sha256_file(&manifest_path)?,
        HOMOPHONE_MANIFEST_ASSET
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_require_distinct_surface_forms() {
        let mut counts = HashMap::new();
        let mut candidates = HashMap::new();
        candidates.insert(
            "交渉".to_owned(),
            CandidateStats {
                count: 2,
                document_count: 1,
                sentence_count: 1,
                lemma_counts: BTreeMap::from([(String::from("交渉"), 2)]),
                ..CandidateStats::default()
            },
        );
        candidates.insert(
            "高尚".to_owned(),
            CandidateStats {
                count: 1,
                document_count: 1,
                sentence_count: 1,
                lemma_counts: BTreeMap::from([(String::from("高尚"), 1)]),
                ..CandidateStats::default()
            },
        );
        counts.insert("こうしょう".to_owned(), candidates);

        let (groups, _) = build_groups(counts, 2, 1, 1, 1);
        assert_eq!(groups["こうしょう"].candidate_count, 2);
        assert_eq!(groups["こうしょう"].total_occurrences, 3);
    }

    #[test]
    fn candidate_frequency_filter_is_applied_before_grouping() {
        let mut counts = HashMap::new();
        let mut candidates = HashMap::new();
        candidates.insert(
            "交渉".to_owned(),
            CandidateStats {
                count: 2,
                ..CandidateStats::default()
            },
        );
        candidates.insert(
            "高尚".to_owned(),
            CandidateStats {
                count: 1,
                ..CandidateStats::default()
            },
        );
        counts.insert("こうしょう".to_owned(), candidates);

        let (groups, _) = build_groups(counts, 2, 2, 1, 1);
        assert!(groups.is_empty());
    }
}
