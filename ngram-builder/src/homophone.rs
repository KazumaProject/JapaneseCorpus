use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vibrato::{Dictionary, Tokenizer};

use crate::tokenizer::{
    split_sentence_spans, AnnotatedToken, AnnotatedTokenSequenceBuilder,
};

const HOMOPHONE_GROUPS_ASSET: &str = "homophone-groups.jsonl.zst";
const HOMOPHONE_OCCURRENCES_ASSET: &str = "homophone-occurrences.jsonl.zst";
const HOMOPHONE_MANIFEST_ASSET: &str = "homophone-manifest.json";
const HOMOPHONE_CHECKSUMS_ASSET: &str = "HOMOPHONE-SHA256SUMS";

#[derive(Debug)]
pub struct HomophoneBuildOptions {
    pub inputs: Vec<PathBuf>,
    pub vibrato_dictionary: PathBuf,
    pub output_dir: PathBuf,
    pub min_group_size: usize,
    pub min_candidate_count: u64,
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
}

#[derive(Debug, Default)]
struct CandidateStats {
    count: u64,
    lemma_counts: BTreeMap<String, u64>,
    pos_counts: BTreeMap<String, u64>,
    source_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Default)]
struct BuildStats {
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
}

#[derive(Debug, Serialize)]
struct CandidateSummary {
    surface: String,
    occurrence_count: u64,
    lemmas: Vec<String>,
    parts_of_speech: Vec<String>,
    source_counts: BTreeMap<String, u64>,
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
    input_assets: Vec<String>,
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
    homophone_groups: u64,
    candidate_forms: u64,
    occurrences: u64,
    pipeline_commit: String,
}

pub fn build_homophones(mut options: HomophoneBuildOptions) -> Result<()> {
    validate_options(&mut options)?;
    fs::create_dir_all(&options.output_dir)?;

    let dictionary = Dictionary::read(
        File::open(&options.vibrato_dictionary).with_context(|| {
            format!(
                "opening Vibrato dictionary {}",
                options.vibrato_dictionary.display()
            )
        })?,
    )?;
    let tokenizer = Tokenizer::new(dictionary);
    let sequence_builder = AnnotatedTokenSequenceBuilder::new(&tokenizer);

    eprintln!("phase 1/3: tokenizing input records and collecting reading groups");
    let mut counts: HashMap<String, HashMap<String, CandidateStats>> = HashMap::new();
    let mut stats = BuildStats::default();
    for input in &options.inputs {
        collect_statistics(input, &sequence_builder, &mut counts, &mut stats)?;
    }

    let groups = build_groups(counts, options.min_group_size, options.min_candidate_count);
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

    let occurrences_path = options.output_dir.join(HOMOPHONE_OCCURRENCES_ASSET);
    let occurrences = write_occurrences(
        &options.inputs,
        &sequence_builder,
        &groups,
        &occurrences_path,
    )?;
    if occurrences != group_occurrences {
        bail!(
            "occurrence recount mismatch: group index has {}, output has {}",
            group_occurrences,
            occurrences
        );
    }

    eprintln!("phase 3/3: writing manifest and checksums");
    let assets = vec![
        asset_metadata(&groups_path, groups.len() as u64)?,
        asset_metadata(&occurrences_path, occurrences)?,
    ];
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
        },
        tokenizer: TokenizerMetadata {
            implementation: "Vibrato",
            implementation_version: "0.5.2",
            dictionary: "IPADIC for Vibrato",
            dictionary_version: options.vibrato_dictionary_version.clone(),
        },
        corpus: CorpusMetadata {
            input_assets: options
                .inputs
                .iter()
                .map(|path| {
                    path.file_name()
                        .unwrap_or(path.as_os_str())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect(),
            documents: stats.documents,
            sentences: stats.sentences,
            valid_tokens: stats.valid_tokens,
            homophone_groups: groups.len() as u64,
            candidate_forms,
            occurrences,
            pipeline_commit: options.pipeline_commit.clone(),
        },
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
        for span in split_sentence_spans(&record.text) {
            stats.sentences += 1;
            let sentence = &record.text[span.start_byte..span.end_byte];
            for token in sequence_builder.tokenize(sentence)? {
                stats.valid_tokens += 1;
                let candidates = counts.entry(token.reading.clone()).or_default();
                let candidate = candidates.entry(token.surface.clone()).or_default();
                observe_candidate(candidate, &token, &record.source);
            }
        }
        Ok(())
    })
}

fn observe_candidate(stats: &mut CandidateStats, token: &AnnotatedToken, source: &str) {
    stats.count += 1;
    *stats.lemma_counts.entry(token.lemma.clone()).or_default() += 1;
    let pos = format!("{}/{}/{}", token.pos, token.subpos, token.subsubpos);
    *stats.pos_counts.entry(pos).or_default() += 1;
    *stats.source_counts.entry(source.to_owned()).or_default() += 1;
}

fn build_groups(
    counts: HashMap<String, HashMap<String, CandidateStats>>,
    min_group_size: usize,
    min_candidate_count: u64,
) -> BTreeMap<String, GroupSummary> {
    let mut groups = BTreeMap::new();
    for (reading, candidates) in counts {
        let mut summaries = Vec::new();
        for (surface, stats) in candidates {
            if !contains_kanji(&surface) {
                continue;
            }
            if stats.count < min_candidate_count {
                continue;
            }
            summaries.push(CandidateSummary {
                surface,
                occurrence_count: stats.count,
                lemmas: stats.lemma_counts.into_keys().collect(),
                parts_of_speech: stats.pos_counts.into_keys().collect(),
                source_counts: stats.source_counts,
            });
        }
        summaries.sort_by(|left, right| left.surface.cmp(&right.surface));
        if summaries.len() < min_group_size {
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
    groups
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

fn write_groups<'a>(path: &Path, groups: impl Iterator<Item = &'a GroupSummary>) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = zstd::stream::write::Encoder::new(file, 3)?;
    for group in groups {
        write_json_line(&mut encoder, group)?;
    }
    encoder.finish()?;
    Ok(())
}

fn write_occurrences(
    inputs: &[PathBuf],
    sequence_builder: &AnnotatedTokenSequenceBuilder<'_>,
    groups: &BTreeMap<String, GroupSummary>,
    path: &Path,
) -> Result<u64> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = zstd::stream::write::Encoder::new(file, 3)?;
    let mut occurrence_count = 0u64;

    for input in inputs {
        for_each_record(input, |record| {
            for (sentence_index, span) in split_sentence_spans(&record.text).into_iter().enumerate() {
                let sentence = &record.text[span.start_byte..span.end_byte];
                let sentence_start = record.text[..span.start_byte].chars().count();
                let sentence_end = sentence_start + sentence.chars().count();
                let sentence_id = format!("{}:s{:06}", record.document_id, sentence_index);
                let tokens = sequence_builder.tokenize(sentence)?;
                for (token_index, token) in tokens.into_iter().enumerate() {
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
                    let target_start = sentence_start + sentence[..token.start_byte].chars().count();
                    let target_end = sentence_start + sentence[..token.end_byte].chars().count();
                    let occurrence = Occurrence {
                        schema_version: 1,
                        group_id: group.group_id.clone(),
                        reading: token.reading,
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
                    write_json_line(&mut encoder, &occurrence)?;
                    occurrence_count += 1;
                }
            }
            Ok(())
        })?;
    }
    encoder.finish()?;
    Ok(occurrence_count)
}

fn for_each_record(
    path: &Path,
    mut consume: impl FnMut(CorpusRecord) -> Result<()>,
) -> Result<()> {
    let reader = open_input(path)?;
    for (line_number, line) in reader.lines().enumerate() {
        let line_number = line_number + 1;
        let line = line.with_context(|| format!("reading {} line {}", path.display(), line_number))?;
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

        let groups = build_groups(counts, 2, 1);
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

        assert!(build_groups(counts, 2, 2).is_empty());
    }
}
