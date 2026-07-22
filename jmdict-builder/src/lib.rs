use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use quick_xml::encoding::Decoder;
use quick_xml::escape::unescape;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use serde::Serialize;
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

const MAX_FIELD_BYTES: usize = 256;
const MANIFEST_NAME: &str = "english-dictionary-manifest.json";
const CHECKSUMS_NAME: &str = "ENGLISH-DICTIONARY-SHA256SUMS";

#[derive(Debug)]
pub struct BuildOptions {
    pub input: PathBuf,
    pub mozc_id_def: PathBuf,
    pub jmdict_license: PathBuf,
    pub output_dir: PathBuf,
    pub entries_per_shard: usize,
    pub base_cost: i32,
    pub source_url: String,
    pub source_etag: String,
    pub source_last_modified: String,
    pub pipeline_commit: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OutputEntry {
    reading: String,
    surface: String,
    cost: i32,
}

#[derive(Default)]
struct RawEntry {
    sequence: String,
    readings: Vec<RawReading>,
    entry_sources: Vec<String>,
    senses: Vec<RawSense>,
}

#[derive(Default)]
struct RawReading {
    text: String,
    priorities: Vec<String>,
}

#[derive(Default)]
struct RawSense {
    restrictions: BTreeSet<String>,
    sources: Vec<String>,
    glosses: Vec<String>,
}

#[derive(Clone, Copy)]
enum TextTarget {
    Sequence,
    Reading,
    Priority,
    Restriction,
    Gloss,
    LanguageSource,
}

impl TextTarget {
    fn element(self) -> &'static [u8] {
        match self {
            Self::Sequence => b"ent_seq",
            Self::Reading => b"reb",
            Self::Priority => b"re_pri",
            Self::Restriction => b"stagr",
            Self::Gloss => b"gloss",
            Self::LanguageSource => b"lsource",
        }
    }
}

#[derive(Default)]
struct ParseStats {
    source_created: String,
    jmdict_entries: u64,
    katakana_entries: u64,
    unique_readings: BTreeSet<String>,
}

#[derive(Clone, Debug, Serialize)]
struct Asset {
    name: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    entries: Option<u64>,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u8,
    format: Format,
    source: Source<'a>,
    parameters: Parameters,
    counts: Counts,
    build: BuildMetadata<'a>,
    assets: &'a [Asset],
}

#[derive(Serialize)]
struct Format {
    name: &'static str,
    columns: [&'static str; 5],
    encoding: &'static str,
    compression: &'static str,
    reading_normalization: &'static str,
}

#[derive(Serialize)]
struct Source<'a> {
    name: &'static str,
    url: &'a str,
    created: &'a str,
    etag: &'a str,
    last_modified: &'a str,
    bytes: u64,
    sha256: &'a str,
    license: &'static str,
}

#[derive(Serialize)]
struct Parameters {
    reading_selection: &'static str,
    translation_language: &'static str,
    include_full_english_lsource: bool,
    excluded_gloss_types: [&'static str; 1],
    base_cost: i32,
    entries_per_shard: usize,
}

#[derive(Serialize)]
struct Counts {
    jmdict_entries: u64,
    katakana_entries: u64,
    unique_readings: u64,
    retained_entries: u64,
}

#[derive(Serialize)]
struct BuildMetadata<'a> {
    pipeline_commit: &'a str,
}

pub fn build(options: BuildOptions) -> Result<()> {
    validate_options(&options)?;
    fs::create_dir_all(&options.output_dir)?;

    let (source_bytes, source_sha256) = file_metadata(&options.input)?;
    let context_id = generic_noun_id(&options.mozc_id_def)?;
    let (entries, stats) = parse_jmdict(&options.input, options.base_cost)?;
    if entries.is_empty() {
        bail!("JMdict produced no katakana-to-English entries");
    }

    let mut assets = write_dictionary_shards(
        &options.output_dir,
        &entries,
        context_id,
        options.entries_per_shard,
    )?;

    let source_name = source_asset_name(&options.input, &stats.source_created, &source_sha256);
    let source_output = options.output_dir.join(source_name);
    fs::copy(&options.input, &source_output)?;
    assets.push(asset_for_file(&source_output, "source", None)?);

    let license_output = options.output_dir.join("JMDICT-LICENSE.html");
    fs::copy(&options.jmdict_license, &license_output)?;
    assets.push(asset_for_file(&license_output, "license", None)?);

    let readme_output = options.output_dir.join("ENGLISH-DICTIONARY-README.md");
    fs::write(&readme_output, dictionary_readme())?;
    assets.push(asset_for_file(&readme_output, "documentation", None)?);
    assets.sort_unstable_by(|left, right| left.name.cmp(&right.name));

    let manifest = Manifest {
        schema_version: 1,
        format: Format {
            name: "Mozc system dictionary source",
            columns: ["reading", "left_id", "right_id", "cost", "surface"],
            encoding: "UTF-8/LF",
            compression: "Zstandard",
            reading_normalization: "Unicode NFKC followed by katakana-to-hiragana conversion",
        },
        source: Source {
            name: "JMdict_e",
            url: &options.source_url,
            created: &stats.source_created,
            etag: &options.source_etag,
            last_modified: &options.source_last_modified,
            bytes: source_bytes,
            sha256: &source_sha256,
            license: "CC-BY-SA-4.0",
        },
        parameters: Parameters {
            reading_selection: "readings consisting entirely of Unicode katakana-block characters",
            translation_language: "eng",
            include_full_english_lsource: true,
            excluded_gloss_types: ["expl"],
            base_cost: options.base_cost,
            entries_per_shard: options.entries_per_shard,
        },
        counts: Counts {
            jmdict_entries: stats.jmdict_entries,
            katakana_entries: stats.katakana_entries,
            unique_readings: stats.unique_readings.len() as u64,
            retained_entries: entries.len() as u64,
        },
        build: BuildMetadata {
            pipeline_commit: &options.pipeline_commit,
        },
        assets: &assets,
    };

    let manifest_path = options.output_dir.join(MANIFEST_NAME);
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    fs::write(&manifest_path, manifest_bytes)?;

    let manifest_asset = asset_for_file(&manifest_path, "metadata", None)?;
    let checksums_path = options.output_dir.join(CHECKSUMS_NAME);
    let mut checksum_entries = assets.clone();
    checksum_entries.push(manifest_asset);
    checksum_entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let mut checksums = File::create(checksums_path)?;
    for asset in checksum_entries {
        writeln!(checksums, "{}  {}", asset.sha256, asset.name)?;
    }

    eprintln!(
        "built {} hiragana-to-English entries from {} katakana JMdict entries",
        entries.len(),
        stats.katakana_entries
    );
    Ok(())
}

fn validate_options(options: &BuildOptions) -> Result<()> {
    for (label, path) in [
        ("JMdict input", &options.input),
        ("Mozc id.def", &options.mozc_id_def),
        ("JMdict licence", &options.jmdict_license),
    ] {
        if !path.is_file() {
            bail!("{label} does not exist: {}", path.display());
        }
    }
    if options.entries_per_shard == 0 {
        bail!("--entries-per-shard must be positive");
    }
    if !(1..=32_000).contains(&options.base_cost) {
        bail!("--base-cost must be between 1 and 32000");
    }
    if options.source_url.trim().is_empty() {
        bail!("--source-url must not be empty");
    }
    Ok(())
}

fn parse_jmdict(path: &Path, base_cost: i32) -> Result<(Vec<OutputEntry>, ParseStats)> {
    let input = open_input(path)?;
    let mut reader = Reader::from_reader(input);
    reader.config_mut().trim_text(false);

    let mut buffer = Vec::new();
    let mut entry: Option<RawEntry> = None;
    let mut reading: Option<RawReading> = None;
    let mut sense: Option<RawSense> = None;
    let mut target: Option<TextTarget> = None;
    let mut target_text = String::new();
    let mut entries = BTreeMap::<(String, String), i32>::new();
    let mut stats = ParseStats::default();

    loop {
        match reader.read_event_into(&mut buffer)? {
            Event::Start(start) => match start.name().as_ref() {
                b"entry" => entry = Some(RawEntry::default()),
                b"r_ele" => reading = Some(RawReading::default()),
                b"sense" => sense = Some(RawSense::default()),
                b"ent_seq" => begin_text(&mut target, &mut target_text, TextTarget::Sequence),
                b"reb" => begin_text(&mut target, &mut target_text, TextTarget::Reading),
                b"re_pri" => begin_text(&mut target, &mut target_text, TextTarget::Priority),
                b"stagr" => begin_text(&mut target, &mut target_text, TextTarget::Restriction),
                b"gloss" if english_gloss(&start, reader.decoder())? => {
                    begin_text(&mut target, &mut target_text, TextTarget::Gloss)
                }
                b"lsource" if full_english_source(&start, reader.decoder())? => {
                    begin_text(&mut target, &mut target_text, TextTarget::LanguageSource)
                }
                _ => {}
            },
            Event::Empty(empty) => {
                if empty.name().as_ref() == b"gloss" && english_gloss(&empty, reader.decoder())? {
                    // Empty glosses are invalid output and need no action.
                }
            }
            Event::Text(text) => {
                if target.is_some() {
                    let decoded = text.decode()?;
                    target_text.push_str(&unescape(&decoded)?);
                }
            }
            Event::CData(text) => {
                if target.is_some() {
                    target_text.push_str(&text.decode()?);
                }
            }
            Event::Comment(comment) => {
                if stats.source_created.is_empty() {
                    let text = comment.decode()?;
                    if let Some(created) = text.trim().strip_prefix("JMdict created:") {
                        stats.source_created = created.trim().to_owned();
                    }
                }
            }
            Event::End(end) => {
                if target.is_some_and(|value| value.element() == end.name().as_ref()) {
                    finish_text(
                        target.take().expect("target was checked"),
                        std::mem::take(&mut target_text),
                        entry.as_mut(),
                        reading.as_mut(),
                        sense.as_mut(),
                    );
                }
                match end.name().as_ref() {
                    b"r_ele" => {
                        if let (Some(current_entry), Some(current_reading)) =
                            (entry.as_mut(), reading.take())
                        {
                            current_entry.readings.push(current_reading);
                        }
                    }
                    b"sense" => {
                        if let (Some(current_entry), Some(current_sense)) =
                            (entry.as_mut(), sense.take())
                        {
                            current_entry.senses.push(current_sense);
                        }
                    }
                    b"entry" => {
                        if let Some(current_entry) = entry.take() {
                            process_entry(current_entry, base_cost, &mut entries, &mut stats);
                        }
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buffer.clear();
    }

    let output = entries
        .into_iter()
        .map(|((reading, surface), cost)| OutputEntry {
            reading,
            surface,
            cost,
        })
        .collect();
    Ok((output, stats))
}

fn begin_text(target: &mut Option<TextTarget>, text: &mut String, value: TextTarget) {
    *target = Some(value);
    text.clear();
}

fn finish_text(
    target: TextTarget,
    text: String,
    entry: Option<&mut RawEntry>,
    reading: Option<&mut RawReading>,
    sense: Option<&mut RawSense>,
) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    match target {
        TextTarget::Sequence => {
            if let Some(entry) = entry {
                entry.sequence = text.to_owned();
            }
        }
        TextTarget::Reading => {
            if let Some(reading) = reading {
                reading.text = text.to_owned();
            }
        }
        TextTarget::Priority => {
            if let Some(reading) = reading {
                reading.priorities.push(text.to_owned());
            }
        }
        TextTarget::Restriction => {
            if let Some(sense) = sense {
                sense.restrictions.insert(text.to_owned());
            }
        }
        TextTarget::Gloss => {
            if let Some(sense) = sense {
                if valid_surface(text) {
                    sense.glosses.push(text.to_owned());
                }
            }
        }
        TextTarget::LanguageSource => {
            if !valid_surface(text) {
                return;
            }
            if let Some(sense) = sense {
                sense.sources.push(text.to_owned());
            } else if let Some(entry) = entry {
                entry.entry_sources.push(text.to_owned());
            }
        }
    }
}

fn process_entry(
    entry: RawEntry,
    base_cost: i32,
    output: &mut BTreeMap<(String, String), i32>,
    stats: &mut ParseStats,
) {
    stats.jmdict_entries += 1;
    let eligible: Vec<&RawReading> = entry
        .readings
        .iter()
        .filter(|reading| is_katakana_reading(&reading.text))
        .collect();
    if eligible.is_empty() {
        return;
    }
    stats.katakana_entries += 1;

    for reading in eligible {
        let normalized = normalize_reading(&reading.text);
        if normalized.len() > MAX_FIELD_BYTES {
            continue;
        }
        stats.unique_readings.insert(normalized.clone());
        let priority_penalty = priority_penalty(&reading.priorities);

        for (source_index, source) in entry.entry_sources.iter().enumerate() {
            retain_candidate(
                output,
                &normalized,
                source,
                candidate_cost(base_cost, priority_penalty, 0, source_index),
            );
        }
        for (sense_index, sense) in entry.senses.iter().enumerate() {
            if !sense.restrictions.is_empty() && !sense.restrictions.contains(&reading.text) {
                continue;
            }
            for (source_index, source) in sense.sources.iter().enumerate() {
                retain_candidate(
                    output,
                    &normalized,
                    source,
                    candidate_cost(base_cost, priority_penalty, sense_index * 20, source_index),
                );
            }
            for (gloss_index, gloss) in sense.glosses.iter().enumerate() {
                retain_candidate(
                    output,
                    &normalized,
                    gloss,
                    candidate_cost(
                        base_cost,
                        priority_penalty,
                        500 + sense_index * 20,
                        gloss_index,
                    ),
                );
            }
        }
    }
}

fn retain_candidate(
    output: &mut BTreeMap<(String, String), i32>,
    reading: &str,
    surface: &str,
    cost: i32,
) {
    output
        .entry((reading.to_owned(), surface.to_owned()))
        .and_modify(|current| *current = (*current).min(cost))
        .or_insert(cost);
}

fn candidate_cost(base: i32, priority: i32, kind: usize, index: usize) -> i32 {
    let value = i64::from(base)
        + i64::from(priority)
        + i64::try_from(kind).unwrap_or(i64::MAX)
        + i64::try_from(index).unwrap_or(i64::MAX);
    value.clamp(1, 32_767) as i32
}

fn priority_penalty(priorities: &[String]) -> i32 {
    let mut penalty = 6_000;
    for priority in priorities {
        let candidate = match priority.as_str() {
            "news1" | "ichi1" | "spec1" | "gai1" => 0,
            "news2" | "ichi2" | "spec2" | "gai2" => 2_000,
            value if value.starts_with("nf") => value[2..]
                .parse::<i32>()
                .ok()
                .map(|rank| (rank - 1).clamp(0, 49) * 100)
                .unwrap_or(6_000),
            _ => 6_000,
        };
        penalty = penalty.min(candidate);
    }
    penalty
}

pub fn normalize_reading(input: &str) -> String {
    let decomposed: String = input
        .nfkd()
        .map(|character| match character {
            '\u{30A1}'..='\u{30F6}' | '\u{30FD}'..='\u{30FE}' => {
                char::from_u32(character as u32 - 0x60).unwrap_or(character)
            }
            _ => character,
        })
        .collect();
    decomposed.nfc().collect()
}

fn is_katakana_reading(input: &str) -> bool {
    !input.is_empty()
        && input
            .chars()
            .all(|character| matches!(character, '\u{30A0}'..='\u{30FF}'))
}

fn valid_surface(input: &str) -> bool {
    !input.is_empty()
        && input.len() <= MAX_FIELD_BYTES
        && !input
            .chars()
            .any(|character| character.is_control() || character == '\t')
}

fn english_gloss(start: &BytesStart<'_>, decoder: Decoder) -> Result<bool> {
    let language = attribute(start, b"xml:lang", decoder)?.unwrap_or_else(|| "eng".to_owned());
    let gloss_type = attribute(start, b"g_type", decoder)?;
    Ok(language == "eng" && gloss_type.as_deref() != Some("expl"))
}

fn full_english_source(start: &BytesStart<'_>, decoder: Decoder) -> Result<bool> {
    let language = attribute(start, b"xml:lang", decoder)?.unwrap_or_else(|| "eng".to_owned());
    let source_type = attribute(start, b"ls_type", decoder)?.unwrap_or_else(|| "full".to_owned());
    Ok(language == "eng" && source_type == "full")
}

fn attribute(start: &BytesStart<'_>, name: &[u8], decoder: Decoder) -> Result<Option<String>> {
    for attribute in start.attributes().with_checks(false) {
        let attribute = attribute?;
        if attribute.key.as_ref() == name {
            return Ok(Some(
                attribute.decode_and_unescape_value(decoder)?.into_owned(),
            ));
        }
    }
    Ok(None)
}

fn open_input(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("opening JMdict {}", path.display()))?;
    if path.extension().is_some_and(|extension| extension == "gz") {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn generic_noun_id(path: &Path) -> Result<u16> {
    let reader = BufReader::new(File::open(path)?);
    let mut fallback = None;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (id, feature) = line
            .split_once(' ')
            .with_context(|| format!("malformed id.def line {}: {line}", line_index + 1))?;
        let id: u16 = id.parse()?;
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(feature.as_bytes());
        let record = csv_reader
            .records()
            .next()
            .transpose()?
            .with_context(|| format!("empty id.def feature at line {}", line_index + 1))?;
        if record.get(0) == Some("名詞") {
            fallback.get_or_insert(id);
            if record.get(1) == Some("一般") {
                return Ok(id);
            }
        }
    }
    fallback.context("Mozc id.def has no noun context ID")
}

fn write_dictionary_shards(
    output_dir: &Path,
    entries: &[OutputEntry],
    context_id: u16,
    entries_per_shard: usize,
) -> Result<Vec<Asset>> {
    let mut assets = Vec::new();
    for (part, chunk) in entries.chunks(entries_per_shard).enumerate() {
        let path = output_dir.join(format!("mozc-english-unigram-{part:05}.txt.zst"));
        let mut encoder = zstd::stream::write::Encoder::new(File::create(&path)?, 10)?;
        encoder.multithread(0)?;
        for entry in chunk {
            writeln!(
                encoder,
                "{}\t{}\t{}\t{}\t{}",
                entry.reading, context_id, context_id, entry.cost, entry.surface
            )?;
        }
        encoder.finish()?;
        assets.push(asset_for_file(
            &path,
            "dictionary",
            Some(chunk.len() as u64),
        )?);
    }
    Ok(assets)
}

fn source_asset_name(input: &Path, created: &str, sha256: &str) -> String {
    let version = if created.is_empty() {
        Cow::Borrowed(&sha256[..12])
    } else {
        Cow::Owned(created.replace('-', ""))
    };
    if input.extension().is_some_and(|extension| extension == "gz") {
        format!("JMdict_e-{version}.xml.gz")
    } else {
        format!("JMdict_e-{version}.xml")
    }
}

fn asset_for_file(path: &Path, kind: &'static str, entries: Option<u64>) -> Result<Asset> {
    let (bytes, sha256) = file_metadata(path)?;
    Ok(Asset {
        name: path
            .file_name()
            .context("asset has no file name")?
            .to_string_lossy()
            .into_owned(),
        kind,
        entries,
        bytes,
        sha256,
    })
}

fn file_metadata(path: &Path) -> Result<(u64, String)> {
    let mut file = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

fn dictionary_readme() -> &'static str {
    "# Hiragana-to-English Mozc dictionary\n\n\
This directory contains English conversion candidates extracted from the\n\
English-only JMdict distribution. Only readings written entirely with\n\
katakana-block characters are selected; dictionary keys are normalized with\n\
NFKC and converted to hiragana. English `gloss` values and complete English\n\
`lsource` values become candidate surfaces.\n\n\
Each `mozc-english-unigram-*.txt.zst` file expands to Mozc's five-column\n\
system-dictionary source format:\n\n\
```text\n\
reading<TAB>left_id<TAB>right_id<TAB>cost<TAB>surface\n\
```\n\n\
JMdict priority markers and gloss order determine candidate costs. See\n\
`english-dictionary-manifest.json` for the exact source checksum, selection\n\
parameters, counts, and generated asset checksums. JMdict attribution and\n\
licence terms are provided in `JMDICT-LICENSE.html`.\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn normalizes_katakana_and_halfwidth_to_hiragana() {
        assert_eq!(normalize_reading("コンピューター"), "こんぴゅーたー");
        assert_eq!(normalize_reading("ｱｰﾄ"), "あーと");
        assert_eq!(
            normalize_reading("ヷヸヹヺヿ"),
            "わ\u{3099}ゐ\u{3099}ゑ\u{3099}を\u{3099}こと"
        );
    }

    #[test]
    fn builds_ranked_dictionary_and_metadata() -> Result<()> {
        let directory = tempdir()?;
        let input = directory.path().join("JMdict_e.xml");
        let id_def = directory.path().join("id.def");
        let license = directory.path().join("licence.html");
        let output = directory.path().join("output");
        fs::write(&input, include_bytes!("../tests/fixtures/JMdict_e.xml"))?;
        fs::write(&id_def, include_bytes!("../tests/fixtures/id.def"))?;
        fs::write(
            &license,
            include_bytes!("../tests/fixtures/JMDICT-LICENSE.html"),
        )?;

        build(BuildOptions {
            input,
            mozc_id_def: id_def,
            jmdict_license: license,
            output_dir: output.clone(),
            entries_per_shard: 100,
            base_cost: 12_000,
            source_url: "https://example.test/JMdict_e.gz".to_owned(),
            source_etag: "fixture".to_owned(),
            source_last_modified: "2026-07-22".to_owned(),
            pipeline_commit: "deadbeef".to_owned(),
        })?;

        let shard = File::open(output.join("mozc-english-unigram-00000.txt.zst"))?;
        let mut text = String::new();
        zstd::stream::read::Decoder::new(shard)?.read_to_string(&mut text)?;
        assert!(text.contains("あーと\t10\t10\t12500\tart\n"));
        assert!(text.contains("こんぴゅーた\t10\t10\t12500\tcomputer\n"));
        assert!(text.contains("こんぴゅーたー\t10\t10\t12500\tcomputer\n"));
        assert!(text.contains("せんす\t10\t10\t12000\tsense\n"));
        assert!(text.contains("てすと\t10\t10\t18500\ttest\n"));
        assert!(text.contains("てすてぃんぐ\t10\t10\t18520\ttesting\n"));
        assert!(!text.contains("explanatory gloss"));
        assert!(!text.contains("ひらがな"));

        let art_cost = text
            .lines()
            .find(|line| line.ends_with("\tart"))
            .and_then(|line| line.split('\t').nth(3))
            .and_then(|cost| cost.parse::<i32>().ok())
            .expect("art entry");
        let acronym_cost = text
            .lines()
            .find(|line| line.ends_with("\tART"))
            .and_then(|line| line.split('\t').nth(3))
            .and_then(|cost| cost.parse::<i32>().ok())
            .expect("ART entry");
        assert!(art_cost < acronym_cost);

        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(output.join(MANIFEST_NAME))?)?;
        assert_eq!(manifest["source"]["created"], "2026-07-22");
        assert_eq!(manifest["counts"]["jmdict_entries"], 6);
        assert_eq!(manifest["counts"]["katakana_entries"], 5);
        assert_eq!(manifest["counts"]["unique_readings"], 6);
        assert_eq!(manifest["counts"]["retained_entries"], 9);

        let checksums = fs::read_to_string(output.join(CHECKSUMS_NAME))?;
        assert!(checksums.contains(MANIFEST_NAME));
        assert!(checksums.contains("JMdict_e-20260722.xml"));
        Ok(())
    }
}
