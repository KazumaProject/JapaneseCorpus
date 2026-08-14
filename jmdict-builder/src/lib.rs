use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum DictionaryKind {
    Generic,
    DirectLoanword,
}

impl DictionaryKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::DirectLoanword => "direct-loanword",
        }
    }
}

#[derive(Debug)]
pub struct BuildOptions {
    pub input: PathBuf,
    pub mozc_id_def: PathBuf,
    pub jmdict_license: PathBuf,
    pub output_dir: PathBuf,
    pub entries_per_shard: usize,
    pub base_cost: i32,
    pub dictionary_kind: DictionaryKind,
    pub pronunciation_dictionary: Option<PathBuf>,
    pub pronunciation_dictionary_commit: Option<String>,
    pub pronunciation_dictionary_sha256: Option<String>,
    pub direct_loanword_allowlist: Option<PathBuf>,
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

struct DirectLoanwordFilter {
    pronunciations: HashMap<String, Vec<Vec<String>>>,
    allowlist: HashSet<(String, String)>,
}

impl DirectLoanwordFilter {
    fn load(pronunciation_path: &Path, allowlist_path: Option<&Path>) -> Result<Self> {
        let mut pronunciations: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for (line_number, raw_line) in BufReader::new(File::open(pronunciation_path)?)
            .lines()
            .enumerate()
        {
            let raw_line = raw_line?;
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with(";;") {
                continue;
            }
            let (raw_word, raw_phones) = line.split_once(' ').with_context(|| {
                format!("invalid CMUdict line {}: {}", line_number + 1, raw_line)
            })?;
            let word = raw_word
                .rsplit_once('(')
                .map(|(base, suffix)| {
                    if suffix.ends_with(')') {
                        base
                    } else {
                        raw_word
                    }
                })
                .unwrap_or(raw_word)
                .to_ascii_lowercase();
            let phones: Vec<String> = raw_phones.split_whitespace().map(str::to_owned).collect();
            if !phones.is_empty() {
                pronunciations.entry(word).or_default().push(phones);
            }
        }

        let mut allowlist = HashSet::new();
        if let Some(path) = allowlist_path {
            for (line_number, raw_line) in BufReader::new(File::open(path)?).lines().enumerate() {
                let raw_line = raw_line?;
                let line = raw_line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let fields: Vec<&str> = line.split('\t').collect();
                if fields.len() != 3
                    || fields[0].is_empty()
                    || fields[1].is_empty()
                    || fields[2].is_empty()
                {
                    bail!(
                        "invalid direct-loanword allowlist line {}: expected reading<TAB>surface<TAB>reason",
                        line_number + 1
                    );
                }
                allowlist.insert((
                    normalize_reading(fields[0]),
                    fields[1].trim().to_ascii_lowercase(),
                ));
            }
        }

        Ok(Self {
            pronunciations,
            allowlist,
        })
    }

    fn check(&self, reading: &str, surface: &str) -> std::result::Result<(), &'static str> {
        if !valid_direct_surface(surface) {
            return Err("invalid-surface");
        }
        if !lexical_direct_surface(surface) {
            return Err("non-lexical-surface");
        }
        let normalized_reading = normalize_reading(reading);
        let surface_key = surface.to_ascii_lowercase();
        let allowlisted = self
            .allowlist
            .contains(&(normalized_reading.clone(), surface_key.clone()));

        if surface.contains(' ') || surface.contains('-') {
            // Compounds are accepted only as an exact reading/surface pair.
            // The allowlist is deliberately pair-keyed, so a surface such as
            // "art gallery" cannot leak into the shorter `ぎゃらりー` entry.
            if allowlisted {
                return Ok(());
            }
            return Err("compound-not-allowlisted");
        }

        let Some(pronunciations) = self.pronunciations.get(&surface_key) else {
            return if allowlisted {
                Ok(())
            } else {
                Err("missing-pronunciation")
            };
        };
        if pronunciations
            .iter()
            .any(|phones| pronunciation_matches(&normalized_reading, phones))
        {
            Ok(())
        } else if allowlisted {
            Ok(())
        } else {
            Err("pronunciation-mismatch")
        }
    }
}

fn valid_direct_surface(surface: &str) -> bool {
    let trimmed = surface.trim();
    !trimmed.is_empty()
        && trimmed == surface
        && !surface.contains('(')
        && !surface.contains(')')
        && !surface.contains("  ")
        && !surface.starts_with('-')
        && !surface.ends_with('-')
        && surface.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == ' ' || character == '-'
        })
}

fn lexical_direct_surface(surface: &str) -> bool {
    if surface.len() == 1 && surface.as_bytes()[0].is_ascii_alphabetic() {
        return false;
    }
    let normalized = surface.to_ascii_lowercase();
    if numeric_surface_words()
        .iter()
        .any(|word| *word == normalized.as_str())
    {
        return false;
    }
    if function_words()
        .iter()
        .any(|word| *word == normalized.as_str())
    {
        return false;
    }
    !(normalized
        .chars()
        .any(|character| character.is_ascii_digit())
        && normalized
            .chars()
            .all(|character| character.is_ascii_digit() || "^+-*/=.".contains(character)))
}

fn numeric_surface_words() -> &'static [&'static str] {
    &[
        "zero",
        "nought",
        "nil",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]
}

fn function_words() -> &'static [&'static str] {
    &[
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "at",
        "by", "for", "with", "from", "as", "is", "am", "are", "be", "was", "were", "been", "being",
        "do", "does", "did", "it", "this", "that", "these", "those", "here", "there", "who",
        "whom", "whose", "which", "what", "when", "where", "why", "how", "me", "my", "mine", "you",
        "your", "yours", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours",
        "they", "them", "their", "theirs", "some", "any", "no", "not", "nor",
    ]
}

fn pronunciation_matches(reading: &str, phones: &[String]) -> bool {
    let expected = expand_long_vowels(&normalize_reading(reading));
    let Some(romaji) = pronunciation_to_romaji(phones) else {
        return false;
    };
    let Ok(katakana) = to_kana::kata(&romaji) else {
        return false;
    };
    let generated = expand_long_vowels(&normalize_reading(&katakana));
    expected == generated
}

fn pronunciation_to_romaji(phones: &[String]) -> Option<String> {
    let normalized: Vec<String> = phones
        .iter()
        .map(|phone| phone.trim_end_matches(['0', '1', '2']).to_owned())
        .collect();
    let key = normalized.join(" ");
    let override_value = match key.as_str() {
        "G AE L ER IY" => "gyararii",
        "K AA R" => "kaa",
        "AA R T" => "aato",
        "AY S" => "aisu",
        "AY D AH L" => "aidoru",
        "K AH M P Y UW T ER" => "konpyuutaa",
        "K IY" => "kii",
        "HH OW L D ER" => "horudaa",
        "K R IY M" => "kuriimu",
        "S M AA R T F OW N" => "sumaatofon",
        "G AE S AH L IY N" => "gasorin",
        "S T AE N D" => "sutando",
        "OW P AH N" => "oopun",
        "K AE M P IH NG" => "kyanpingu",
        "M AY" => "mai",
        "T AY M" => "taimu",
        "S EY L" => "seeru",
        "P ER S IH N AH L" => "paasonaru",
        "AE S K IY" => "asukii",
        _ => "",
    };
    if !override_value.is_empty() {
        return Some(override_value.to_owned());
    }

    let mut output = String::new();
    let vowels = [
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    ];
    let vowel = |phone: &str| vowels.contains(&phone);
    let map_consonant = |phone: &str| match phone {
        "P" => Some("p"),
        "B" => Some("b"),
        "M" => Some("m"),
        "F" => Some("f"),
        "V" => Some("v"),
        "T" => Some("t"),
        "D" => Some("d"),
        "K" => Some("k"),
        "G" => Some("g"),
        "S" => Some("s"),
        "Z" => Some("z"),
        "SH" => Some("sh"),
        "ZH" => Some("j"),
        "CH" => Some("ch"),
        "JH" => Some("j"),
        "N" => Some("n"),
        "NG" => Some("ng"),
        "HH" => Some("h"),
        "L" | "R" => Some("r"),
        "W" => Some("w"),
        "Y" => Some("y"),
        "TH" => Some("s"),
        _ => None,
    };
    let map_vowel = |phone: &str| match phone {
        "AA" => Some("aa"),
        "AE" => Some("a"),
        "AH" => Some("o"),
        "AO" => Some("o"),
        "AW" => Some("au"),
        "AY" => Some("ai"),
        "EH" => Some("e"),
        "ER" => Some("aa"),
        "EY" => Some("ei"),
        "IH" => Some("i"),
        "IY" => Some("ii"),
        "OW" => Some("ou"),
        "OY" => Some("oi"),
        "UH" => Some("u"),
        "UW" => Some("uu"),
        _ => None,
    };

    let mut index = 0;
    while index < normalized.len() {
        let phone = normalized[index].as_str();
        if phone == "R"
            && !output.is_empty()
            && (index + 1 == normalized.len() || !vowel(normalized[index + 1].as_str()))
        {
            index += 1;
            continue;
        }
        if vowel(phone) {
            output.push_str(map_vowel(phone)?);
            index += 1;
            continue;
        }

        let mut end = index;
        while end < normalized.len() && !vowel(normalized[end].as_str()) {
            end += 1;
        }
        if end == normalized.len() {
            for consonant in &normalized[index..end] {
                match consonant.as_str() {
                    "R" => {}
                    "N" | "NG" => output.push('n'),
                    "L" => output.push_str("ru"),
                    "M" => output.push_str("mu"),
                    value => output.push_str(map_consonant(value)?),
                }
            }
            break;
        }

        let cluster = &normalized[index..end];
        if cluster.len() > 1 {
            for consonant in &cluster[..cluster.len() - 1] {
                if consonant == "R" {
                    continue;
                }
                output.push_str(map_consonant(consonant.as_str())?);
                output.push('u');
            }
        }
        let onset = cluster
            .last()
            .and_then(|value| map_consonant(value.as_str()))?;
        let onset = if normalized[end] == "AE"
            && cluster.len() == 1
            && matches!(cluster[0].as_str(), "G" | "K")
        {
            format!("{onset}y")
        } else {
            onset.to_owned()
        };
        output.push_str(&onset);
        output.push_str(map_vowel(normalized[end].as_str())?);
        index = end + 1;
    }
    Some(output)
}

fn expand_long_vowels(input: &str) -> String {
    let mut output = String::new();
    let mut previous_vowel = None;
    for character in input.chars() {
        if character == 'ー' {
            if let Some(vowel) = previous_vowel {
                output.push(vowel);
            }
            continue;
        }
        output.push(character);
        previous_vowel = kana_vowel(character).or(previous_vowel);
    }
    output
}

fn kana_vowel(character: char) -> Option<char> {
    match character {
        'あ' | 'か' | 'が' | 'さ' | 'ざ' | 'た' | 'だ' | 'な' | 'は' | 'ば' | 'ぱ' | 'ま'
        | 'や' | 'ら' | 'わ' | 'ぁ' | 'ゃ' => Some('あ'),
        'い' | 'き' | 'ぎ' | 'し' | 'じ' | 'ち' | 'ぢ' | 'に' | 'ひ' | 'び' | 'ぴ' | 'み'
        | 'り' | 'ゐ' | 'ぃ' => Some('い'),
        'う' | 'く' | 'ぐ' | 'す' | 'ず' | 'つ' | 'づ' | 'ぬ' | 'ふ' | 'ぶ' | 'ぷ' | 'む'
        | 'ゆ' | 'る' | 'ゔ' | 'ぅ' | 'ゅ' => Some('う'),
        'え' | 'け' | 'げ' | 'せ' | 'ぜ' | 'て' | 'で' | 'ね' | 'へ' | 'べ' | 'ぺ' | 'め'
        | 'れ' | 'ゑ' | 'ぇ' => Some('え'),
        'お' | 'こ' | 'ご' | 'そ' | 'ぞ' | 'と' | 'ど' | 'の' | 'ほ' | 'ぼ' | 'ぽ' | 'も'
        | 'よ' | 'ろ' | 'を' | 'ぉ' | 'ょ' => Some('お'),
        _ => None,
    }
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
    rejected_candidates: BTreeMap<String, u64>,
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
    dictionary_kind: &'static str,
    reading_selection: &'static str,
    translation_language: &'static str,
    include_full_english_lsource: bool,
    excluded_gloss_types: [&'static str; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dictionary: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dictionary_commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pronunciation_dictionary_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    direct_loanword_allowlist: Option<&'static str>,
    base_cost: i32,
    entries_per_shard: usize,
}

#[derive(Serialize)]
struct Counts {
    jmdict_entries: u64,
    katakana_entries: u64,
    unique_readings: u64,
    retained_entries: u64,
    rejected_candidates: BTreeMap<String, u64>,
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
    let direct_filter = if options.dictionary_kind == DictionaryKind::DirectLoanword {
        Some(DirectLoanwordFilter::load(
            options
                .pronunciation_dictionary
                .as_deref()
                .context("--pronunciation-dictionary is required for direct-loanword output")?,
            options.direct_loanword_allowlist.as_deref(),
        )?)
    } else {
        None
    };
    let (entries, stats) = parse_jmdict(&options.input, options.base_cost, direct_filter.as_ref())?;
    if entries.is_empty() {
        bail!("JMdict produced no katakana-to-English entries");
    }

    let mut assets = write_dictionary_shards(
        &options.output_dir,
        &entries,
        context_id,
        options.entries_per_shard,
        options.dictionary_kind,
    )?;

    let source_name = source_asset_name(&options.input, &stats.source_created, &source_sha256);
    let source_output = options.output_dir.join(source_name);
    fs::copy(&options.input, &source_output)?;
    assets.push(asset_for_file(&source_output, "source", None)?);

    let license_output = options.output_dir.join("JMDICT-LICENSE.html");
    fs::copy(&options.jmdict_license, &license_output)?;
    assets.push(asset_for_file(&license_output, "license", None)?);

    let readme_output = options.output_dir.join("ENGLISH-DICTIONARY-README.md");
    fs::write(&readme_output, dictionary_readme(options.dictionary_kind))?;
    assets.push(asset_for_file(&readme_output, "documentation", None)?);
    if options.dictionary_kind == DictionaryKind::DirectLoanword {
        if let Some(allowlist) = options.direct_loanword_allowlist.as_deref() {
            let allowlist_output = options.output_dir.join("direct-loanword-allowlist.tsv");
            fs::copy(allowlist, &allowlist_output)?;
            assets.push(asset_for_file(&allowlist_output, "policy", None)?);
        }
    }
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
            dictionary_kind: options.dictionary_kind.as_str(),
            reading_selection: "readings consisting entirely of Unicode katakana-block characters",
            translation_language: "eng",
            include_full_english_lsource: true,
            excluded_gloss_types: ["expl"],
            pronunciation_dictionary: (options.dictionary_kind == DictionaryKind::DirectLoanword)
                .then(|| "CMUdict"),
            pronunciation_dictionary_commit: options.pronunciation_dictionary_commit.clone(),
            pronunciation_dictionary_sha256: options.pronunciation_dictionary_sha256.clone(),
            direct_loanword_allowlist: (options.dictionary_kind == DictionaryKind::DirectLoanword)
                .then(|| "direct-loanword-allowlist.tsv"),
            base_cost: options.base_cost,
            entries_per_shard: options.entries_per_shard,
        },
        counts: Counts {
            jmdict_entries: stats.jmdict_entries,
            katakana_entries: stats.katakana_entries,
            unique_readings: stats.unique_readings.len() as u64,
            retained_entries: entries.len() as u64,
            rejected_candidates: stats.rejected_candidates.clone(),
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
        "built {} {} hiragana-to-English entries from {} katakana JMdict entries",
        entries.len(),
        options.dictionary_kind.as_str(),
        stats.katakana_entries
    );
    if !stats.rejected_candidates.is_empty() {
        eprintln!(
            "direct-loanword rejection counts: {:?}",
            stats.rejected_candidates
        );
    }
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
    if options.dictionary_kind == DictionaryKind::Generic
        && (options.pronunciation_dictionary.is_some()
            || options.pronunciation_dictionary_commit.is_some()
            || options.pronunciation_dictionary_sha256.is_some()
            || options.direct_loanword_allowlist.is_some())
    {
        bail!("pronunciation and direct-loanword allowlist options require --dictionary-kind direct-loanword");
    }
    if options.dictionary_kind == DictionaryKind::DirectLoanword {
        if options.pronunciation_dictionary.is_none() {
            bail!("--pronunciation-dictionary is required for direct-loanword output");
        }
        if options.pronunciation_dictionary_commit.is_none()
            || options.pronunciation_dictionary_sha256.is_none()
        {
            bail!("direct-loanword output requires CMUdict commit and SHA-256 metadata");
        }
    }
    Ok(())
}

fn parse_jmdict(
    path: &Path,
    base_cost: i32,
    direct_filter: Option<&DirectLoanwordFilter>,
) -> Result<(Vec<OutputEntry>, ParseStats)> {
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
                            process_entry(
                                current_entry,
                                base_cost,
                                direct_filter,
                                &mut entries,
                                &mut stats,
                            );
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
        .collect::<Vec<_>>();
    let output = if direct_filter.is_some() {
        let mut deduplicated: BTreeMap<(String, String), OutputEntry> = BTreeMap::new();
        for entry in output {
            let key = (entry.reading.clone(), entry.surface.to_ascii_lowercase());
            let replace = deduplicated.get(&key).map_or(true, |current| {
                (
                    entry.cost,
                    entry.surface != entry.surface.to_ascii_lowercase(),
                    &entry.surface,
                ) < (
                    current.cost,
                    current.surface != current.surface.to_ascii_lowercase(),
                    &current.surface,
                )
            });
            if replace {
                deduplicated.insert(key, entry);
            }
        }
        deduplicated.into_values().collect()
    } else {
        output
    };
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
    direct_filter: Option<&DirectLoanwordFilter>,
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
                direct_filter,
                stats,
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
                    direct_filter,
                    stats,
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
                    direct_filter,
                    stats,
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
    direct_filter: Option<&DirectLoanwordFilter>,
    stats: &mut ParseStats,
) {
    if let Some(filter) = direct_filter {
        if let Err(reason) = filter.check(reading, surface) {
            *stats
                .rejected_candidates
                .entry(reason.to_owned())
                .or_default() += 1;
            return;
        }
    }
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
    dictionary_kind: DictionaryKind,
) -> Result<Vec<Asset>> {
    let mut assets = Vec::new();
    let prefix = match dictionary_kind {
        DictionaryKind::Generic => "mozc-english-unigram",
        DictionaryKind::DirectLoanword => "mozc-english-reading-unigram",
    };
    for (part, chunk) in entries.chunks(entries_per_shard).enumerate() {
        let path = output_dir.join(format!("{prefix}-{part:05}.txt.zst"));
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

fn dictionary_readme(dictionary_kind: DictionaryKind) -> &'static str {
    if dictionary_kind == DictionaryKind::DirectLoanword {
        return "# Hiragana-to-English direct-loanword Mozc dictionary\n\n\
This directory contains conservative English loanword candidates extracted
from the English-only JMdict distribution. A candidate must be a complete
CMUdict pronunciation match for the full Japanese reading, or an exact entry
in the direct-loanword allowlist. Gloss explanations, partial phrases, and
parenthetical notes are not emitted.\n\n\
Each `mozc-english-reading-unigram-*.txt.zst` file expands to Mozc's
five-column system-dictionary source format. See
`english-dictionary-manifest.json` for the pronunciation source, selection
parameters, rejection counts, and generated asset checksums. The exact
compound exceptions are maintained in `direct-loanword-allowlist.tsv`.\n";
    }
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
            dictionary_kind: DictionaryKind::Generic,
            pronunciation_dictionary: None,
            pronunciation_dictionary_commit: None,
            pronunciation_dictionary_sha256: None,
            direct_loanword_allowlist: None,
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

    #[test]
    fn direct_loanword_mode_keeps_only_full_natural_matches() -> Result<()> {
        let directory = tempdir()?;
        let output = directory.path().join("output");
        let input = directory.path().join("direct-loanword-JMdict_e.xml");
        let pronunciation = directory.path().join("cmudict.dict");
        let allowlist = directory.path().join("allowlist.tsv");
        let id_def = directory.path().join("id.def");
        let license = directory.path().join("licence.html");
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        fs::copy(fixture_dir.join("direct-loanword-JMdict_e.xml"), &input)?;
        fs::copy(
            fixture_dir.join("direct-loanword-cmudict.dict"),
            &pronunciation,
        )?;
        fs::copy(
            fixture_dir.join("direct-loanword-allowlist.tsv"),
            &allowlist,
        )?;
        fs::write(&id_def, include_bytes!("../tests/fixtures/id.def"))?;
        fs::write(&license, "fixture")?;

        build(BuildOptions {
            input,
            mozc_id_def: id_def,
            jmdict_license: license,
            output_dir: output.clone(),
            entries_per_shard: 100,
            base_cost: 12_000,
            dictionary_kind: DictionaryKind::DirectLoanword,
            pronunciation_dictionary: Some(pronunciation),
            pronunciation_dictionary_commit: Some("fixture".to_owned()),
            pronunciation_dictionary_sha256: Some("fixture-sha256".to_owned()),
            direct_loanword_allowlist: Some(allowlist),
            source_url: "https://example.test/direct-JMdict_e.gz".to_owned(),
            source_etag: "fixture".to_owned(),
            source_last_modified: "2026-08-14".to_owned(),
            pipeline_commit: "deadbeef".to_owned(),
        })?;

        let shard = File::open(output.join("mozc-english-reading-unigram-00000.txt.zst"))?;
        let mut text = String::new();
        zstd::stream::read::Decoder::new(shard)?.read_to_string(&mut text)?;
        assert!(text.contains("ぎゃらりー\t10\t10\t12502\tgallery\n"));
        assert!(text.contains("あーと\t10\t10\t12500\tart\n"));
        assert!(text.contains("かー\t10\t10\t12500\tcar\n"));
        assert!(text.contains("あいすくりーむ\t10\t10\t12500\tice cream\n"));
        assert!(text.contains("あーとぎゃらりー\t10\t10\t12500\tart gallery\n"));
        assert!(text.contains("あいあいおーてぃー\t10\t10\t12500\tIIoT\n"));
        for rejected in [
            "art gallery",
            "corridor",
            "upper gallery (in a theatre)",
            "assisted reproductive technologies",
            "ART",
        ] {
            if rejected == "art gallery" {
                assert_eq!(
                    text.lines()
                        .filter(|line| line.ends_with("\tart gallery"))
                        .count(),
                    1
                );
                assert!(text
                    .lines()
                    .any(|line| line.starts_with("あーとぎゃらりー\t")));
            } else {
                assert!(
                    !text
                        .lines()
                        .any(|line| line.ends_with(&format!("\t{rejected}"))),
                    "{rejected}"
                );
            }
        }
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(output.join(MANIFEST_NAME))?)?;
        assert_eq!(manifest["parameters"]["dictionary_kind"], "direct-loanword");
        assert_eq!(
            manifest["parameters"]["pronunciation_dictionary"],
            "CMUdict"
        );
        assert_eq!(
            manifest["parameters"]["pronunciation_dictionary_commit"],
            "fixture"
        );
        assert_eq!(
            manifest["parameters"]["direct_loanword_allowlist"],
            "direct-loanword-allowlist.tsv"
        );
        assert!(
            manifest["counts"]["rejected_candidates"]["compound-not-allowlisted"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        Ok(())
    }
}
