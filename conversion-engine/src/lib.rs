use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use unicode_normalization::UnicodeNormalization;

pub mod ajimee;

/// Cost assigned to one input character that is not covered by the dictionary.
pub const DEFAULT_UNKNOWN_COST: i64 = 50_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DictionaryEntry {
    pub reading: String,
    pub surface: String,
    pub left_id: u16,
    pub right_id: u16,
    pub cost: i32,
    /// 1, 2, or 3 for the dictionary asset from which this entry was loaded.
    pub order: u8,
}

/// An in-memory, reading-sorted representation of the generated dictionaries.
#[derive(Debug)]
pub struct Dictionary {
    entries: Vec<DictionaryEntry>,
    max_reading_bytes: usize,
}

impl Dictionary {
    /// Loads all generated dictionary shards found in a directory.
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_paths([path.as_ref()])
    }

    /// Loads dictionary files and/or directories. Directories are scanned non-recursively.
    pub fn load_paths<I, P>(paths: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let files = collect_dictionary_files(paths)?;
        let mut entries = Vec::new();
        for path in &files {
            let order = order_from_path(path)?;
            load_file(path, order, &mut entries)?;
        }
        Self::from_entries(entries).with_context(|| {
            format!(
                "no usable entries were loaded from {} dictionary file(s)",
                files.len()
            )
        })
    }

    /// Constructs a dictionary directly. This is also convenient for embedding entries.
    pub fn from_entries(mut entries: Vec<DictionaryEntry>) -> Result<Self> {
        for entry in &mut entries {
            entry.reading = normalize_owned(std::mem::take(&mut entry.reading));
            validate_entry(entry)?;
        }

        // The engine does not use a connection matrix, so entries with the same reading and
        // surface are equivalent. Retain the cheapest representation deterministically.
        entries.sort_unstable_by(|left, right| {
            (
                &left.reading,
                &left.surface,
                left.cost,
                left.order,
                left.left_id,
                left.right_id,
            )
                .cmp(&(
                    &right.reading,
                    &right.surface,
                    right.cost,
                    right.order,
                    right.left_id,
                    right.right_id,
                ))
        });
        entries
            .dedup_by(|right, left| right.reading == left.reading && right.surface == left.surface);
        entries.sort_unstable_by(compare_entries);

        if entries.is_empty() {
            bail!("dictionary contains no entries");
        }
        let max_reading_bytes = entries
            .iter()
            .map(|entry| entry.reading.len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            entries,
            max_reading_bytes,
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[DictionaryEntry] {
        &self.entries
    }

    fn lookup(&self, reading: &str) -> &[DictionaryEntry] {
        let start = self
            .entries
            .partition_point(|entry| entry.reading.as_str() < reading);
        let end = self.entries[start..].partition_point(|entry| entry.reading.as_str() == reading)
            + start;
        &self.entries[start..end]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub reading: String,
    pub surface: String,
    pub left_id: u16,
    pub right_id: u16,
    pub cost: i64,
    /// 0 denotes an unknown-character fallback; 1 through 3 denote dictionary order.
    pub order: u8,
}

impl Segment {
    pub fn is_unknown(&self) -> bool {
        self.order == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub cost: i64,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// Maximum number of results returned.
    pub candidate_limit: usize,
    /// Maximum hypotheses retained at each input boundary.
    pub beam_width: usize,
    /// Cost for copying one input character not selected from the dictionary.
    pub unknown_cost: i64,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            candidate_limit: 5,
            beam_width: 64,
            unknown_cost: DEFAULT_UNKNOWN_COST,
        }
    }
}

#[derive(Debug)]
pub struct Converter {
    dictionary: Dictionary,
}

impl Converter {
    pub fn new(dictionary: Dictionary) -> Self {
        Self { dictionary }
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    pub fn convert(&self, input: &str, candidate_limit: usize) -> Result<Vec<Candidate>> {
        let beam_width = candidate_limit.saturating_mul(8).max(64);
        self.convert_with_options(
            input,
            &ConvertOptions {
                candidate_limit,
                beam_width,
                ..ConvertOptions::default()
            },
        )
    }

    /// Converts a reading with a left-to-right beam search over dictionary phrase entries.
    pub fn convert_with_options(
        &self,
        input: &str,
        options: &ConvertOptions,
    ) -> Result<Vec<Candidate>> {
        if options.candidate_limit == 0 {
            bail!("candidate_limit must be positive");
        }
        if options.beam_width < options.candidate_limit {
            bail!("beam_width must be at least candidate_limit");
        }
        if options.unknown_cost <= 0 {
            bail!("unknown_cost must be positive");
        }

        let reading = normalize_reading(input);
        if reading.chars().any(char::is_control) {
            bail!("input must not contain control characters");
        }
        if reading.is_empty() {
            return Ok(vec![Candidate {
                text: String::new(),
                cost: 0,
                segments: Vec::new(),
            }]);
        }

        let mut beams = vec![Vec::<Hypothesis>::new(); reading.len() + 1];
        beams[0].push(Hypothesis::default());

        for start in reading.char_indices().map(|(offset, _)| offset) {
            if beams[start].is_empty() {
                continue;
            }
            let hypotheses = std::mem::take(&mut beams[start]);
            let mut edges = Vec::<(usize, Segment)>::new();

            for (relative, character) in reading[start..].char_indices() {
                let end = start + relative + character.len_utf8();
                if end - start > self.dictionary.max_reading_bytes {
                    break;
                }
                let prefix = &reading[start..end];
                for entry in self.dictionary.lookup(prefix) {
                    edges.push((
                        end,
                        Segment {
                            reading: entry.reading.clone(),
                            surface: entry.surface.clone(),
                            left_id: entry.left_id,
                            right_id: entry.right_id,
                            cost: i64::from(entry.cost),
                            order: entry.order,
                        },
                    ));
                }
            }

            let character = reading[start..]
                .chars()
                .next()
                .expect("start is a character boundary");
            let unknown_end = start + character.len_utf8();
            let copied = character.to_string();
            edges.push((
                unknown_end,
                Segment {
                    reading: copied.clone(),
                    surface: copied,
                    left_id: 0,
                    right_id: 0,
                    cost: options.unknown_cost,
                    order: 0,
                },
            ));

            for (end, segment) in edges {
                for hypothesis in &hypotheses {
                    beams[end].push(hypothesis.extend(&segment));
                }
                prune(&mut beams[end], options.beam_width);
            }
        }

        let mut candidates: Vec<Candidate> = beams[reading.len()]
            .drain(..)
            .map(|hypothesis| Candidate {
                text: hypothesis.text,
                cost: hypothesis.cost,
                segments: hypothesis.segments,
            })
            .collect();
        candidates.sort_unstable_by(compare_candidates);
        candidates.dedup_by(|right, left| right.text == left.text);
        candidates.truncate(options.candidate_limit);
        Ok(candidates)
    }
}

#[derive(Clone, Debug, Default)]
struct Hypothesis {
    text: String,
    cost: i64,
    unknowns: usize,
    segments: Vec<Segment>,
}

impl Hypothesis {
    fn extend(&self, segment: &Segment) -> Self {
        let mut text = self.text.clone();
        text.push_str(&segment.surface);
        let mut segments = self.segments.clone();
        segments.push(segment.clone());
        Self {
            text,
            cost: self.cost.saturating_add(segment.cost),
            unknowns: self.unknowns + usize::from(segment.is_unknown()),
            segments,
        }
    }
}

fn prune(hypotheses: &mut Vec<Hypothesis>, beam_width: usize) {
    hypotheses.sort_unstable_by(compare_hypotheses);
    // Once the input position and emitted text are identical, future expansions are also
    // identical. Only the cheapest segmentation can affect a result.
    hypotheses.dedup_by(|right, left| right.text == left.text);
    hypotheses.truncate(beam_width);
}

fn compare_entries(left: &DictionaryEntry, right: &DictionaryEntry) -> Ordering {
    (
        &left.reading,
        left.cost,
        &left.surface,
        left.order,
        left.left_id,
        left.right_id,
    )
        .cmp(&(
            &right.reading,
            right.cost,
            &right.surface,
            right.order,
            right.left_id,
            right.right_id,
        ))
}

fn compare_hypotheses(left: &Hypothesis, right: &Hypothesis) -> Ordering {
    (left.cost, left.unknowns, left.segments.len(), &left.text).cmp(&(
        right.cost,
        right.unknowns,
        right.segments.len(),
        &right.text,
    ))
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    let left_unknowns = left
        .segments
        .iter()
        .filter(|segment| segment.is_unknown())
        .count();
    let right_unknowns = right
        .segments
        .iter()
        .filter(|segment| segment.is_unknown())
        .count();
    (left.cost, left_unknowns, left.segments.len(), &left.text).cmp(&(
        right.cost,
        right_unknowns,
        right.segments.len(),
        &right.text,
    ))
}

/// Applies NFKC and converts ordinary katakana code points to hiragana.
pub fn normalize_reading(input: &str) -> String {
    input
        .nfkc()
        .map(|character| match character {
            '\u{30A1}'..='\u{30F6}' | '\u{30FD}'..='\u{30FE}' => {
                char::from_u32(character as u32 - 0x60).unwrap_or(character)
            }
            _ => character,
        })
        .collect()
}

fn normalize_owned(input: String) -> String {
    let is_already_normalized = input.nfkc().eq(input.chars())
        && !input.chars().any(|character| {
            matches!(
                character,
                '\u{30A1}'..='\u{30F6}' | '\u{30FD}'..='\u{30FE}'
            )
        });
    if is_already_normalized {
        input
    } else {
        normalize_reading(&input)
    }
}

fn collect_dictionary_files<I, P>(paths: I) -> Result<Vec<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut files = Vec::new();
    for path in paths {
        let path = path.as_ref();
        if path.is_dir() {
            for item in fs::read_dir(path)
                .with_context(|| format!("reading dictionary directory {}", path.display()))?
            {
                let item = item?;
                let item_path = item.path();
                if item_path.is_file()
                    && is_dictionary_filename(
                        item_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or(""),
                    )
                {
                    files.push(item_path);
                }
            }
        } else if path.is_file() {
            order_from_path(path)?;
            files.push(path.to_owned());
        } else {
            bail!("dictionary path does not exist: {}", path.display());
        }
    }
    files.sort_unstable();
    files.dedup();
    if files.is_empty() {
        bail!("no mozc-unigram/bigram/trigram .txt or .txt.zst files were found");
    }
    Ok(files)
}

fn is_dictionary_filename(name: &str) -> bool {
    name.starts_with("mozc-")
        && (name.contains("-unigram-") || name.contains("-bigram-") || name.contains("-trigram-"))
        && (name.ends_with(".txt") || name.ends_with(".txt.zst"))
}

fn order_from_path(path: &Path) -> Result<u8> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("dictionary filename is not UTF-8: {}", path.display()))?;
    if name.contains("unigram") {
        Ok(1)
    } else if name.contains("bigram") {
        Ok(2)
    } else if name.contains("trigram") {
        Ok(3)
    } else {
        bail!(
            "cannot infer n-gram order from dictionary filename: {}",
            path.display()
        )
    }
}

fn load_file(path: &Path, order: u8, output: &mut Vec<DictionaryEntry>) -> Result<()> {
    let file =
        File::open(path).with_context(|| format!("opening dictionary file {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|extension| extension == "zst") {
        Box::new(
            zstd::stream::read::Decoder::new(BufReader::new(file))
                .with_context(|| format!("opening zstd stream {}", path.display()))?,
        )
    } else {
        Box::new(file)
    };
    load_reader(BufReader::new(reader), path, order, output)
}

fn load_reader(
    reader: impl BufRead,
    path: &Path,
    order: u8,
    output: &mut Vec<DictionaryEntry>,
) -> Result<()> {
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "reading dictionary {} at line {}",
                path.display(),
                line_index + 1
            )
        })?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let field_count = line.bytes().filter(|byte| *byte == b'\t').count() + 1;
        if field_count != 5 {
            bail!(
                "{}:{}: expected 5 tab-separated columns, found {}",
                path.display(),
                line_index + 1,
                field_count
            );
        }
        let mut fields = line.split('\t');
        let reading = fields.next().expect("field count was checked");
        let left_id = fields.next().expect("field count was checked");
        let right_id = fields.next().expect("field count was checked");
        let cost = fields.next().expect("field count was checked");
        let surface = fields.next().expect("field count was checked");
        let entry = DictionaryEntry {
            reading: reading.to_owned(),
            left_id: parse_field(left_id, path, line_index, "left_id")?,
            right_id: parse_field(right_id, path, line_index, "right_id")?,
            cost: parse_field(cost, path, line_index, "cost")?,
            surface: surface.to_owned(),
            order,
        };
        validate_entry(&entry).with_context(|| format!("{}:{}", path.display(), line_index + 1))?;
        output.push(entry);
    }
    Ok(())
}

fn parse_field<T>(value: &str, path: &Path, line_index: usize, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value.parse().with_context(|| {
        format!(
            "{}:{}: invalid {name} value {value:?}",
            path.display(),
            line_index + 1
        )
    })
}

fn validate_entry(entry: &DictionaryEntry) -> Result<()> {
    if entry.reading.is_empty() || entry.surface.is_empty() {
        bail!("reading and surface must not be empty");
    }
    if entry.cost < 0 {
        bail!("cost must not be negative");
    }
    if !(1..=3).contains(&entry.order) {
        bail!("dictionary order must be 1, 2, or 3");
    }
    if entry
        .reading
        .chars()
        .chain(entry.surface.chars())
        .any(|character| {
            character == '\t' || character == '\n' || character == '\r' || character.is_control()
        })
    {
        bail!("reading and surface must not contain tabs, newlines, or controls");
    }
    Ok(())
}

/// Loads one uncompressed dictionary stream. Primarily useful for embedding and tests.
pub fn load_dictionary_text(reader: impl Read, order: u8) -> Result<Dictionary> {
    let mut entries = Vec::new();
    load_reader(
        BufReader::new(reader),
        Path::new("<dictionary>"),
        order,
        &mut entries,
    )?;
    Dictionary::from_entries(entries)
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn entry(reading: &str, surface: &str, cost: i32, order: u8) -> DictionaryEntry {
        DictionaryEntry {
            reading: reading.to_owned(),
            surface: surface.to_owned(),
            left_id: 10,
            right_id: 10,
            cost,
            order,
        }
    }

    fn converter(entries: Vec<DictionaryEntry>) -> Result<Converter> {
        Ok(Converter::new(Dictionary::from_entries(entries)?))
    }

    #[test]
    fn phrase_entries_supply_context() -> Result<()> {
        let converter = converter(vec![
            entry("わたし", "私", 100, 1),
            entry("は", "は", 100, 1),
            entry("がっこう", "学校", 100, 1),
            entry("わたしは", "私は", 50, 2),
            entry("はがっこう", "歯学校", 500, 2),
        ])?;
        let candidates = converter.convert("わたしはがっこう", 3)?;
        assert_eq!(candidates[0].text, "私は学校");
        assert_eq!(candidates[0].cost, 150);
        assert_eq!(candidates[0].segments.len(), 2);
        assert_eq!(candidates[0].segments[0].order, 2);
        Ok(())
    }

    #[test]
    fn homophones_are_returned_in_cost_order() -> Result<()> {
        let converter = converter(vec![
            entry("はし", "箸", 200, 1),
            entry("はし", "橋", 100, 1),
            entry("はし", "端", 300, 1),
        ])?;
        let candidates = converter.convert("はし", 3)?;
        assert_eq!(
            candidates
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            vec!["橋", "箸", "端"]
        );
        Ok(())
    }

    #[test]
    fn katakana_and_halfwidth_input_are_normalized() -> Result<()> {
        let converter = converter(vec![entry("がっこう", "学校", 100, 1)])?;
        assert_eq!(normalize_reading("ｶﾞｯｺｳ"), "がっこう");
        assert_eq!(converter.convert("ガッコウ", 1)?[0].text, "学校");
        assert_eq!(converter.convert("ｶﾞｯｺｳ", 1)?[0].text, "学校");
        Ok(())
    }

    #[test]
    fn unknown_characters_are_copied() -> Result<()> {
        let converter = converter(vec![entry("きょう", "今日", 100, 1)])?;
        let candidate = &converter.convert("きょう!", 1)?[0];
        assert_eq!(candidate.text, "今日!");
        assert!(candidate.segments.last().unwrap().is_unknown());
        Ok(())
    }

    #[test]
    fn duplicate_surface_keeps_the_cheapest_entry() -> Result<()> {
        let dictionary = Dictionary::from_entries(vec![
            entry("きょう", "今日", 500, 1),
            entry("きょう", "今日", 100, 2),
        ])?;
        assert_eq!(dictionary.len(), 1);
        assert_eq!(dictionary.entries()[0].cost, 100);
        assert_eq!(dictionary.entries()[0].order, 2);
        Ok(())
    }

    #[test]
    fn parses_mozc_five_column_text() -> Result<()> {
        let dictionary = load_dictionary_text(io::Cursor::new("きょう\t10\t20\t123\t今日\n"), 1)?;
        assert_eq!(dictionary.len(), 1);
        assert_eq!(dictionary.entries()[0].left_id, 10);
        assert_eq!(dictionary.entries()[0].right_id, 20);
        Ok(())
    }

    #[test]
    fn loads_plain_and_zstd_shards_from_a_directory() -> Result<()> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "japanese-corpus-converter-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&directory)?;
        fs::write(
            directory.join("mozc-unigram-00000.txt"),
            "きょう\t10\t10\t100\t今日\n",
        )?;
        fs::write(
            directory.join("mozc-english-unigram-00000.txt"),
            "こんぴゅーた\t10\t10\t12000\tcomputer\n",
        )?;
        let compressed = File::create(directory.join("mozc-bigram-00000.txt.zst"))?;
        let mut encoder = zstd::stream::write::Encoder::new(compressed, 1)?;
        encoder.write_all("きょうは\t10\t20\t50\t今日は\n".as_bytes())?;
        encoder.finish()?;

        let dictionary = Dictionary::load_dir(&directory)?;
        assert_eq!(dictionary.len(), 3);
        let converter = Converter::new(dictionary);
        assert_eq!(converter.convert("きょうは", 1)?[0].text, "今日は");
        assert_eq!(converter.convert("コンピュータ", 1)?[0].text, "computer");
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn malformed_rows_are_rejected() {
        let error = load_dictionary_text(io::Cursor::new("きょう\t10\t20\t今日\n"), 1).unwrap_err();
        assert!(error.to_string().contains("expected 5"));
    }

    #[test]
    fn empty_input_has_one_empty_candidate() -> Result<()> {
        let converter = converter(vec![entry("あ", "亜", 100, 1)])?;
        let candidates = converter.convert("", 3)?;
        assert_eq!(
            candidates,
            vec![Candidate {
                text: String::new(),
                cost: 0,
                segments: vec![]
            }]
        );
        Ok(())
    }
}
