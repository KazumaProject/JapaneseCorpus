use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::RwLock;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use vibrato::Tokenizer;

// Three retained tokens must remain below Mozc's 1024-byte conversion-key
// limit even after their readings are concatenated into a trigram phrase.
const MAX_TOKEN_FIELD_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct TokenKey {
    pub surface: String,
    pub reading: String,
    pub left_id: u16,
    pub right_id: u16,
}

impl Ord for TokenKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.reading, &self.surface, self.left_id, self.right_id).cmp(&(
            &other.reading,
            &other.surface,
            other.left_id,
            other.right_id,
        ))
    }
}

impl PartialOrd for TokenKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct IdEntry {
    id: u16,
    fields: [String; 7],
}

#[derive(Debug)]
pub struct MozcIdMap {
    entries: Vec<IdEntry>,
    exact: HashMap<String, u16>,
    generic_noun_id: u16,
    cache: RwLock<HashMap<String, u16>>,
}

impl MozcIdMap {
    pub fn load(path: &Path) -> Result<Self> {
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("opening Mozc id.def {}", path.display()))?,
        );
        let mut entries = Vec::new();
        let mut exact: HashMap<String, u16> = HashMap::new();
        for (line_number, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, feature) = line
                .split_once(' ')
                .with_context(|| format!("malformed id.def line {}: {line}", line_number + 1))?;
            let id: u16 = id.parse()?;
            let mut csv_reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(feature.as_bytes());
            let record = csv_reader
                .records()
                .next()
                .transpose()?
                .with_context(|| format!("empty id.def feature at line {}", line_number + 1))?;
            let parts: Vec<String> = record.iter().map(str::to_owned).collect();
            let fields: [String; 7] = parts.try_into().map_err(|parts: Vec<String>| {
                anyhow::anyhow!(
                    "id.def line {} has {} fields instead of 7",
                    line_number + 1,
                    parts.len()
                )
            })?;
            exact
                .entry(fields.join("\u{1f}"))
                .and_modify(|existing| *existing = (*existing).min(id))
                .or_insert(id);
            entries.push(IdEntry { id, fields });
        }
        if entries.is_empty() {
            bail!("Mozc id.def is empty");
        }
        let generic_noun_id = best_match(&entries, &["名詞", "一般", "*", "*", "*", "*", "*"])
            .context("Mozc id.def has no generic noun context ID")?;
        Ok(Self {
            entries,
            exact,
            generic_noun_id,
            cache: RwLock::new(HashMap::new()),
        })
    }

    fn resolve(&self, feature: &str) -> u16 {
        if let Some(id) = self.cache.read().expect("ID cache poisoned").get(feature) {
            return *id;
        }
        let id = self.resolve_uncached(feature);
        self.cache
            .write()
            .expect("ID cache poisoned")
            .insert(feature.to_owned(), id);
        id
    }

    fn resolve_uncached(&self, feature: &str) -> u16 {
        let parts: Vec<&str> = feature.split(',').collect();
        if parts.len() < 7 || parts[0] == "UNK" {
            return self.generic_noun_id;
        }

        let exact_key = parts[..7].join("\u{1f}");
        if let Some(id) = self.exact.get(&exact_key) {
            return *id;
        }

        let attempts = [
            [
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], "*",
            ],
            [parts[0], parts[1], parts[2], parts[3], "*", "*", "*"],
            [parts[0], parts[1], parts[2], "*", "*", "*", "*"],
            [parts[0], parts[1], "*", "*", "*", "*", "*"],
            [parts[0], "*", "*", "*", "*", "*", "*"],
        ];
        for attempt in attempts {
            if let Some(id) = self.exact.get(&attempt.join("\u{1f}")) {
                return *id;
            }
        }
        best_match(&self.entries, &parts[..7]).unwrap_or(self.generic_noun_id)
    }
}

fn best_match(entries: &[IdEntry], feature: &[&str]) -> Option<u16> {
    entries
        .iter()
        .filter_map(|entry| {
            let mut score = 0u16;
            for (index, (expected, actual)) in entry.fields.iter().zip(feature).enumerate() {
                if expected == "*" {
                    continue;
                }
                if expected != actual {
                    return None;
                }
                score += if index == 0 { 16 } else { 1 };
            }
            Some((score, entry.id))
        })
        .max_by(|left, right| match left.0.cmp(&right.0) {
            Ordering::Equal => right.1.cmp(&left.1),
            order => order,
        })
        .map(|(_, id)| id)
}

#[derive(Debug)]
struct MorphToken {
    key: TokenKey,
    pos: String,
    subpos: String,
    subsubpos: String,
}

pub struct TokenSequenceBuilder<'a> {
    tokenizer: &'a Tokenizer,
    id_map: &'a MozcIdMap,
}

impl<'a> TokenSequenceBuilder<'a> {
    pub fn new(tokenizer: &'a Tokenizer, id_map: &'a MozcIdMap) -> Self {
        Self { tokenizer, id_map }
    }

    pub fn tokenize(&self, sentence: &str) -> Result<Vec<Vec<TokenKey>>> {
        let mut worker = self.tokenizer.new_worker();
        worker.reset_sentence(sentence);
        worker.tokenize();

        let mut result = Vec::new();
        let mut contiguous = Vec::<MorphToken>::new();
        for index in 0..worker.num_tokens() {
            let token = worker.token(index);
            let feature = token.feature();
            let parts: Vec<&str> = feature.split(',').collect();
            let pos = parts.first().copied().unwrap_or("UNK");
            let surface = token.surface();
            let reading_raw = parts.get(7).copied().unwrap_or("*");
            let reading = if reading_raw == "*" {
                if surface.chars().all(is_kana) {
                    katakana_to_hiragana(surface)
                } else {
                    flush_contiguous(&mut contiguous, &mut result);
                    continue;
                }
            } else {
                katakana_to_hiragana(reading_raw)
            };

            if pos == "記号"
                || !contains_japanese_letter(surface)
                || !valid_dictionary_text(surface)
                || !valid_reading(&reading)
            {
                flush_contiguous(&mut contiguous, &mut result);
                continue;
            }

            let context_id = self.id_map.resolve(feature);
            contiguous.push(MorphToken {
                key: TokenKey {
                    surface: surface.to_owned(),
                    reading,
                    left_id: context_id,
                    right_id: context_id,
                },
                pos: pos.to_owned(),
                subpos: parts.get(1).copied().unwrap_or("UNK").to_owned(),
                subsubpos: parts.get(2).copied().unwrap_or("UNK").to_owned(),
            });
        }
        flush_contiguous(&mut contiguous, &mut result);
        Ok(result)
    }
}

fn flush_contiguous(tokens: &mut Vec<MorphToken>, output: &mut Vec<Vec<TokenKey>>) {
    if tokens.is_empty() {
        return;
    }
    output.push(merge_terms(tokens));
    tokens.clear();
}

// This follows Akaza's IPADIC segmentation policy: suffixes, conjunctive
// particles, and auxiliary sequences are kept as conversion-sized units.
fn merge_terms(tokens: &[MorphToken]) -> Vec<TokenKey> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let mut key = tokens[index].key.clone();
        let mut previous = &tokens[index];
        let mut next = index + 1;
        while next < tokens.len() {
            let current = &tokens[next];
            let merge = (current.pos == "助動詞"
                && (previous.pos == "動詞" || previous.pos == "助動詞"))
                || current.subpos == "接続助詞"
                || current.subpos == "接尾";
            if !merge {
                break;
            }
            key.surface.push_str(&current.key.surface);
            if current.key.surface == "家"
                && current.key.reading == "か"
                && previous.subsubpos == "人名"
            {
                key.reading.push('け');
            } else {
                key.reading.push_str(&current.key.reading);
            }
            key.right_id = current.key.right_id;
            previous = current;
            next += 1;
        }
        result.push(key);
        index = next;
    }
    result
}

pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    for line in text.lines() {
        let mut start = 0;
        for (offset, character) in line.char_indices() {
            if matches!(character, '。' | '！' | '？' | '!' | '?') {
                let sentence = line[start..offset].trim();
                if !sentence.is_empty() {
                    result.push(sentence);
                }
                start = offset + character.len_utf8();
            }
        }
        let remainder = line[start..].trim();
        if !remainder.is_empty() {
            result.push(remainder);
        }
    }
    result
}

fn katakana_to_hiragana(input: &str) -> String {
    input
        .chars()
        .map(|character| match character {
            '\u{30A1}'..='\u{30F6}' | '\u{30FD}'..='\u{30FE}' => {
                char::from_u32(character as u32 - 0x60).unwrap_or(character)
            }
            _ => character,
        })
        .collect()
}

fn contains_japanese_letter(input: &str) -> bool {
    input.chars().any(|character| {
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

fn is_kana(character: char) -> bool {
    matches!(
        character,
        '\u{3041}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}'
    )
}

fn valid_reading(input: &str) -> bool {
    !input.is_empty()
        && input
            .chars()
            .any(|character| matches!(character, '\u{3041}'..='\u{309F}' | 'ー'))
        && valid_dictionary_text(input)
}

fn valid_dictionary_text(input: &str) -> bool {
    !input.is_empty()
        && input.len() <= MAX_TOKEN_FIELD_BYTES
        && !input.chars().any(|character| {
            character == '\t' || character == '\n' || character == '\r' || character.is_control()
        })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn sentence_boundaries_do_not_cross_documents() {
        assert_eq!(
            split_sentences("今日は晴れ。明日も晴れ！\n最終行"),
            vec!["今日は晴れ", "明日も晴れ", "最終行"]
        );
    }

    #[test]
    fn katakana_readings_are_normalized() {
        assert_eq!(katakana_to_hiragana("ガッコーヽ"), "がっこーゝ");
    }

    #[test]
    fn id_def_exact_and_fallback_matching() -> Result<()> {
        let directory =
            std::env::temp_dir().join(format!("japanese-corpus-id-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("id.def");
        let mut file = File::create(&path)?;
        writeln!(file, "10 名詞,一般,*,*,*,*,*")?;
        writeln!(file, "20 名詞,固有名詞,人名,姓,*,*,山田")?;
        let map = MozcIdMap::load(&path)?;
        assert_eq!(
            map.resolve("名詞,固有名詞,人名,姓,*,*,山田,ヤマダ,ヤマダ"),
            20
        );
        assert_eq!(map.resolve("UNK"), 10);
        std::fs::remove_dir_all(directory)?;
        Ok(())
    }
}
