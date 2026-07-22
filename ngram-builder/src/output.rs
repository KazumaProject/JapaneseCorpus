use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::tokenizer::TokenKey;
use crate::{unpack_bigram, unpack_trigram};

pub struct OrderCounts {
    pub unigram: Vec<u64>,
    pub bigram: FxHashMap<u64, u64>,
    pub trigram: FxHashMap<u128, u64>,
    pub unigram_total: u64,
    pub bigram_total: u64,
    pub trigram_total: u64,
}

pub struct BuildManifestInput {
    pub input_assets: Vec<String>,
    pub documents: u64,
    pub sentences: u64,
    pub valid_tokens: u64,
    pub raw_unique_tokens: u64,
    pub unigram_min_count: u64,
    pub ngram_min_count: u64,
    pub map_min_count: u32,
    pub cost_scale: f64,
    pub entries_per_shard: usize,
    pub mozc_commit: String,
    pub vibrato_dictionary_version: String,
    pub pipeline_commit: String,
}

pub struct ThirdPartyFiles<'a> {
    pub mozc_id_def: &'a Path,
    pub mozc_license: &'a Path,
    pub ipadic_copying: &'a Path,
    pub ipadic_notice: &'a Path,
}

#[derive(Debug, Serialize)]
struct Asset {
    name: String,
    order: Option<u8>,
    entries: Option<u64>,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u8,
    format: Format,
    tokenizer: TokenizerMetadata<'a>,
    parameters: Parameters,
    corpus: CorpusMetadata<'a>,
    orders: Vec<OrderMetadata>,
    assets: &'a [Asset],
}

#[derive(Serialize)]
struct Format {
    name: &'static str,
    columns: [&'static str; 5],
    encoding: &'static str,
    compression: &'static str,
    ngram_representation: &'static str,
    cost: &'static str,
}

#[derive(Serialize)]
struct TokenizerMetadata<'a> {
    implementation: &'static str,
    implementation_version: &'static str,
    dictionary: &'static str,
    dictionary_version: &'a str,
    mozc_context_id_commit: &'a str,
}

#[derive(Serialize)]
struct Parameters {
    unigram_min_count: u64,
    ngram_min_count: u64,
    map_min_count: u32,
    exact_count_guarantee: String,
    cost_scale: f64,
    entries_per_shard: usize,
}

#[derive(Serialize)]
struct CorpusMetadata<'a> {
    input_assets: &'a [String],
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
    raw_unique_tokens: u64,
    pipeline_commit: &'a str,
}

#[derive(Serialize)]
struct OrderMetadata {
    order: u8,
    total_observations: u64,
    retained_entries: u64,
}

pub fn write_release_files(
    output_dir: &Path,
    third_party: ThirdPartyFiles<'_>,
    vocabulary: &[TokenKey],
    counts: OrderCounts,
    input: BuildManifestInput,
) -> Result<usize> {
    let mut assets = Vec::new();

    let mut unigram_writer = ShardedWriter::new(output_dir, 1, input.entries_per_shard)?;
    for (id, token) in vocabulary.iter().enumerate() {
        let count = counts.unigram[id];
        if count < input.unigram_min_count {
            continue;
        }
        unigram_writer.write(&MozcRow {
            reading: &token.reading,
            left_id: token.left_id,
            right_id: token.right_id,
            cost: mozc_cost(count, counts.unigram_total, input.cost_scale),
            surface: &token.surface,
        })?;
    }
    let unigram_entries = unigram_writer.total_entries;
    assets.extend(unigram_writer.finish()?);

    let mut bigram_keys: Vec<u64> = counts
        .bigram
        .iter()
        .filter_map(|(key, count)| (*count >= input.ngram_min_count).then_some(*key))
        .collect();
    bigram_keys.sort_unstable();
    let mut bigram_writer = ShardedWriter::new(output_dir, 2, input.entries_per_shard)?;
    let mut reading = String::new();
    let mut surface = String::new();
    for key in bigram_keys {
        let (first, second) = unpack_bigram(key);
        let first = &vocabulary[first as usize];
        let second = &vocabulary[second as usize];
        reading.clear();
        reading.push_str(&first.reading);
        reading.push_str(&second.reading);
        surface.clear();
        surface.push_str(&first.surface);
        surface.push_str(&second.surface);
        bigram_writer.write(&MozcRow {
            reading: &reading,
            left_id: first.left_id,
            right_id: second.right_id,
            cost: mozc_cost(counts.bigram[&key], counts.bigram_total, input.cost_scale),
            surface: &surface,
        })?;
    }
    let bigram_entries = bigram_writer.total_entries;
    assets.extend(bigram_writer.finish()?);

    let mut trigram_keys: Vec<u128> = counts
        .trigram
        .iter()
        .filter_map(|(key, count)| (*count >= input.ngram_min_count).then_some(*key))
        .collect();
    trigram_keys.sort_unstable();
    let mut trigram_writer = ShardedWriter::new(output_dir, 3, input.entries_per_shard)?;
    for key in trigram_keys {
        let (first, second, third) = unpack_trigram(key);
        let first = &vocabulary[first as usize];
        let second = &vocabulary[second as usize];
        let third = &vocabulary[third as usize];
        reading.clear();
        reading.push_str(&first.reading);
        reading.push_str(&second.reading);
        reading.push_str(&third.reading);
        surface.clear();
        surface.push_str(&first.surface);
        surface.push_str(&second.surface);
        surface.push_str(&third.surface);
        trigram_writer.write(&MozcRow {
            reading: &reading,
            left_id: first.left_id,
            right_id: third.right_id,
            cost: mozc_cost(counts.trigram[&key], counts.trigram_total, input.cost_scale),
            surface: &surface,
        })?;
    }
    let trigram_entries = trigram_writer.total_entries;
    assets.extend(trigram_writer.finish()?);

    let id_def_output = output_dir.join("mozc-id.def");
    fs::copy(third_party.mozc_id_def, &id_def_output)?;
    assets.push(asset_for_file(&id_def_output, None, None)?);

    for (source, name) in [
        (third_party.mozc_license, "MOZC-LICENSE"),
        (third_party.ipadic_copying, "IPADIC-COPYING"),
        (third_party.ipadic_notice, "IPADIC-NOTICE"),
    ] {
        let output = output_dir.join(name);
        fs::copy(source, &output)?;
        assets.push(asset_for_file(&output, None, None)?);
    }

    let readme_path = output_dir.join("MOZC-DICTIONARY-README.md");
    fs::write(
        &readme_path,
        dictionary_readme(
            input.unigram_min_count,
            input.ngram_min_count,
            input.cost_scale,
        ),
    )?;
    assets.push(asset_for_file(&readme_path, None, None)?);

    assets.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let manifest = Manifest {
        schema_version: 1,
        format: Format {
            name: "Mozc system dictionary source",
            columns: ["reading", "left_id", "right_id", "cost", "surface"],
            encoding: "UTF-8/LF",
            compression: "Zstandard",
            ngram_representation: "Adjacent tokens are concatenated into one Mozc phrase entry",
            cost: "round(-ln(count / total_order_observations) * cost_scale), clamped to 1..32767",
        },
        tokenizer: TokenizerMetadata {
            implementation: "Vibrato",
            implementation_version: "0.5.2",
            dictionary: "IPADIC for Vibrato",
            dictionary_version: &input.vibrato_dictionary_version,
            mozc_context_id_commit: &input.mozc_commit,
        },
        parameters: Parameters {
            unigram_min_count: input.unigram_min_count,
            ngram_min_count: input.ngram_min_count,
            map_min_count: input.map_min_count,
            exact_count_guarantee: format!(
                "All retained n-grams are recounted over all {} assets. Candidate discovery is lossless at count >= {} because {} > {} * ({} - 1).",
                input.input_assets.len(),
                input.ngram_min_count,
                input.ngram_min_count,
                input.input_assets.len(),
                input.map_min_count
            ),
            cost_scale: input.cost_scale,
            entries_per_shard: input.entries_per_shard,
        },
        corpus: CorpusMetadata {
            input_assets: &input.input_assets,
            documents: input.documents,
            sentences: input.sentences,
            valid_tokens: input.valid_tokens,
            raw_unique_tokens: input.raw_unique_tokens,
            pipeline_commit: &input.pipeline_commit,
        },
        orders: vec![
            OrderMetadata {
                order: 1,
                total_observations: counts.unigram_total,
                retained_entries: unigram_entries,
            },
            OrderMetadata {
                order: 2,
                total_observations: counts.bigram_total,
                retained_entries: bigram_entries,
            },
            OrderMetadata {
                order: 3,
                total_observations: counts.trigram_total,
                retained_entries: trigram_entries,
            },
        ],
        assets: &assets,
    };
    let manifest_path = output_dir.join("ngram-manifest.json");
    let mut manifest_file = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.write_all(b"\n")?;

    let mut checksum_assets = assets;
    checksum_assets.push(asset_for_file(&manifest_path, None, None)?);
    checksum_assets.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let sums_path = output_dir.join("NGRAM-SHA256SUMS");
    let mut sums = File::create(&sums_path)?;
    for asset in &checksum_assets {
        writeln!(sums, "{}  {}", asset.sha256, asset.name)?;
    }
    Ok(checksum_assets.len() + 1)
}

fn mozc_cost(count: u64, total: u64, scale: f64) -> i32 {
    if count == 0 || total == 0 {
        return 32767;
    }
    let probability = count as f64 / total as f64;
    (-probability.ln() * scale).round().clamp(1.0, 32767.0) as i32
}

struct MozcRow<'a> {
    reading: &'a str,
    left_id: u16,
    right_id: u16,
    cost: i32,
    surface: &'a str,
}

struct ShardedWriter {
    output_dir: PathBuf,
    order: u8,
    entries_per_shard: usize,
    current: Option<CurrentShard>,
    assets: Vec<Asset>,
    total_entries: u64,
}

struct CurrentShard {
    path: PathBuf,
    encoder: zstd::stream::write::Encoder<'static, File>,
    entries: usize,
}

impl ShardedWriter {
    fn new(output_dir: &Path, order: u8, entries_per_shard: usize) -> Result<Self> {
        Ok(Self {
            output_dir: output_dir.to_owned(),
            order,
            entries_per_shard,
            current: None,
            assets: Vec::new(),
            total_entries: 0,
        })
    }

    fn write(&mut self, row: &MozcRow<'_>) -> Result<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.entries >= self.entries_per_shard)
        {
            self.finish_current()?;
        }
        if self.current.is_none() {
            let part = self.assets.len();
            let path = self
                .output_dir
                .join(format!("mozc-{}-{part:05}.txt.zst", order_name(self.order)));
            let mut encoder = zstd::stream::write::Encoder::new(File::create(&path)?, 10)?;
            encoder.multithread(0)?;
            self.current = Some(CurrentShard {
                path,
                encoder,
                entries: 0,
            });
        }
        let current = self.current.as_mut().expect("writer was opened");
        writeln!(
            current.encoder,
            "{}\t{}\t{}\t{}\t{}",
            row.reading, row.left_id, row.right_id, row.cost, row.surface
        )?;
        current.entries += 1;
        self.total_entries += 1;
        Ok(())
    }

    fn finish_current(&mut self) -> Result<()> {
        if let Some(current) = self.current.take() {
            let entries = current.entries as u64;
            current.encoder.finish()?;
            self.assets.push(asset_for_file(
                &current.path,
                Some(self.order),
                Some(entries),
            )?);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Asset>> {
        self.finish_current()?;
        Ok(self.assets)
    }
}

fn order_name(order: u8) -> &'static str {
    match order {
        1 => "unigram",
        2 => "bigram",
        3 => "trigram",
        _ => "ngram",
    }
}

fn asset_for_file(path: &Path, order: Option<u8>, entries: Option<u64>) -> Result<Asset> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(Asset {
        name: path
            .file_name()
            .context("asset path has no file name")?
            .to_string_lossy()
            .into_owned(),
        order,
        entries,
        bytes: path.metadata()?.len(),
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn dictionary_readme(unigram_min: u64, ngram_min: u64, cost_scale: f64) -> String {
    format!(
        "# Mozc-format n-gram dictionaries\n\n\
Each decompressed `mozc-*-NNNNN.txt.zst` file uses Mozc's five-column system\n\
dictionary source format:\n\n\
```text\n\
reading<TAB>left_id<TAB>right_id<TAB>cost<TAB>surface\n\
```\n\n\
Bigram and trigram rows are phrase entries: the adjacent token readings and\n\
surfaces are concatenated, the first token supplies `left_id`, and the last\n\
token supplies `right_id`. Context IDs correspond to the included\n\
`mozc-id.def`. Costs are `-ln(count / observations) * {cost_scale}` rounded and\n\
clamped to Mozc's positive 16-bit range.\n\n\
The complete corpus is scanned. Unigrams require count >= {unigram_min};\n\
bigrams and trigrams require count >= {ngram_min}. Candidate n-grams are\n\
recounted exactly over every input asset before filtering. See\n\
`ngram-manifest.json` for source versions, totals, parameters, and checksums.\n\n\
These files are system dictionary *source* files, not Mozc user-dictionary\n\
four-column exports. They can be supplied to Mozc's dictionary build tooling\n\
alongside the matching `id.def`.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequent_items_have_lower_cost() {
        assert!(mozc_cost(100, 1000, 800.0) < mozc_cost(10, 1000, 800.0));
        assert_eq!(mozc_cost(0, 1000, 800.0), 32767);
    }
}
