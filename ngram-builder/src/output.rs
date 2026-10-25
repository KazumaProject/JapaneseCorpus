use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::tokenizer::TokenKey;

pub struct OrderCounts {
    pub unigram: Vec<u64>,
    pub unigram_total: u64,
}

pub struct NgramOrderCounts {
    pub order: u8,
    pub counts: FxHashMap<[u32; 5], u64>,
    pub total: u64,
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
    maximum_phrase_field_bytes: usize,
    skipped_overlong_entries: u64,
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
    higher_orders: impl IntoIterator<Item = Result<NgramOrderCounts>>,
    input: BuildManifestInput,
) -> Result<usize> {
    let mut assets = Vec::new();

    let mut unigram_writer = ShardedWriter::new(output_dir, 1, input.entries_per_shard)?;
    let mut unigram_analysis = AnalysisInputWriter::new(&output_dir.join("analysis-inputs"), 1)?;
    for (id, token) in vocabulary.iter().enumerate() {
        let count = counts.unigram[id];
        if count < input.unigram_min_count {
            continue;
        }
        let cost = mozc_cost(count, counts.unigram_total, input.cost_scale);
        unigram_writer.write(&MozcRow {
            reading: &token.reading,
            left_id: token.left_id,
            right_id: token.right_id,
            cost,
            surface: &token.surface,
        })?;
        unigram_analysis.write(&AnalysisInputRecord {
            order: 1,
            count,
            total_observations: counts.unigram_total,
            reading: &token.reading,
            surface: &token.surface,
            left_id: token.left_id,
            right_id: token.right_id,
            cost,
            source_tokens: vec![token.clone()],
        })?;
    }
    let unigram_entries = unigram_writer.total_entries;
    assets.extend(unigram_writer.finish()?);
    unigram_analysis.finish()?;

    let mut orders = vec![OrderMetadata {
        order: 1,
        total_observations: counts.unigram_total,
        retained_entries: unigram_entries,
    }];
    let mut skipped_overlong_entries = 0u64;
    let mut expected_order = 2;
    for higher_order in higher_orders {
        let higher_order = higher_order?;
        if higher_order.order != expected_order || !(2..=5).contains(&higher_order.order) {
            anyhow::bail!(
                "expected n-gram order {expected_order}, found {}",
                higher_order.order
            );
        }
        let (order_metadata, order_assets, skipped) =
            write_ngram_order(output_dir, vocabulary, higher_order, &input)?;
        orders.push(order_metadata);
        assets.extend(order_assets);
        skipped_overlong_entries += skipped;
        expected_order += 1;
    }
    if expected_order != 6 {
        anyhow::bail!("expected n-gram orders 2 through 5");
    }

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
            maximum_phrase_field_bytes: 1023,
            skipped_overlong_entries,
        },
        corpus: CorpusMetadata {
            input_assets: &input.input_assets,
            documents: input.documents,
            sentences: input.sentences,
            valid_tokens: input.valid_tokens,
            raw_unique_tokens: input.raw_unique_tokens,
            pipeline_commit: &input.pipeline_commit,
        },
        orders,
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

fn write_ngram_order(
    output_dir: &Path,
    vocabulary: &[TokenKey],
    order: NgramOrderCounts,
    input: &BuildManifestInput,
) -> Result<(OrderMetadata, Vec<Asset>, u64)> {
    let NgramOrderCounts {
        order,
        mut counts,
        total,
    } = order;
    let mut keys: Vec<[u32; 5]> = counts
        .iter()
        .filter_map(|(key, count)| (*count >= input.ngram_min_count).then_some(*key))
        .collect();
    keys.sort_unstable();

    let mut dictionary_writer = ShardedWriter::new(output_dir, order, input.entries_per_shard)?;
    let mut analysis_writer = AnalysisInputWriter::new(&output_dir.join("analysis-inputs"), order)?;
    let mut reading = String::new();
    let mut surface = String::new();
    let mut skipped_overlong = 0u64;
    for key in keys {
        let count = counts.remove(&key).expect("retained key exists");
        let source_tokens: Vec<&TokenKey> = key[..order as usize]
            .iter()
            .map(|id| &vocabulary[*id as usize])
            .collect();
        reading.clear();
        surface.clear();
        for token in &source_tokens {
            reading.push_str(&token.reading);
            surface.push_str(&token.surface);
        }
        if reading.len() > 1023 || surface.len() > 1023 {
            skipped_overlong += 1;
            continue;
        }
        let first = source_tokens
            .first()
            .expect("n-gram has at least two tokens");
        let last = source_tokens
            .last()
            .expect("n-gram has at least two tokens");
        let cost = mozc_cost(count, total, input.cost_scale);
        dictionary_writer.write(&MozcRow {
            reading: &reading,
            left_id: first.left_id,
            right_id: last.right_id,
            cost,
            surface: &surface,
        })?;
        analysis_writer.write(&AnalysisInputRecord {
            order,
            count,
            total_observations: total,
            reading: &reading,
            surface: &surface,
            left_id: first.left_id,
            right_id: last.right_id,
            cost,
            source_tokens: source_tokens.into_iter().cloned().collect(),
        })?;
    }
    let retained_entries = dictionary_writer.total_entries;
    let assets = dictionary_writer.finish()?;
    analysis_writer.finish()?;
    Ok((
        OrderMetadata {
            order,
            total_observations: total,
            retained_entries,
        },
        assets,
        skipped_overlong,
    ))
}

#[derive(Serialize)]
struct AnalysisInputRecord<'a> {
    order: u8,
    count: u64,
    total_observations: u64,
    reading: &'a str,
    surface: &'a str,
    left_id: u16,
    right_id: u16,
    cost: i32,
    source_tokens: Vec<TokenKey>,
}

struct AnalysisInputWriter {
    directory: PathBuf,
    order: u8,
    part: usize,
    entries: usize,
    encoder: Option<zstd::stream::write::Encoder<'static, File>>,
}

impl AnalysisInputWriter {
    const ENTRIES_PER_PART: usize = 100_000;

    fn new(directory: &Path, order: u8) -> Result<Self> {
        fs::create_dir_all(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
            order,
            part: 0,
            entries: 0,
            encoder: None,
        })
    }

    fn write(&mut self, record: &AnalysisInputRecord<'_>) -> Result<()> {
        if self.entries == Self::ENTRIES_PER_PART {
            self.finish_part()?;
        }
        if self.encoder.is_none() {
            let path = self
                .directory
                .join(format!("ngram-{}-{:05}.jsonl.zst", self.order, self.part));
            let mut encoder = zstd::stream::write::Encoder::new(File::create(path)?, 9)?;
            encoder.multithread(0)?;
            self.encoder = Some(encoder);
        }
        let encoder = self.encoder.as_mut().expect("analysis shard was opened");
        serde_json::to_writer(&mut *encoder, record)?;
        encoder.write_all(b"\n")?;
        self.entries += 1;
        Ok(())
    }

    fn finish_part(&mut self) -> Result<()> {
        if let Some(encoder) = self.encoder.take() {
            encoder.finish()?;
            self.part += 1;
            self.entries = 0;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.finish_part()
    }
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
        4 => "4gram",
        5 => "5gram",
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
N-gram rows are phrase entries: the adjacent token readings and\n\
surfaces are concatenated, the first token supplies `left_id`, and the last\n\
token supplies `right_id`. Context IDs correspond to the included\n\
`mozc-id.def`. Costs are `-ln(count / observations) * {cost_scale}` rounded and\n\
clamped to Mozc's positive 16-bit range.\n\n\
The complete corpus is scanned. Unigrams require count >= {unigram_min};\n\
orders 2 through 5 require count >= {ngram_min}. Candidate n-grams are\n\
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
