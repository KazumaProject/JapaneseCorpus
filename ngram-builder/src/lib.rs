mod homophone;
mod output;
mod tokenizer;

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use vibrato::{Dictionary, Tokenizer};

use output::{
    write_release_files, BuildManifestInput, NgramOrderCounts, OrderCounts, ThirdPartyFiles,
};
use tokenizer::{split_sentences, MozcIdMap, TokenKey, TokenSequenceBuilder};

pub use homophone::{build_homophones, HomophoneBuildOptions};

const MISSING_VOCAB_ID: u32 = u32::MAX;
const TOKENIZE_BATCH_CHARACTERS: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct BuildOptions {
    pub inputs: Vec<PathBuf>,
    pub vibrato_dictionary: PathBuf,
    pub mozc_id_def: PathBuf,
    pub mozc_license: PathBuf,
    pub ipadic_copying: PathBuf,
    pub ipadic_notice: PathBuf,
    pub output_dir: PathBuf,
    pub work_dir: PathBuf,
    pub unigram_min_count: u64,
    pub ngram_min_count: u64,
    pub map_min_count: u32,
    pub cost_scale: f64,
    pub entries_per_shard: usize,
    pub mozc_commit: String,
    pub vibrato_dictionary_version: String,
    pub pipeline_commit: String,
}

#[derive(Debug, Default)]
struct CorpusStats {
    documents: u64,
    sentences: u64,
    valid_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct CorpusRecord {
    text: String,
}

#[derive(Default)]
struct Interner {
    ids: FxHashMap<TokenKey, u32>,
    tokens: Vec<TokenKey>,
    counts: Vec<u64>,
}

impl Interner {
    fn intern(&mut self, token: TokenKey) -> u32 {
        if let Some(id) = self.ids.get(&token) {
            let id = *id;
            self.counts[id as usize] += 1;
            return id;
        }
        let id = u32::try_from(self.tokens.len()).expect("raw token ID overflow");
        self.ids.insert(token.clone(), id);
        self.tokens.push(token);
        self.counts.push(1);
        id
    }
}

pub fn build(mut options: BuildOptions) -> Result<()> {
    validate_options(&mut options)?;
    fs::create_dir_all(&options.output_dir)?;
    fs::create_dir_all(&options.work_dir)?;

    let (interner, stats, token_files) = {
        let dictionary =
            Dictionary::read(File::open(&options.vibrato_dictionary).with_context(|| {
                format!(
                    "opening Vibrato dictionary {}",
                    options.vibrato_dictionary.display()
                )
            })?)?;
        let tokenizer = Tokenizer::new(dictionary);
        let id_map = MozcIdMap::load(&options.mozc_id_def)?;
        let sequence_builder = TokenSequenceBuilder::new(&tokenizer, &id_map);

        let mut interner = Interner::default();
        let mut stats = CorpusStats::default();
        let mut token_files = Vec::with_capacity(options.inputs.len());

        eprintln!("phase 1/4: tokenizing all corpus records and counting unigrams");
        for (index, input) in options.inputs.iter().enumerate() {
            let token_path = options.work_dir.join(format!("tokens-{index:05}.txt.zst"));
            tokenize_input(
                input,
                &token_path,
                &sequence_builder,
                &mut interner,
                &mut stats,
            )?;
            token_files.push(token_path);
        }
        (interner, stats, token_files)
    };

    let (vocabulary, raw_to_vocab, unigram_counts) = make_vocabulary(
        &interner.tokens,
        &interner.counts,
        options.unigram_min_count,
    )?;
    let raw_unique_tokens = interner.tokens.len() as u64;
    drop(interner);

    eprintln!(
        "retained {} of {} raw tokens at unigram count >= {}",
        vocabulary.len(),
        raw_unique_tokens,
        options.unigram_min_count
    );

    eprintln!("phase 2/4: discovering and recounting orders 2 through 5");
    let (bigram_candidates, trigram_candidates) =
        discover_lower_candidates(&token_files, &raw_to_vocab, options.map_min_count)?;
    let lower_orders = recount_lower_candidates(
        &token_files,
        &raw_to_vocab,
        bigram_candidates,
        trigram_candidates,
    )?;
    let higher_orders = (4u8..=5).map(|order| {
        eprintln!("discovering order {order} candidates");
        let candidates = discover_order_candidates(
            &token_files,
            &raw_to_vocab,
            options.map_min_count,
            usize::from(order),
        )?;
        eprintln!("order {order}: {} candidate keys", candidates.len());
        let recounted =
            recount_order_candidates(&token_files, &raw_to_vocab, candidates, usize::from(order))?;
        Ok(NgramOrderCounts {
            order,
            counts: recounted.counts,
            total: recounted.total,
        })
    });
    let higher_orders = lower_orders.into_iter().map(Ok).chain(higher_orders);

    eprintln!("phase 3/4: writing deterministic Mozc dictionary shards and analysis inputs");
    let assets = write_release_files(
        &options.output_dir,
        ThirdPartyFiles {
            mozc_id_def: &options.mozc_id_def,
            mozc_license: &options.mozc_license,
            ipadic_copying: &options.ipadic_copying,
            ipadic_notice: &options.ipadic_notice,
        },
        &vocabulary,
        OrderCounts {
            unigram: unigram_counts,
            unigram_total: stats.valid_tokens,
        },
        higher_orders,
        BuildManifestInput {
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
            raw_unique_tokens,
            unigram_min_count: options.unigram_min_count,
            ngram_min_count: options.ngram_min_count,
            map_min_count: options.map_min_count,
            cost_scale: options.cost_scale,
            entries_per_shard: options.entries_per_shard,
            mozc_commit: options.mozc_commit,
            vibrato_dictionary_version: options.vibrato_dictionary_version,
            pipeline_commit: options.pipeline_commit,
        },
    )?;
    eprintln!("completed {} release assets", assets);
    Ok(())
}

fn validate_options(options: &mut BuildOptions) -> Result<()> {
    if options.inputs.is_empty() {
        bail!("at least one --input is required");
    }
    options.inputs.sort();
    for input in &options.inputs {
        if !input.is_file() {
            bail!("input does not exist: {}", input.display());
        }
    }
    for (label, path) in [
        ("Vibrato dictionary", &options.vibrato_dictionary),
        ("Mozc id.def", &options.mozc_id_def),
        ("Mozc license", &options.mozc_license),
        ("IPADIC COPYING", &options.ipadic_copying),
        ("IPADIC NOTICE", &options.ipadic_notice),
    ] {
        if !path.is_file() {
            bail!("{label} does not exist: {}", path.display());
        }
    }
    if options.unigram_min_count == 0 || options.ngram_min_count == 0 {
        bail!("frequency thresholds must be positive");
    }
    if options.map_min_count < 2 {
        bail!("--map-min-count must be at least 2");
    }
    let maximum_undiscovered = options.inputs.len() as u64 * u64::from(options.map_min_count - 1);
    if options.ngram_min_count <= maximum_undiscovered {
        bail!(
            "ngram threshold {} does not guarantee exact candidate discovery across {} assets; it must be greater than {}",
            options.ngram_min_count,
            options.inputs.len(),
            maximum_undiscovered
        );
    }
    if !options.cost_scale.is_finite() || options.cost_scale <= 0.0 {
        bail!("--cost-scale must be finite and positive");
    }
    if options.entries_per_shard == 0 {
        bail!("--entries-per-shard must be positive");
    }
    Ok(())
}

fn tokenize_input(
    input_path: &Path,
    token_path: &Path,
    builder: &TokenSequenceBuilder<'_>,
    interner: &mut Interner,
    stats: &mut CorpusStats,
) -> Result<()> {
    eprintln!("tokenizing {}", input_path.display());
    let input_file = File::open(input_path)?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::new(input_file))?;
    let reader = BufReader::new(decoder);

    let output_file = File::create(token_path)?;
    let mut encoder = zstd::stream::write::Encoder::new(output_file, 6)?;
    encoder.multithread(0)?;

    let mut batch = Vec::<String>::new();
    let mut batch_characters = 0usize;
    for line in reader.lines() {
        let line = line?;
        let record: CorpusRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing a record in {}", input_path.display()))?;
        stats.documents += 1;
        for sentence in split_sentences(&record.text) {
            batch_characters += sentence.len();
            batch.push(sentence.to_owned());
            if batch_characters >= TOKENIZE_BATCH_CHARACTERS {
                write_tokenized_batch(&batch, builder, interner, stats, &mut encoder)?;
                batch.clear();
                batch_characters = 0;
            }
        }
        if stats.documents.is_multiple_of(10_000) {
            eprintln!(
                "  documents={} sentences={} valid_tokens={}",
                stats.documents, stats.sentences, stats.valid_tokens
            );
        }
    }
    if !batch.is_empty() {
        write_tokenized_batch(&batch, builder, interner, stats, &mut encoder)?;
    }
    encoder.finish()?;
    Ok(())
}

fn write_tokenized_batch(
    batch: &[String],
    builder: &TokenSequenceBuilder<'_>,
    interner: &mut Interner,
    stats: &mut CorpusStats,
    writer: &mut impl Write,
) -> Result<()> {
    let results: Vec<Result<Vec<Vec<TokenKey>>>> = batch
        .par_iter()
        .map(|sentence| builder.tokenize(sentence))
        .collect();
    for result in results {
        for sequence in result? {
            if sequence.is_empty() {
                continue;
            }
            stats.sentences += 1;
            stats.valid_tokens += sequence.len() as u64;
            let mut first = true;
            for token in sequence {
                let id = interner.intern(token);
                if !first {
                    writer.write_all(b" ")?;
                }
                write!(writer, "{id}")?;
                first = false;
            }
            writer.write_all(b"\n")?;
        }
    }
    Ok(())
}

fn make_vocabulary(
    raw_tokens: &[TokenKey],
    raw_counts: &[u64],
    min_count: u64,
) -> Result<(Vec<TokenKey>, Vec<u32>, Vec<u64>)> {
    let mut retained: Vec<(usize, &TokenKey, u64)> = raw_tokens
        .iter()
        .enumerate()
        .filter_map(|(raw_id, token)| {
            let count = raw_counts[raw_id];
            (count >= min_count).then_some((raw_id, token, count))
        })
        .collect();
    retained.sort_unstable_by(|left, right| left.1.cmp(right.1));
    if retained.len() > u32::MAX as usize {
        bail!("vocabulary exceeds u32 IDs");
    }

    let mut raw_to_vocab = vec![MISSING_VOCAB_ID; raw_tokens.len()];
    let mut vocabulary = Vec::with_capacity(retained.len());
    let mut counts = Vec::with_capacity(retained.len());
    for (vocab_id, (raw_id, token, count)) in retained.into_iter().enumerate() {
        raw_to_vocab[raw_id] = vocab_id as u32;
        vocabulary.push(token.clone());
        counts.push(count);
    }
    Ok((vocabulary, raw_to_vocab, counts))
}

fn discover_lower_candidates(
    token_files: &[PathBuf],
    raw_to_vocab: &[u32],
    map_min_count: u32,
) -> Result<(FxHashSet<[u32; 5]>, FxHashSet<[u32; 5]>)> {
    let mut bigram_candidates = FxHashSet::default();
    let mut trigram_candidates = FxHashSet::default();
    for token_file in token_files {
        eprintln!("candidate scan {}", token_file.display());
        let mut local_bigrams: FxHashMap<[u32; 5], u32> = FxHashMap::default();
        let mut local_trigrams: FxHashMap<[u32; 5], u32> = FxHashMap::default();
        scan_token_file(token_file, raw_to_vocab, |sequence| {
            for window in sequence.windows(2) {
                *local_bigrams.entry(pack_ngram(window)).or_default() += 1;
            }
            for window in sequence.windows(3) {
                *local_trigrams.entry(pack_ngram(window)).or_default() += 1;
            }
        })?;
        bigram_candidates.extend(
            local_bigrams
                .into_iter()
                .filter_map(|(key, count)| (count >= map_min_count).then_some(key)),
        );
        trigram_candidates.extend(
            local_trigrams
                .into_iter()
                .filter_map(|(key, count)| (count >= map_min_count).then_some(key)),
        );
    }
    Ok((bigram_candidates, trigram_candidates))
}

fn discover_order_candidates(
    token_files: &[PathBuf],
    raw_to_vocab: &[u32],
    map_min_count: u32,
    order: usize,
) -> Result<FxHashSet<[u32; 5]>> {
    let mut candidates = FxHashSet::default();
    for token_file in token_files {
        eprintln!("candidate scan {}", token_file.display());
        let mut local: FxHashMap<[u32; 5], u32> = FxHashMap::default();
        scan_token_file(token_file, raw_to_vocab, |sequence| {
            for window in sequence.windows(order) {
                *local.entry(pack_ngram(window)).or_default() += 1;
            }
        })?;
        candidates.extend(
            local
                .into_iter()
                .filter_map(|(key, count)| (count >= map_min_count).then_some(key)),
        );
    }
    Ok(candidates)
}

struct RecountedNgrams {
    counts: FxHashMap<[u32; 5], u64>,
    total: u64,
}

fn recount_lower_candidates(
    token_files: &[PathBuf],
    raw_to_vocab: &[u32],
    bigram_candidates: FxHashSet<[u32; 5]>,
    trigram_candidates: FxHashSet<[u32; 5]>,
) -> Result<Vec<NgramOrderCounts>> {
    let mut bigrams: FxHashMap<[u32; 5], u64> =
        bigram_candidates.into_iter().map(|key| (key, 0)).collect();
    let mut trigrams: FxHashMap<[u32; 5], u64> =
        trigram_candidates.into_iter().map(|key| (key, 0)).collect();
    let mut bigram_total = 0u64;
    let mut trigram_total = 0u64;
    for token_file in token_files {
        eprintln!("exact bigram/trigram recount {}", token_file.display());
        scan_token_file(token_file, raw_to_vocab, |sequence| {
            for window in sequence.windows(2) {
                bigram_total += 1;
                if let Some(count) = bigrams.get_mut(&pack_ngram(window)) {
                    *count += 1;
                }
            }
            for window in sequence.windows(3) {
                trigram_total += 1;
                if let Some(count) = trigrams.get_mut(&pack_ngram(window)) {
                    *count += 1;
                }
            }
        })?;
    }
    Ok(vec![
        NgramOrderCounts {
            order: 2,
            counts: bigrams,
            total: bigram_total,
        },
        NgramOrderCounts {
            order: 3,
            counts: trigrams,
            total: trigram_total,
        },
    ])
}

fn recount_order_candidates(
    token_files: &[PathBuf],
    raw_to_vocab: &[u32],
    candidates: FxHashSet<[u32; 5]>,
    order: usize,
) -> Result<RecountedNgrams> {
    let mut counts: FxHashMap<[u32; 5], u64> = candidates.into_iter().map(|key| (key, 0)).collect();
    let mut total = 0u64;
    for token_file in token_files {
        eprintln!("exact order-{order} recount {}", token_file.display());
        scan_token_file(token_file, raw_to_vocab, |sequence| {
            for window in sequence.windows(order) {
                total += 1;
                if let Some(count) = counts.get_mut(&pack_ngram(window)) {
                    *count += 1;
                }
            }
        })?;
    }
    Ok(RecountedNgrams { counts, total })
}

fn scan_token_file(
    path: &Path,
    raw_to_vocab: &[u32],
    mut consume: impl FnMut(&[u32]),
) -> Result<()> {
    let file = File::open(path)?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::new(file))?;
    let reader = BufReader::new(decoder);
    let mut contiguous = Vec::new();
    for line in reader.lines() {
        contiguous.clear();
        for raw_id in line?.split_ascii_whitespace() {
            let raw_id: usize = raw_id.parse()?;
            let vocab_id = *raw_to_vocab
                .get(raw_id)
                .with_context(|| format!("raw token ID {raw_id} is out of range"))?;
            if vocab_id == MISSING_VOCAB_ID {
                if !contiguous.is_empty() {
                    consume(&contiguous);
                    contiguous.clear();
                }
            } else {
                contiguous.push(vocab_id);
            }
        }
        if !contiguous.is_empty() {
            consume(&contiguous);
        }
    }
    Ok(())
}

fn pack_ngram(tokens: &[u32]) -> [u32; 5] {
    let mut key = [0; 5];
    key[..tokens.len()].copy_from_slice(tokens);
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_ids_round_trip() {
        assert_eq!(pack_ngram(&[1, u32::MAX]), [1, u32::MAX, 0, 0, 0]);
        assert_eq!(pack_ngram(&[4, 5, 6, 7, u32::MAX]), [4, 5, 6, 7, u32::MAX]);
    }

    #[test]
    fn discovers_and_exactly_recounts_orders_two_through_five() -> Result<()> {
        let directory = std::env::temp_dir().join(format!(
            "japanese-corpus-ngram-orders-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir_all(&directory)?;
        let mut files = Vec::new();
        for index in 0..2 {
            let path = directory.join(format!("tokens-{index}.txt.zst"));
            let mut encoder = zstd::stream::write::Encoder::new(File::create(&path)?, 1)?;
            encoder.write_all(b"0 1 2 3 4\n0 1 2 3 4\n")?;
            encoder.finish()?;
            files.push(path);
        }
        let raw_to_vocab: Vec<u32> = (0..5).collect();
        let (bigrams, trigrams) = discover_lower_candidates(&files, &raw_to_vocab, 2)?;
        let lower = recount_lower_candidates(&files, &raw_to_vocab, bigrams, trigrams)?;
        for (order, recounted) in lower.into_iter().enumerate() {
            let order = order + 2;
            assert_eq!(recounted.total, 2 * 2 * (6 - order) as u64);
            assert_eq!(
                recounted.counts.get(&pack_ngram(&[0, 1, 2, 3, 4][..order])),
                Some(&4)
            );
        }
        for order in 4..=5 {
            let candidates = discover_order_candidates(&files, &raw_to_vocab, 2, order)?;
            let recounted = recount_order_candidates(&files, &raw_to_vocab, candidates, order)?;
            assert_eq!(recounted.total, 2 * 2 * (6 - order) as u64);
            assert_eq!(
                recounted.counts.get(&pack_ngram(&[0, 1, 2, 3, 4][..order])),
                Some(&4)
            );
        }
        fs::remove_dir_all(directory)?;
        Ok(())
    }

    #[test]
    fn exact_discovery_bound_is_enforced() {
        let mut options = BuildOptions {
            inputs: (0..16).map(|n| PathBuf::from(format!("{n}"))).collect(),
            vibrato_dictionary: PathBuf::new(),
            mozc_id_def: PathBuf::new(),
            mozc_license: PathBuf::new(),
            ipadic_copying: PathBuf::new(),
            ipadic_notice: PathBuf::new(),
            output_dir: PathBuf::new(),
            work_dir: PathBuf::new(),
            unigram_min_count: 16,
            ngram_min_count: 16,
            map_min_count: 2,
            cost_scale: 800.0,
            entries_per_shard: 10,
            mozc_commit: String::new(),
            vibrato_dictionary_version: String::new(),
            pipeline_commit: String::new(),
        };
        // File existence is checked first, so exercise the mathematical condition directly.
        let maximum_undiscovered =
            options.inputs.len() as u64 * u64::from(options.map_min_count - 1);
        assert!(options.ngram_min_count <= maximum_undiscovered);
        options.ngram_min_count = maximum_undiscovered + 1;
        assert!(options.ngram_min_count > maximum_undiscovered);
    }
}
