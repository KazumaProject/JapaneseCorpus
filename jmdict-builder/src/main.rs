use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use japanese_corpus_jmdict::{build, BuildOptions, DictionaryKind};

#[derive(Debug, Parser)]
#[command(about = "Build hiragana-to-English Mozc dictionaries from JMdict")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Build {
        /// Official JMdict_e XML input, optionally gzip-compressed.
        #[arg(long)]
        input: PathBuf,

        /// Mozc id.def matching the target dictionary build.
        #[arg(long)]
        mozc_id_def: PathBuf,

        /// Saved copy of the EDRDG licence page.
        #[arg(long)]
        jmdict_license: PathBuf,

        #[arg(long)]
        output_dir: PathBuf,

        #[arg(long, default_value_t = 1_000_000)]
        entries_per_shard: usize,

        /// Base Mozc cost. JMdict priority and gloss order add penalties.
        #[arg(long, default_value_t = 12_000)]
        base_cost: i32,

        /// Output policy. `generic` preserves all English glosses; `direct-loanword`
        /// keeps only complete, pronunciation-matched loanword candidates.
        #[arg(long, value_enum, default_value_t = DictionaryKind::Generic)]
        dictionary_kind: DictionaryKind,

        /// Pinned CMUdict file used by the direct-loanword policy.
        #[arg(long)]
        pronunciation_dictionary: Option<PathBuf>,

        /// Immutable CMUdict revision recorded in the manifest.
        #[arg(long)]
        pronunciation_dictionary_commit: Option<String>,

        /// SHA-256 of the downloaded CMUdict file recorded in the manifest.
        #[arg(long)]
        pronunciation_dictionary_sha256: Option<String>,

        /// Exact reading/surface exceptions for candidates absent from CMUdict.
        #[arg(long)]
        direct_loanword_allowlist: Option<PathBuf>,

        #[arg(long)]
        source_url: String,

        #[arg(long, default_value = "")]
        source_etag: String,

        #[arg(long, default_value = "")]
        source_last_modified: String,

        #[arg(long)]
        pipeline_commit: String,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build {
            input,
            mozc_id_def,
            jmdict_license,
            output_dir,
            entries_per_shard,
            base_cost,
            dictionary_kind,
            pronunciation_dictionary,
            pronunciation_dictionary_commit,
            pronunciation_dictionary_sha256,
            direct_loanword_allowlist,
            source_url,
            source_etag,
            source_last_modified,
            pipeline_commit,
        } => build(BuildOptions {
            input,
            mozc_id_def,
            jmdict_license,
            output_dir,
            entries_per_shard,
            base_cost,
            dictionary_kind,
            pronunciation_dictionary,
            pronunciation_dictionary_commit,
            pronunciation_dictionary_sha256,
            direct_loanword_allowlist,
            source_url,
            source_etag,
            source_last_modified,
            pipeline_commit,
        }),
    }
}
