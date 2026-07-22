use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use japanese_corpus_jmdict::{build, BuildOptions};

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
            source_url,
            source_etag,
            source_last_modified,
            pipeline_commit,
        }),
    }
}
