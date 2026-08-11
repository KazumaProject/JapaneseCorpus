use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use japanese_corpus_ngram::{build, build_homophones, BuildOptions, HomophoneBuildOptions};

#[derive(Debug, Parser)]
#[command(about = "Build Mozc-format 1/2/3-gram dictionaries from JapaneseCorpus")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Build {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long)]
        vibrato_dictionary: PathBuf,
        #[arg(long)]
        mozc_id_def: PathBuf,
        #[arg(long)]
        mozc_license: PathBuf,
        #[arg(long)]
        ipadic_copying: PathBuf,
        #[arg(long)]
        ipadic_notice: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        work_dir: PathBuf,
        #[arg(long, default_value_t = 16)]
        unigram_min_count: u64,
        #[arg(long, default_value_t = 32)]
        ngram_min_count: u64,
        #[arg(long, default_value_t = 2)]
        map_min_count: u32,
        #[arg(long, default_value_t = 800.0)]
        cost_scale: f64,
        #[arg(long, default_value_t = 1_000_000)]
        entries_per_shard: usize,
        #[arg(long)]
        mozc_commit: String,
        #[arg(long)]
        vibrato_dictionary_version: String,
        #[arg(long)]
        pipeline_commit: String,
    },
    BuildHomophones {
        #[arg(long, required = true)]
        input: Vec<PathBuf>,
        #[arg(long, required = true)]
        vibrato_dictionary: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 2)]
        min_group_size: usize,
        #[arg(long, default_value_t = 1)]
        min_candidate_count: u64,
        #[arg(long, default_value_t = 2)]
        min_natural_occurrences: u64,
        #[arg(long, default_value_t = 2)]
        min_natural_sentences: u64,
        #[arg(long, default_value = "unknown")]
        vibrato_dictionary_version: String,
        #[arg(long, default_value = "working-tree")]
        pipeline_commit: String,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build {
            input,
            vibrato_dictionary,
            mozc_id_def,
            mozc_license,
            ipadic_copying,
            ipadic_notice,
            output_dir,
            work_dir,
            unigram_min_count,
            ngram_min_count,
            map_min_count,
            cost_scale,
            entries_per_shard,
            mozc_commit,
            vibrato_dictionary_version,
            pipeline_commit,
        } => build(BuildOptions {
            inputs: input,
            vibrato_dictionary,
            mozc_id_def,
            mozc_license,
            ipadic_copying,
            ipadic_notice,
            output_dir,
            work_dir,
            unigram_min_count,
            ngram_min_count,
            map_min_count,
            cost_scale,
            entries_per_shard,
            mozc_commit,
            vibrato_dictionary_version,
            pipeline_commit,
        }),
        Command::BuildHomophones {
            input,
            vibrato_dictionary,
            output_dir,
            min_group_size,
            min_candidate_count,
            min_natural_occurrences,
            min_natural_sentences,
            vibrato_dictionary_version,
            pipeline_commit,
        } => build_homophones(HomophoneBuildOptions {
            inputs: input,
            vibrato_dictionary,
            output_dir,
            min_group_size,
            min_candidate_count,
            min_natural_occurrences,
            min_natural_sentences,
            vibrato_dictionary_version,
            pipeline_commit,
        }),
    }
}
