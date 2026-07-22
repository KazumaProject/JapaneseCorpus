use std::io::{self, BufRead};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use japanese_corpus_converter::{ConvertOptions, Converter, Dictionary, DEFAULT_UNKNOWN_COST};

#[derive(Debug, Parser)]
#[command(
    about = "Convert kana with JapaneseCorpus Mozc-format n-gram dictionaries",
    after_help = "If READING is omitted, one reading is converted per standard-input line."
)]
struct Cli {
    /// Dictionary shard or directory. May be specified more than once.
    #[arg(short = 'd', long = "dictionary", required = true)]
    dictionaries: Vec<PathBuf>,

    /// Number of conversion candidates to print.
    #[arg(short = 'n', long, default_value_t = 1)]
    candidates: usize,

    /// Maximum hypotheses kept at each reading boundary.
    #[arg(long)]
    beam_width: Option<usize>,

    /// Cost of copying one character absent from the dictionary.
    #[arg(long, default_value_t = DEFAULT_UNKNOWN_COST)]
    unknown_cost: i64,

    /// Show the segmentation and per-segment costs on stderr.
    #[arg(long)]
    details: bool,

    /// Hiragana, katakana, or halfwidth-katakana reading to convert.
    reading: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !(1..=100).contains(&cli.candidates) {
        bail!("--candidates must be between 1 and 100");
    }
    let dictionary = Dictionary::load_paths(&cli.dictionaries)?;
    eprintln!("loaded {} dictionary entries", dictionary.len());
    let converter = Converter::new(dictionary);
    let options = ConvertOptions {
        candidate_limit: cli.candidates,
        beam_width: cli
            .beam_width
            .unwrap_or_else(|| cli.candidates.saturating_mul(8).max(64)),
        unknown_cost: cli.unknown_cost,
    };
    if options.beam_width < options.candidate_limit {
        bail!("--beam-width must be at least --candidates");
    }

    if let Some(reading) = cli.reading {
        print_conversion(&converter, &reading, &options, cli.details)?;
    } else {
        for line in io::stdin().lock().lines() {
            let line = line?;
            print_conversion(&converter, &line, &options, cli.details)?;
        }
    }
    Ok(())
}

fn print_conversion(
    converter: &Converter,
    reading: &str,
    options: &ConvertOptions,
    details: bool,
) -> Result<()> {
    let candidates = converter.convert_with_options(reading, options)?;
    if options.candidate_limit == 1 {
        println!("{}", candidates[0].text);
    } else {
        for (index, candidate) in candidates.iter().enumerate() {
            println!("{}\t{}\t{}", index + 1, candidate.cost, candidate.text);
        }
    }
    if details {
        for (index, candidate) in candidates.iter().enumerate() {
            eprintln!("candidate {}: cost={}", index + 1, candidate.cost);
            for segment in &candidate.segments {
                let source = if segment.is_unknown() {
                    "unknown".to_owned()
                } else {
                    format!("{}-gram", segment.order)
                };
                eprintln!(
                    "  {} -> {} ({source}, cost={})",
                    segment.reading, segment.surface, segment.cost
                );
            }
        }
    }
    Ok(())
}
