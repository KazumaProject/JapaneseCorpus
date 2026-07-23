use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use japanese_corpus_converter::ajimee::{evaluate, load_items, EvaluationSummary};
use japanese_corpus_converter::{Converter, Dictionary};
use serde::Serialize;
use sha2::{Digest, Sha256};

const DATASET_PATH: &str = "JWTD_v2/v1/evaluation_items.json";
const DATASET_LICENSE: &str = "CC-BY-SA-3.0";

#[derive(Debug, Parser)]
#[command(about = "Evaluate conversion accuracy with AJIMEE-Bench")]
struct Arguments {
    /// Dictionary file or directory. May be repeated.
    #[arg(short = 'd', long = "dictionary", required = true)]
    dictionaries: Vec<PathBuf>,

    /// AJIMEE-Bench JWTD_v2/v1/evaluation_items.json.
    #[arg(long)]
    dataset: PathBuf,

    /// Aggregate JSON report destination.
    #[arg(long)]
    output: PathBuf,

    /// Pinned AJIMEE-Bench source commit.
    #[arg(long)]
    benchmark_commit: String,

    #[arg(long, default_value = "https://github.com/azooKey/AJIMEE-Bench")]
    benchmark_repository: String,

    #[arg(long, default_value_t = 200)]
    expected_items: usize,

    /// Fail after writing the report if overall Accuracy@1 is below this value.
    #[arg(long)]
    minimum_accuracy_at_1: Option<f64>,

    /// Fail after writing the report if overall mean MinCER is above this value.
    #[arg(long)]
    maximum_mean_min_cer: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u8,
    benchmark: Benchmark,
    engine: Engine,
    metrics: EvaluationSummary,
}

#[derive(Debug, Serialize)]
struct Benchmark {
    name: &'static str,
    dataset: &'static str,
    repository_url: String,
    commit: String,
    sha256: String,
    license: &'static str,
    items: usize,
}

#[derive(Debug, Serialize)]
struct Engine {
    candidate_limit: usize,
    context_mode: &'static str,
    dictionary_entries: usize,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    validate_thresholds(&arguments)?;

    let mut dataset_file = File::open(&arguments.dataset)
        .with_context(|| format!("failed to open dataset {}", arguments.dataset.display()))?;
    let mut dataset_bytes = Vec::new();
    dataset_file.read_to_end(&mut dataset_bytes)?;
    let dataset_sha256 = format!("{:x}", Sha256::digest(&dataset_bytes));
    let items = load_items(dataset_bytes.as_slice())?;
    if items.len() != arguments.expected_items {
        bail!(
            "expected {} AJIMEE-Bench items, found {}",
            arguments.expected_items,
            items.len()
        );
    }

    let dictionary = Dictionary::load_paths(&arguments.dictionaries)?;
    let dictionary_entries = dictionary.len();
    let metrics = evaluate(&Converter::new(dictionary), &items)?;
    let report = Report {
        schema_version: 1,
        benchmark: Benchmark {
            name: "AJIMEE-Bench",
            dataset: DATASET_PATH,
            repository_url: arguments.benchmark_repository.clone(),
            commit: arguments.benchmark_commit.clone(),
            sha256: dataset_sha256,
            license: DATASET_LICENSE,
            items: items.len(),
        },
        engine: Engine {
            candidate_limit: 1,
            context_mode: "ignored",
            dictionary_entries,
        },
        metrics,
    };

    if let Some(parent) = arguments.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut encoded = serde_json::to_vec_pretty(&report)?;
    encoded.push(b'\n');
    fs::write(&arguments.output, encoded)
        .with_context(|| format!("failed to write {}", arguments.output.display()))?;

    eprintln!(
        "AJIMEE-Bench: {}/{} Accuracy@1={:.6} mean MinCER={:.6}",
        report.metrics.overall.correct_at_1,
        report.metrics.overall.items,
        report.metrics.overall.accuracy_at_1,
        report.metrics.overall.mean_min_cer,
    );
    enforce_thresholds(&arguments, &report.metrics)
}

fn validate_thresholds(arguments: &Arguments) -> Result<()> {
    if arguments.expected_items == 0 {
        bail!("expected_items must be positive");
    }
    if let Some(minimum) = arguments.minimum_accuracy_at_1 {
        if !(0.0..=1.0).contains(&minimum) {
            bail!("minimum_accuracy_at_1 must be between 0 and 1");
        }
    }
    if let Some(maximum) = arguments.maximum_mean_min_cer {
        if maximum < 0.0 || !maximum.is_finite() {
            bail!("maximum_mean_min_cer must be finite and non-negative");
        }
    }
    Ok(())
}

fn enforce_thresholds(arguments: &Arguments, metrics: &EvaluationSummary) -> Result<()> {
    if let Some(minimum) = arguments.minimum_accuracy_at_1 {
        if metrics.overall.accuracy_at_1 < minimum {
            bail!(
                "AJIMEE-Bench Accuracy@1 {:.6} is below required {:.6}",
                metrics.overall.accuracy_at_1,
                minimum
            );
        }
    }
    if let Some(maximum) = arguments.maximum_mean_min_cer {
        if metrics.overall.mean_min_cer > maximum {
            bail!(
                "AJIMEE-Bench mean MinCER {:.6} exceeds allowed {:.6}",
                metrics.overall.mean_min_cer,
                maximum
            );
        }
    }
    Ok(())
}
