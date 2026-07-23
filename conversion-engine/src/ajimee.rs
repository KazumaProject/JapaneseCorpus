//! AJIMEE-Bench-compatible evaluation for Japanese input-method conversion.

use std::io::Read;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::Converter;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct EvaluationItem {
    pub index: String,
    pub context_text: String,
    pub input: String,
    pub expected_output: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricSet {
    pub items: usize,
    pub correct_at_1: usize,
    pub accuracy_at_1: f64,
    pub mean_min_cer: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct EvaluationSummary {
    pub overall: MetricSet,
    pub with_context: MetricSet,
    pub without_context: MetricSet,
}

#[derive(Default)]
struct Accumulator {
    items: usize,
    correct_at_1: usize,
    min_cer_sum: f64,
}

impl Accumulator {
    fn record(&mut self, correct: bool, minimum_cer: f64) {
        self.items += 1;
        self.correct_at_1 += usize::from(correct);
        self.min_cer_sum += minimum_cer;
    }

    fn finish(self) -> MetricSet {
        let divisor = self.items as f64;
        MetricSet {
            items: self.items,
            correct_at_1: self.correct_at_1,
            accuracy_at_1: if self.items == 0 {
                0.0
            } else {
                self.correct_at_1 as f64 / divisor
            },
            mean_min_cer: if self.items == 0 {
                0.0
            } else {
                self.min_cer_sum / divisor
            },
        }
    }
}

pub fn load_items(reader: impl Read) -> Result<Vec<EvaluationItem>> {
    let items: Vec<EvaluationItem> =
        serde_json::from_reader(reader).context("failed to parse AJIMEE-Bench dataset")?;
    if items.is_empty() {
        bail!("AJIMEE-Bench dataset contains no items");
    }
    for item in &items {
        if item.expected_output.is_empty() {
            bail!("AJIMEE-Bench item {} has no expected output", item.index);
        }
        if item.expected_output.iter().any(String::is_empty) {
            bail!(
                "AJIMEE-Bench item {} has an empty expected output",
                item.index
            );
        }
    }
    Ok(items)
}

/// Evaluates the first conversion candidate using AJIMEE-Bench's exact-match Accuracy@1
/// and minimum character error rate definitions.
///
/// The current converter has no left-context API. Context is therefore used only to
/// stratify the report, never as converter input.
pub fn evaluate(converter: &Converter, items: &[EvaluationItem]) -> Result<EvaluationSummary> {
    if items.is_empty() {
        bail!("AJIMEE-Bench dataset contains no items");
    }

    let mut overall = Accumulator::default();
    let mut with_context = Accumulator::default();
    let mut without_context = Accumulator::default();

    for item in items {
        if item.expected_output.is_empty() {
            bail!("AJIMEE-Bench item {} has no expected output", item.index);
        }
        let hypothesis = converter
            .convert(&item.input, 1)?
            .into_iter()
            .next()
            .context("converter returned no first candidate")?
            .text;
        let correct = accuracy_at_1(&item.expected_output, &hypothesis);
        let minimum_cer = min_cer(&item.expected_output, &hypothesis)?;
        overall.record(correct, minimum_cer);
        if item.context_text.is_empty() {
            without_context.record(correct, minimum_cer);
        } else {
            with_context.record(correct, minimum_cer);
        }
    }

    Ok(EvaluationSummary {
        overall: overall.finish(),
        with_context: with_context.finish(),
        without_context: without_context.finish(),
    })
}

pub fn accuracy_at_1(references: &[String], hypothesis: &str) -> bool {
    references.iter().any(|reference| reference == hypothesis)
}

pub fn min_cer(references: &[String], hypothesis: &str) -> Result<f64> {
    references
        .iter()
        .map(|reference| cer(reference, hypothesis))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .reduce(f64::min)
        .context("cannot calculate MinCER without a reference")
}

pub fn cer(reference: &str, hypothesis: &str) -> Result<f64> {
    let reference_length = reference.chars().count();
    if reference_length == 0 {
        bail!("cannot calculate CER for an empty reference");
    }
    Ok(character_distance(reference, hypothesis) as f64 / reference_length as f64)
}

pub fn character_distance(left: &str, right: &str) -> usize {
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];

    for (left_index, left_character) in left.chars().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            let substitution =
                previous[right_index] + usize::from(left_character != *right_character);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current[right_index + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dictionary, DictionaryEntry};

    fn converter() -> Converter {
        Converter::new(
            Dictionary::from_entries(vec![
                DictionaryEntry {
                    reading: "きょう".into(),
                    surface: "今日".into(),
                    left_id: 1,
                    right_id: 1,
                    cost: 100,
                    order: 1,
                },
                DictionaryEntry {
                    reading: "はし".into(),
                    surface: "橋".into(),
                    left_id: 1,
                    right_id: 1,
                    cost: 100,
                    order: 1,
                },
            ])
            .unwrap(),
        )
    }

    #[test]
    fn official_metrics_use_exact_match_and_best_reference() {
        let references = vec!["橋".to_owned(), "箸".to_owned()];
        assert!(accuracy_at_1(&references, "橋"));
        assert!(!accuracy_at_1(&references, "端"));
        assert_eq!(min_cer(&references, "橋").unwrap(), 0.0);
        assert_eq!(min_cer(&references, "端").unwrap(), 1.0);
    }

    #[test]
    fn character_error_rate_uses_unicode_characters() {
        assert_eq!(character_distance("コンピュータ", "こんぴゅーた"), 5);
        assert_eq!(character_distance("今日です", "今日"), 2);
        assert_eq!(cer("今日です", "今日").unwrap(), 0.5);
    }

    #[test]
    fn evaluation_is_stratified_by_context() {
        let items = vec![
            EvaluationItem {
                index: "1".into(),
                context_text: String::new(),
                input: "きょう".into(),
                expected_output: vec!["今日".into()],
            },
            EvaluationItem {
                index: "2".into(),
                context_text: "渡る".into(),
                input: "はし".into(),
                expected_output: vec!["箸".into(), "橋".into()],
            },
        ];
        let summary = evaluate(&converter(), &items).unwrap();
        assert_eq!(summary.overall.items, 2);
        assert_eq!(summary.overall.correct_at_1, 2);
        assert_eq!(summary.overall.accuracy_at_1, 1.0);
        assert_eq!(summary.with_context.items, 1);
        assert_eq!(summary.without_context.items, 1);
    }

    #[test]
    fn dataset_validation_rejects_missing_references() {
        let error = load_items(
            r#"[{"index":"1","context_text":"","input":"てすと","expected_output":[]}]"#.as_bytes(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("no expected output"));
    }
}
