from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

from tqdm.auto import tqdm

from suffix_eval.config import DatasetEvalConfig, EvalConfig
from suffix_eval.data import available_group_columns, iter_evaluation_rows, load_evaluation_dataset
from suffix_eval.generation import generate_suffix_batch
from suffix_eval.modeling import load_model_and_tokenizer
from suffix_eval.results import AggregateTracker, JsonlWriter, write_aggregate_results
from suffix_eval.scoring import ScoreResult, build_scorer


LOGGER = logging.getLogger("suffix_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate suffix generation on one or more Hugging Face datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


def batched(iterator, batch_size: int):  # type: ignore[no-untyped-def]
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def run_dataset(config: DatasetEvalConfig, suite_config: EvalConfig, loaded) -> None:  # type: ignore[no-untyped-def]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    row_results_path = config.output_dir / "row_results.jsonl"
    aggregate_path = config.output_dir / "aggregate_results.json"

    LOGGER.info("Loading dataset %s split %s", config.dataset_name, config.dataset_split)
    dataset = load_evaluation_dataset(config)

    scorer = build_scorer(config.mode, config.code_language)
    group_columns = available_group_columns(dataset, config.mode)
    aggregates = AggregateTracker(group_columns=group_columns)
    score_failures = 0
    generation_params = suite_config.generation.to_generation_kwargs()
    generation_params.update(
        {
            "max_new_tokens": suite_config.generation.max_new_tokens,
            "batch_size": suite_config.generation.batch_size,
            "device": str(loaded.device),
            "dtype": suite_config.generation.dtype,
            "greedy": suite_config.generation.greedy,
        }
    )

    LOGGER.info("Writing row-level results to %s", row_results_path)
    rows = iter_evaluation_rows(dataset, config)
    with JsonlWriter(row_results_path) as writer:
        progress = tqdm(
            batched(rows, suite_config.generation.batch_size),
            total=(len(dataset) + suite_config.generation.batch_size - 1)
            // suite_config.generation.batch_size,
            desc=f"Evaluating {config.label or config.dataset_name}",
        )
        for batch in progress:
            generated_batch = generate_suffix_batch(
                batch,
                loaded.model,
                loaded.tokenizer,
                loaded.device,
                suite_config.generation,
            )
            for generated in generated_batch:
                score_error = None
                try:
                    score = scorer.score(generated.generated_suffix, generated.row.suffix)
                except Exception as exc:
                    score_failures += 1
                    score_error = str(exc)
                    LOGGER.warning("Scoring failed for UUID %s: %s", generated.row.uuid, exc)
                    score = ScoreResult(metric=scorer.metric_name, value=0.0)
                writer.write_row(generated, score, generation_params, score_error=score_error)
                aggregates.add(score.value, generated.row.metadata)

    average_score, grouped_averages = aggregates.averages()
    write_aggregate_results(
        aggregate_path,
        config=config,
        metric=scorer.metric_name,
        num_examples=len(aggregates.scores),
        average_score=average_score,
        grouped_averages=grouped_averages,
        score_failures=score_failures,
    )
    LOGGER.info("Wrote aggregate results to %s", aggregate_path)
    LOGGER.info("Average %s: %.6f", scorer.metric_name, average_score)


def run(config: EvalConfig) -> None:
    LOGGER.info("Loading model %s", config.model_name)
    loaded = load_model_and_tokenizer(config)
    if len(loaded.tokenizer) > loaded.model.get_input_embeddings().num_embeddings:
        loaded.model.resize_token_embeddings(len(loaded.tokenizer))

    for dataset_config in config.datasets:
        run_dataset(dataset_config, config, loaded)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        config = EvalConfig.from_yaml(Path(args.config))
        run(config)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
