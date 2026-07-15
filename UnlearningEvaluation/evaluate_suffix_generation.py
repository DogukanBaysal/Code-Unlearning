from __future__ import annotations

import argparse
import logging
import sys
from itertools import islice
from pathlib import Path

from tqdm.auto import tqdm

from suffix_eval.config import DatasetEvalConfig, EvalConfig
from suffix_eval.data import (
    available_group_columns,
    iter_evaluation_rows,
    load_aggregate_uuid_filter,
    load_evaluation_dataset,
)
from suffix_eval.generation import generate_suffix_batch
from suffix_eval.modeling import load_model_and_tokenizer
from suffix_eval.results import (
    AggregateTracker,
    JsonlWriter,
    max_similarity_at_k,
    pass_at_k_cutoffs,
    write_aggregate_results,
)
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
    all_results_path = config.output_dir / "all_results.jsonl"
    aggregate_path = config.output_dir / "aggregate_results.json"
    filtered_aggregate_path = config.output_dir / "aggregate_results_filtered.json"

    LOGGER.info("Loading dataset %s split %s", config.dataset_name, config.dataset_split)
    dataset = load_evaluation_dataset(config)

    scorer = build_scorer(config.mode, config.code_language)
    group_columns = available_group_columns(dataset, config.mode)
    configured_pass_k = suite_config.generation.pass_k
    aggregates_by_pass_k = {
        cutoff: AggregateTracker(group_columns=group_columns)
        for cutoff in pass_at_k_cutoffs(configured_pass_k)
    }
    uuid_filter = None
    filtered_aggregates_by_pass_k = None
    if suite_config.aggregate_filter_csv is not None:
        uuid_filter = load_aggregate_uuid_filter(
            suite_config.aggregate_filter_csv,
            config,
        )
        filtered_aggregates_by_pass_k = {
            cutoff: AggregateTracker(group_columns=group_columns)
            for cutoff in pass_at_k_cutoffs(configured_pass_k)
        }
    score_failures = 0
    filtered_score_failures = 0
    generation_params = suite_config.generation.to_generation_kwargs()
    generation_params.update(
        {
            "max_new_tokens": suite_config.generation.max_new_tokens,
            "pass_k": suite_config.generation.pass_k,
            "batch_size": suite_config.generation.batch_size,
            "device": str(loaded.device),
            "dtype": suite_config.generation.dtype,
            "greedy": suite_config.generation.greedy,
        }
    )

    LOGGER.info("Writing worst-case row results to %s", row_results_path)
    LOGGER.info("Writing all generated results to %s", all_results_path)
    rows = iter_evaluation_rows(dataset, config)
    with JsonlWriter(row_results_path) as writer, JsonlWriter(all_results_path) as all_writer:
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
            for start in range(0, len(generated_batch), configured_pass_k):
                attempts = generated_batch[start : start + configured_pass_k]
                scored_attempts: list[tuple[ScoreResult, str | None]] = []
                for generated in attempts:
                    score_error = None
                    try:
                        score = scorer.score(generated.generated_suffix, generated.row.suffix)
                    except Exception as exc:
                        score_failures += 1
                        if uuid_filter is not None and generated.row.uuid in uuid_filter.uuids:
                            filtered_score_failures += 1
                        score_error = str(exc)
                        LOGGER.warning(
                            "Scoring failed for UUID %s pass %d: %s",
                            generated.row.uuid,
                            generated.pass_index,
                            exc,
                        )
                        score = ScoreResult(metric=scorer.metric_name, value=0.0)
                    scored_attempts.append((score, score_error))

                worst_case_index = max(
                    range(len(scored_attempts)),
                    key=lambda index: scored_attempts[index][0].value,
                )
                scores_at_k = max_similarity_at_k(
                    [score.value for score, _ in scored_attempts]
                )
                for cutoff, cutoff_score in scores_at_k.items():
                    aggregates_by_pass_k[cutoff].add(
                        cutoff_score,
                        attempts[0].row.metadata,
                    )
                    if (
                        uuid_filter is not None
                        and filtered_aggregates_by_pass_k is not None
                        and attempts[0].row.uuid in uuid_filter.uuids
                    ):
                        filtered_aggregates_by_pass_k[cutoff].add(
                            cutoff_score,
                            attempts[0].row.metadata,
                        )
                for index, (generated, (score, score_error)) in enumerate(
                    zip(attempts, scored_attempts)
                ):
                    all_writer.write_row(
                        generated,
                        score,
                        generation_params,
                        score_error=score_error,
                        is_worst_case=index == worst_case_index,
                    )

                worst_generated = attempts[worst_case_index]
                worst_score, worst_error = scored_attempts[worst_case_index]
                writer.write_row(
                    worst_generated,
                    worst_score,
                    generation_params,
                    score_error=worst_error,
                    is_worst_case=True,
                )
    pass_at_k_results = {
        cutoff: aggregates.averages()
        for cutoff, aggregates in aggregates_by_pass_k.items()
    }
    aggregates = aggregates_by_pass_k[configured_pass_k]
    average_score, grouped_averages = pass_at_k_results[configured_pass_k]
    write_aggregate_results(
        aggregate_path,
        config=config,
        metric=scorer.metric_name,
        num_examples=len(aggregates.scores),
        average_score=average_score,
        grouped_averages=grouped_averages,
        pass_k=configured_pass_k,
        num_generated_results=len(aggregates.scores) * configured_pass_k,
        score_failures=score_failures,
        pass_at_k_results=pass_at_k_results,
    )
    LOGGER.info("Wrote aggregate results to %s", aggregate_path)
    for cutoff, (cutoff_average, _) in pass_at_k_results.items():
        LOGGER.info(
            "Average %s pass@%d: %.6f",
            scorer.metric_name,
            cutoff,
            cutoff_average,
        )

    if uuid_filter is not None and filtered_aggregates_by_pass_k is not None:
        filtered_aggregates = filtered_aggregates_by_pass_k[configured_pass_k]
        if not filtered_aggregates.scores:
            raise ValueError(
                "No evaluated examples matched the UUID filter for "
                f"{config.label or config.dataset_name}"
            )
        filtered_pass_at_k_results = {
            cutoff: aggregates.averages()
            for cutoff, aggregates in filtered_aggregates_by_pass_k.items()
        }
        filtered_average, filtered_grouped_averages = filtered_pass_at_k_results[
            configured_pass_k
        ]
        write_aggregate_results(
            filtered_aggregate_path,
            config=config,
            metric=scorer.metric_name,
            num_examples=len(filtered_aggregates.scores),
            average_score=filtered_average,
            grouped_averages=filtered_grouped_averages,
            pass_k=configured_pass_k,
            num_generated_results=len(filtered_aggregates.scores) * configured_pass_k,
            score_failures=filtered_score_failures,
            pass_at_k_results=filtered_pass_at_k_results,
            uuid_filter=uuid_filter.as_dict(len(filtered_aggregates.scores)),
        )
        LOGGER.info("Wrote UUID-filtered aggregate results to %s", filtered_aggregate_path)


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
