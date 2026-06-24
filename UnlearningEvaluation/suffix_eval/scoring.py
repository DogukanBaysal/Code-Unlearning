from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    metric: str
    value: float


class Scorer:
    metric_name: str

    def score(self, prediction: str, reference: str) -> ScoreResult:
        raise NotImplementedError


def normalize_percent_score(score: float) -> float:
    return max(0.0, min(1.0, float(score) / 100.0))


class BleuScorer(Scorer):
    metric_name = "bleu"

    def __init__(self) -> None:
        try:
            import sacrebleu
        except ImportError as exc:
            raise RuntimeError(
                "BLEU scoring requires sacrebleu. Install it with: pip install sacrebleu"
            ) from exc
        self._sacrebleu = sacrebleu

    def score(self, prediction: str, reference: str) -> ScoreResult:
        try:
            score = self._sacrebleu.sentence_bleu(
                prediction,
                [reference],
                smooth_method="exp",
            ).score
        except Exception as exc:
            raise RuntimeError(f"BLEU scoring failed: {exc}") from exc
        return ScoreResult(metric=self.metric_name, value=normalize_percent_score(score))


class ChrfScorer(Scorer):
    metric_name = "chrf"

    def __init__(self) -> None:
        try:
            import sacrebleu
        except ImportError as exc:
            raise RuntimeError(
                "chrF scoring requires sacrebleu. Install it with: pip install sacrebleu"
            ) from exc
        self._sacrebleu = sacrebleu

    def score(self, prediction: str, reference: str) -> ScoreResult:
        try:
            score = self._sacrebleu.sentence_chrf(prediction, [reference]).score
        except Exception as exc:
            raise RuntimeError(f"chrF scoring failed: {exc}") from exc
        return ScoreResult(metric=self.metric_name, value=normalize_percent_score(score))


def build_scorer(mode: str, code_language: str) -> Scorer:
    if mode == "code":
        return BleuScorer()
    if mode == "secret":
        return ChrfScorer()
    raise ValueError(f"Unsupported mode: {mode!r}")
