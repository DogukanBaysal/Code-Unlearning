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
                tokenize="char",
            ).score
        except Exception as exc:
            raise RuntimeError(f"BLEU scoring failed: {exc}") from exc
        return ScoreResult(metric=self.metric_name, value=float(score) / 100.0)


class CodeBleuScorer(Scorer):
    metric_name = "codebleu"

    def __init__(self, language: str) -> None:
        try:
            from codebleu import calc_codebleu
        except ImportError as exc:
            raise RuntimeError(
                "CodeBLEU scoring requires codebleu. Install it with: pip install codebleu"
            ) from exc
        self._calc_codebleu = calc_codebleu
        self._language = language

    def score(self, prediction: str, reference: str) -> ScoreResult:
        try:
            result = self._calc_codebleu(
                references=[reference],
                predictions=[prediction],
                lang=self._language,
            )
            value = result["codebleu"] if isinstance(result, dict) else result
        except Exception as exc:
            raise RuntimeError(f"CodeBLEU scoring failed: {exc}") from exc
        return ScoreResult(metric=self.metric_name, value=float(value))


def build_scorer(mode: str, code_language: str) -> Scorer:
    if mode == "code":
        return CodeBleuScorer(language=code_language)
    if mode == "secret":
        return BleuScorer()
    raise ValueError(f"Unsupported mode: {mode!r}")
