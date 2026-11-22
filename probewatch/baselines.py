"""Scorer interfaces for comparing detectors on the same turn logs."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .detector import Detector


class TurnScorer(Protocol):
    name: str

    def score_turns(self, turns: list[dict]) -> np.ndarray: ...


@dataclass
class DetectorScorer:
    name: str
    detector: Detector

    def score_turns(self, turns: list[dict]) -> np.ndarray:
        scores, _ = self.detector.turn_scores(turns)
        return scores
