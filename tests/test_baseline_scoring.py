import numpy as np
import pytest

from probewatch.baselines import DetectorScorer
from probewatch.detector import Detector
from probewatch.evaluate import first_crossing, threshold_for_fpr


def test_existing_scorer_and_threshold_ties():
    turns = [
        {"turn_index": 4, "prompt": "Hello", "response": "Hi"},
        {"turn_index": 9, "prompt": "A private code?", "response": None},
    ]
    detector = Detector(mode="prompt").fit([turns])
    scorer = DetectorScorer("features_prompt", detector)
    np.testing.assert_array_equal(scorer.score_turns(turns), detector.turn_scores(turns)[0])

    threshold = threshold_for_fpr(np.array([0.2] * 98 + [1.0, 1.0]), 0.01)
    assert threshold > 1.0
    assert first_crossing(turns, np.array([0.2, 1.0], dtype=np.float32), threshold) is None
    assert first_crossing(turns, np.array([0.2, threshold]), threshold) == 9


@pytest.mark.parametrize("peaks", [[], [np.nan], [np.inf]])
def test_invalid_calibration_peaks(peaks):
    with pytest.raises(ValueError):
        threshold_for_fpr(np.array(peaks), 0.01)
