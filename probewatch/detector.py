"""The detector: scores whole sessions from logs alone.

It never imports lab.py and drops the `label` field on load, so it can run on
any app's exported chat logs that have session_id, ts, prompt, response.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from . import ATLAS_IDS

SIGNALS = ["near_dup", "block_density", "strategy_switch", "topic_fixation", "velocity"]
DEFAULT_WEIGHTS = {"near_dup": 0.30, "block_density": 0.25, "strategy_switch": 0.20,
                   "topic_fixation": 0.15, "velocity": 0.10}

# Generic cues for "kinds of trick". Deliberately broad words, not the lab's templates.
TRICK_CUES = {
    "roleplay": r"\b(pretend|act as|you are now|role|character|game|actor)\b",
    "translation": r"\b(translate|in (french|spanish|german|italian)|how would you say)\b",
    "encoding": r"\b(backwards|base64|spell|encode|first letter|one letter)\b",
    "story_poem": r"\b(poem|story|song|chorus|lyrics)\b",
    "hypothetical": r"\b(hypothetical(ly)?|imagine|what if|thought experiment|in a world)\b",
    "override": r"\b(ignore|authori[sz]e|admin|developer|new policy|override)\b",
}
TRICK_RE = {k: re.compile(v, re.I) for k, v in TRICK_CUES.items()}
REFUSAL_RE = re.compile(r"\b(sorry|can't|cannot|unable|not able|won't)\b", re.I)

# What this app protects. In a real deployment you'd write this per app.
ASSET_TEXT = "secret password code hidden private confidential word phrase keep guard protect"

_vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                         alternate_sign=False, norm="l2")


def load_sessions(path: Path) -> dict[str, list[dict]]:
    """Load turns, drop lab-only labels, group and sort by session."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    with Path(path).open() as f:
        for line in f:
            t = json.loads(line)
            t.pop("label", None)
            for req in ("session_id", "turn_index", "ts", "prompt"):
                if req not in t:
                    raise ValueError(f"turn missing required field {req!r}")
            sessions[t["session_id"]].append(t)
    for turns in sessions.values():
        turns.sort(key=lambda t: t["turn_index"])
    return dict(sessions)


def load_labels(path: Path) -> dict[str, dict]:
    """Eval-only: per-session ground truth. Never used by the detector."""
    out: dict[str, dict] = {}
    with Path(path).open() as f:
        for line in f:
            t = json.loads(line)
            s = out.setdefault(t["session_id"], {"actor": t["label"]["actor"], "leak_turn": None,
                                                 "persona": t.get("meta", {}).get("persona")})
            if t["label"]["leaked"] and s["leak_turn"] is None:
                s["leak_turn"] = t["turn_index"]
    return out


def _embed(prompts: list[str]):
    return _vec.transform(prompts)  # sparse, L2-normalised rows


_ASSET = _embed([ASSET_TEXT])


def _is_blocked_or_refused(t: dict) -> bool:
    if t.get("blocked"):
        return True
    r = t.get("response")
    return r is None or bool(REFUSAL_RE.search(r))


def raw_signals(turns: list[dict], window: int = 10, mode: str = "session") -> np.ndarray:
    """Return an array of shape (n_turns, 5) of raw (uncalibrated) signal values.

    mode="session": the real detector, looks at the last `window` turns.
    mode="prompt":  baseline with no memory; each turn judged on its own,
                    using the same features (so the comparison is fair).
    """
    n = len(turns)
    E = _embed([t["prompt"] for t in turns])
    asset_sim = (E @ _ASSET.T).toarray().ravel()
    tricks = [{k for k, rx in TRICK_RE.items() if rx.search(t["prompt"])} for t in turns]
    bad = [_is_blocked_or_refused(t) for t in turns]
    out = np.zeros((n, len(SIGNALS)), dtype=np.float32)

    if mode == "prompt":
        out[:, 1] = bad
        out[:, 2] = [len(s) for s in tricks]
        out[:, 3] = asset_sim
        return out

    sim = (E @ E.T).toarray()
    times = [datetime.fromisoformat(t["ts"]) for t in turns]
    for i in range(n):
        lo = max(0, i - window + 1)
        # near_dup: for recent prompts, how close is each to its nearest predecessor?
        vals = [sim[j, max(0, j - 5):j].max() for j in range(max(lo, 1), i + 1)]
        out[i, 0] = float(np.mean(vals)) if vals else 0.0
        # block_density: blocked/refused turns in the window. (v0 used a consecutive
        # streak, which reset whenever the bot answered once; density is sturdier.)
        out[i, 1] = sum(bad[lo:i + 1])
        # strategy_switch: distinct trick kinds seen in the window
        out[i, 2] = len(set().union(*tricks[lo:i + 1]))
        # topic_fixation: how much the window circles the protected asset
        out[i, 3] = float(asset_sim[lo:i + 1].mean())
        # velocity: turns per minute across the window
        span = (times[i] - times[lo]).total_seconds()
        out[i, 4] = (i - lo) / (span / 60) if span > 0 else 0.0
    return out


@dataclass
class Detector:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    window: int = 10
    mode: str = "session"          # "session" or "prompt" (baseline)
    learned: bool = False          # v2: logistic regression with interaction terms
    coef: np.ndarray | None = None
    intercept: float = 0.0
    lo: np.ndarray | None = None   # benign p50 per signal
    hi: np.ndarray | None = None   # benign p99 per signal
    threshold: float = 1.0

    def _norm(self, raw: np.ndarray) -> np.ndarray:
        return np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1)

    @staticmethod
    def _features(norm: np.ndarray) -> np.ndarray:
        """Signals plus interactions: repetition or topic focus only matter alongside failure."""
        nd, bd, ss, tf, ve = norm.T
        return np.column_stack([norm, nd * bd, tf * bd, nd * tf])

    def turn_scores(self, turns: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        norm = self._norm(raw_signals(turns, self.window, self.mode))
        if self.learned:
            z = self._features(norm) @ self.coef + self.intercept
            return 1 / (1 + np.exp(-z)), norm
        w = np.array([self.weights.get(s, 0.0) for s in SIGNALS])
        w = w / w.sum() if w.sum() > 0 else w
        return norm @ w, norm

    def fit(self, benign_sessions: list[list[dict]], max_fpr: float = 0.02,
            attack_sessions: list[list[dict]] | None = None):
        """Calibrate on benign sessions, optionally learn weights, then pick the threshold."""
        raw = np.vstack([raw_signals(s, self.window, self.mode) for s in benign_sessions])
        self.lo = np.percentile(raw, 50, axis=0)
        self.hi = np.percentile(raw, 99, axis=0)
        self.hi = np.where(self.hi - self.lo < 1e-6, self.lo + 1.0, self.hi)
        if self.learned:
            from sklearn.linear_model import LogisticRegression
            Xb = [self._features(self._norm(raw_signals(s, self.window, self.mode))) for s in benign_sessions]
            # attack turns from turn 3 on: the pattern needs a few turns to exist
            Xa = [self._features(self._norm(raw_signals(s, self.window, self.mode)))[3:] for s in attack_sessions]
            X = np.vstack(Xb + [x for x in Xa if len(x)])
            y = np.r_[np.zeros(sum(len(x) for x in Xb)), np.ones(sum(len(x) for x in Xa))]
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, y)
            self.coef, self.intercept = clf.coef_[0], float(clf.intercept_[0])
        peaks = np.array([self.turn_scores(s)[0].max() for s in benign_sessions])
        # smallest threshold that keeps benign session false-positive rate <= max_fpr
        self.threshold = float(np.quantile(peaks, 1 - max_fpr, method="higher")) + 1e-6
        return self

    def first_alert(self, turns: list[dict]) -> dict | None:
        scores, norm = self.turn_scores(turns)
        hits = np.nonzero(scores >= self.threshold)[0]
        if len(hits) == 0:
            return None
        i = int(hits[0])
        lo = max(0, i - self.window + 1)
        evidence = [int(turns[j]["turn_index"]) for j in range(lo, i + 1)
                    if _is_blocked_or_refused(turns[j])][-5:]
        sig = {s: round(float(v), 2) for s, v in zip(SIGNALS, norm[i])}
        top = sorted(sig, key=sig.get, reverse=True)[:3]
        return {"session_id": turns[0]["session_id"], "alert_turn": int(turns[i]["turn_index"]),
                "score": round(float(scores[i]), 3), "threshold": round(self.threshold, 3),
                "signals": sig, "evidence_turns": evidence, "atlas": ATLAS_IDS,
                "reason": "top signals: " + ", ".join(f"{s}={sig[s]}" for s in top)}

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"weights": self.weights, "window": self.window, "mode": self.mode, "learned": self.learned,
                                          "coef": None if self.coef is None else self.coef.tolist(),
                                          "intercept": self.intercept,
                                          "lo": self.lo.tolist(), "hi": self.hi.tolist(),
                                          "threshold": self.threshold}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Detector":
        d = json.loads(Path(path).read_text())
        return cls(weights=d["weights"], window=d["window"], mode=d.get("mode", "session"), learned=d.get("learned", False),
                   coef=None if d.get("coef") is None else np.array(d["coef"]), intercept=d.get("intercept", 0.0), lo=np.array(d["lo"]),
                   hi=np.array(d["hi"]), threshold=d["threshold"])
