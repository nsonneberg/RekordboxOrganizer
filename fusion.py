"""Calibrated fusion of genre evidence.

Each source (the audio model, the classifier trained on your tags, release
metadata, tempo...) gives a score per genre family. They are combined by
log-linear pooling:

    z(f) = w_prior * log prior(f) + bias(f) + sum over sources s of
           strength_s * (w_s + delta_s(f)) * term_s(f)

    confidence(f) = softmax(z)(f)

- term_s(f) is the log of the source's share for f, smoothed with a fixed
  total of SMOOTHING spread evenly over all families (so no single source
  can veto a genre outright, however many families there are). Tempo and
  the vague tag are "fixed" sources: their terms are 0 (fits) or -1 (doesn't). A source with no opinion of a
  track (no metadata match, no audio) is simply left out: missing evidence is
  never evidence against a genre.
- strength_s scales a source per track; for metadata it is the match quality.
- The weights w_s are fitted by maximum likelihood on *out-of-fold*
  predictions (a learned source's opinion of a labelled song always comes
  from a model that never saw that song), with an L2 pull towards
  PRIOR_WEIGHTS. Because the weights are fitted jointly, two sources that
  say the same thing for the same reason (the tag-trained classifier is
  built on the audio model's own embeddings) aren't counted twice the way
  naive Bayes would count them; agreeing independent sources reinforce each
  other instead of being averaged down.
- delta_s(f) and bias(f) (family-specific reliability) are only fitted for
  families with at least CLASS_MIN labelled recordings, also pulled to 0.

The label prior (how often you've tagged each family) is used while fitting,
so the source weights aren't distorted by it, but not when predicting
(PREDICT_WITH_PRIOR): the songs being guessed are the ones you *didn't* tag,
and their genre mix is unknown, so "most of my tagged songs are House" isn't
evidence that an untagged song is House. Predictions are therefore the
evidence alone under an even prior. The confidence is a probability under
the model's assumptions; with few tagged songs and sources that hear the same
audio it can be too high, so automatic tagging relies on measured held-out
accuracy, not on the confidence alone.

Whether a genre is *assigned automatically* is a separate decision
(choose_threshold): the lowest confidence threshold at which, on held-out
checks inside the training data, the estimated error rate of automatic
assignments stays at or below AUTO_MAX_ERROR. If no threshold achieves that,
nothing is assigned automatically. The threshold is an operating point
chosen from a point estimate; it is not a guarantee of any error rate.

On top of that (family_reliable), a family is only assigned automatically
when its own held-out guesses at the threshold make a low error rate likely:
the one-sided AUTO_CONFIDENCE (95%) Clopper-Pearson upper bound on their
error rate must be at most AUTO_MAX_ERROR. With no mistakes that takes 29
checks of that family; with one, 46. A family short of that is only
*suggested* (shown for the user to approve), never assigned. Even then the
bound is only as good as the checks: they come from your own tags, which
may be inconsistent, and from songs like the ones you've already tagged.

Pure numpy and deterministic. All constants are fixed in advance (they are
priors and policy, not tuned on any test set).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SMOOTHING = 0.05             # share mass every source spreads evenly over all families
FIXED = ("tempo", "tag")     # sources whose shares are exp(term) per family, not a distribution
PRIOR_WEIGHTS = {"prior": 1.0, "model": 1.0, "library": 1.0, "sound": 0.5,
                 "metadata": 1.0, "tempo": 0.5, "tag": 0.5}
L2 = 1.0                     # strength of the pull towards PRIOR_WEIGHTS (and deltas to 0)
CLASS_MIN = 10               # labelled recordings before a family gets its own adjustments
PRIOR_SMOOTHING = 1.0        # Dirichlet pseudo-count per family for the label prior
ITERATIONS, RATE = 400, 0.05

AUTO_MAX_ERROR = 0.10        # policy: automatic assignments should be wrong at most ~10%
AUTO_CONFIDENCE = 0.95       # how sure the per-family upper bound on that error must be
THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))   # 0.50 ... 0.95
MIN_AUTO_CHECKS = 8          # held-out automatic assignments needed to trust a threshold
UNFITTED_THRESHOLD = 0.9     # with too few tagged songs to measure anything
MIN_FIT_EXAMPLES = 10        # labelled recordings (in 2+ families) before fitting
PREDICT_WITH_PRIOR = False   # the label prior fits the weights but doesn't vote (see above)


@dataclass
class Row:
    """One track's evidence: source -> {family: share}, and source -> strength.

    For sources listed in `partial`, families missing from their shares are
    neutral (the source only knows some families, like a classifier trained on
    the families you've tagged) instead of floored.
    """
    shares: dict = field(default_factory=dict)
    strength: dict = field(default_factory=dict)
    partial: tuple = ()

    def term(self, source: str, family: str, n_classes: int) -> float:
        s = self.shares[source]
        if source in FIXED:
            return float(np.log(s.get(family, 1.0)))
        total = sum(s.values()) or 1.0
        even = SMOOTHING / max(n_classes, 1)
        if family in s:
            return float(np.log((1 - SMOOTHING) * s[family] / total + even))
        if source in self.partial:          # neutral: as if it split its share evenly
            return float(np.log((1 - SMOOTHING) / max(len(s), 1) + even))
        return float(np.log(even))


def label_prior(labels: list[str], classes: list[str]) -> np.ndarray:
    counts = Counter(labels)
    raw = np.array([counts.get(c, 0) + PRIOR_SMOOTHING for c in classes], dtype=float)
    return np.log(raw / raw.sum())


class Fusion:
    """Fitted (or, with too little data, prior-only) log-linear pooling."""

    def __init__(self, classes: list[str], sources: list[str], log_prior: np.ndarray,
                 big: list[str]):
        self.classes = list(classes)
        self.sources = list(sources)
        self.log_prior = log_prior
        self.big = [c for c in big if c in self.classes]      # families with own adjustments
        self.w = np.array([PRIOR_WEIGHTS.get(s, 0.5) for s in self.sources])
        self.w_prior = PRIOR_WEIGHTS["prior"]
        self.delta = np.zeros((len(self.sources), len(self.classes)))
        self.bias = np.zeros(len(self.classes))
        self.fitted = False

    # -- features -----------------------------------------------------------
    def _tensor(self, rows: list[Row]) -> np.ndarray:
        """[n, sources, classes] of strength x term (0 where a source is missing)."""
        x = np.zeros((len(rows), len(self.sources), len(self.classes)))
        for i, r in enumerate(rows):
            for j, s in enumerate(self.sources):
                if s not in r.shares or not r.shares[s]:
                    continue
                a = float(r.strength.get(s, 1.0))
                if a <= 0:
                    continue
                x[i, j] = [a * r.term(s, c, len(self.classes)) for c in self.classes]
        return x

    def _logits(self, x: np.ndarray) -> np.ndarray:
        weights = self.w[:, None] + self.delta              # [sources, classes]
        return (self.w_prior * self.log_prior + self.bias
                + np.einsum("isc,sc->ic", x, weights))

    # -- fitting ------------------------------------------------------------
    def fit(self, rows: list[Row], labels: list[str]) -> "Fusion":
        y = np.array([self.classes.index(l) for l in labels])
        x = self._tensor(rows)
        n, k = len(y), len(self.classes)
        onehot = np.eye(k)[y]
        w0 = np.array([PRIOR_WEIGHTS.get(s, 0.5) for s in self.sources])
        big = np.array([c in self.big for c in self.classes], dtype=float)
        # Adam on: w (sources), w_prior, delta (sources x big classes), bias (big classes)
        params = [self.w.copy(), np.array([self.w_prior]), self.delta.copy(), self.bias.copy()]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for step in range(1, ITERATIONS + 1):
            self.w, self.w_prior, self.delta, self.bias = (params[0], float(params[1][0]),
                                                           params[2], params[3])
            z = self._logits(x)
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            g = (p - onehot) / n                                   # dNLL/dz, [n, k]
            gs = np.einsum("ic,isc->sc", g, x)                     # d/d(weights) [s, k]
            grads = [gs.sum(axis=1) + 2 * L2 / n * (params[0] - w0),
                     np.array([float((g * self.log_prior).sum())
                               + 2 * L2 / n * (params[1][0] - PRIOR_WEIGHTS["prior"])]),
                     (gs + 2 * L2 / n * params[2]) * big[None, :],
                     (g.sum(axis=0) + 2 * L2 / n * params[3]) * big]
            for i in range(4):
                m[i] = 0.9 * m[i] + 0.1 * grads[i]
                v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
                mh = m[i] / (1 - 0.9 ** step)
                vh = v[i] / (1 - 0.999 ** step)
                params[i] = params[i] - RATE * mh / (np.sqrt(vh) + 1e-8)
            params[0] = np.maximum(params[0], 0.0)     # a source can't count backwards
            params[1] = np.maximum(params[1], 0.0)
        self.w, self.w_prior, self.delta, self.bias = (params[0], float(params[1][0]),
                                                       params[2], params[3])
        self.fitted = True
        return self

    # -- prediction -----------------------------------------------------------
    def predict(self, rows: list[Row], prior: Optional[bool] = None) -> list[dict[str, float]]:
        """Confidence per family. prior: include the label prior (default PREDICT_WITH_PRIOR)."""
        if not rows:
            return []
        z = self._logits(self._tensor(rows))
        if not (PREDICT_WITH_PRIOR if prior is None else prior):
            z = z - (self.w_prior * self.log_prior + self.bias)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return [{c: float(v) for c, v in zip(self.classes, row)} for row in p]

    def weight(self, source: str, family: Optional[str] = None) -> float:
        """The fitted weight of a source (for one family, if given)."""
        if source not in self.sources:
            return 0.0
        j = self.sources.index(source)
        extra = self.delta[j, self.classes.index(family)] if family in self.classes else 0.0
        return float(self.w[j] + extra)


def build(rows: list[Row], labels: list[str], extra_classes=()) -> Fusion:
    """A Fusion over every family seen in the labels or the evidence, fitted if possible."""
    classes = sorted({l for l in labels} | set(extra_classes)
                     | {f for r in rows for s in r.shares.values() for f in s})
    # Every source, even one no training row has: it then keeps its prior weight.
    sources = sorted(set(PRIOR_WEIGHTS) - {"prior"} | {s for r in rows for s in r.shares})
    counts = Counter(labels)
    big = [c for c, n in counts.items() if n >= CLASS_MIN]
    fusion = Fusion(classes, sources, label_prior(labels, classes), big)
    if len(labels) >= MIN_FIT_EXAMPLES and len(counts) >= 2:
        fusion.fit(rows, labels)
    return fusion


def error_estimate(errors: int, n: int) -> float:
    """Posterior mean error rate with a uniform prior: (e + 1) / (n + 2)."""
    return (errors + 1) / (n + 2)


def error_upper_bound(errors: int, n: int, level: float = AUTO_CONFIDENCE) -> float:
    """One-sided Clopper-Pearson upper confidence bound on an error rate (exact binomial).

    The largest error rate p for which seeing `errors` or fewer mistakes in n
    tries still has probability at least 1 - level. 1.0 with no tries, and
    for anything malformed (so a broken input can never make a genre look reliable).
    """
    try:
        errors, n, level = int(errors), int(n), float(level)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if n <= 0 or errors < 0 or errors >= n or not 0.0 < level < 1.0:
        return 1.0
    from math import comb
    alpha = 1.0 - level

    def cdf(p: float) -> float:
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(errors + 1))
    lo, hi = errors / n, 1.0
    for _ in range(60):                 # cdf falls as p rises: bisect cdf(p) = alpha
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def choose_threshold(confidences: list[float], correct: list[bool]) -> Optional[float]:
    """The lowest threshold whose held-out automatic assignments meet AUTO_MAX_ERROR.

    confidences / correct: out-of-fold confidence of the predicted family and
    whether it was right. None if no threshold qualifies (assign nothing).
    """
    conf = np.array(confidences, dtype=float)
    ok = np.array(correct, dtype=bool)
    for t in THRESHOLDS:
        sel = conf >= t
        n = int(sel.sum())
        if n >= MIN_AUTO_CHECKS and error_estimate(int((~ok[sel]).sum()), n) <= AUTO_MAX_ERROR:
            return t
    return None


def family_checks(predicted: list[str], confidences: list[float], correct: list[bool],
                  threshold: Optional[float],
                  groups: Optional[list[str]] = None) -> dict[str, tuple[int, int]]:
    """{family: (held-out checks at the threshold, wrong ones)}.

    groups: each check's related-recordings group (versions, remixes, songs
    sharing a name part). Related recordings count as ONE check per family,
    wrong if any of them was wrong, so a song with five remixes can't count as
    five independent pieces of evidence. Checks with a missing or non-finite
    confidence are ignored.
    """
    import math
    if threshold is None:
        return {}
    if groups is None:
        groups = [str(i) for i in range(len(predicted))]
    worst: dict[tuple, bool] = {}
    for fam, c, ok, g in zip(predicted, confidences, correct, groups):
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(c) or c < threshold or not fam:
            continue
        worst[(fam, g)] = worst.get((fam, g), True) and bool(ok)
    out: dict[str, list] = {}
    for (fam, _), ok in worst.items():
        n, w = out.get(fam, (0, 0))
        out[fam] = (n + 1, w + (not ok))
    return out


def family_reliable(predicted: list[str], confidences: list[float], correct: list[bool],
                    threshold: Optional[float],
                    groups: Optional[list[str]] = None) -> dict[str, bool]:
    """{family: may be assigned automatically} from held-out checks at the threshold.

    A family qualifies only if the held-out songs guessed as it with
    confidence >= threshold (related recordings counted once, see
    family_checks) were wrong rarely enough that the upper bound on their
    error rate (error_upper_bound) is at most AUTO_MAX_ERROR. Families never
    checked aren't listed (no).
    """
    return {f: k >= MIN_AUTO_CHECKS and error_upper_bound(w, k) <= AUTO_MAX_ERROR
            for f, (k, w) in family_checks(predicted, confidences, correct, threshold,
                                           groups).items()}


def folds(keys: list[str], k: int) -> dict[str, int]:
    """Deterministic assignment of group keys to k folds (balanced, order-independent)."""
    import hashlib
    ordered = sorted(set(keys), key=lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    return {key: i % k for i, key in enumerate(ordered)}
