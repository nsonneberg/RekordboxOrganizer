"""Tests for the calibrated fusion (fusion.py) and its use in genres.py."""

import random
import unittest
from unittest import mock

import numpy as np

import fusion
import genres
from fusion import Row
from test_genres import HermeticLabels, direction, meta, model, track


def discogs(segments, embedding=None):
    r = model(segments, embedding)
    r.model = "discogs-effnet"
    return r


class PoolingTests(unittest.TestCase):
    classes = ["House", "Techno", "Trance"]

    def fused(self, rows, labels=(), **kw):
        f = fusion.Fusion(self.classes, ["model", "library", "metadata", "tempo"],
                          fusion.label_prior(list(labels), self.classes), [])
        return f.predict(rows)

    def test_missing_source_is_neutral(self):
        only = Row({"model": {"House": 0.7, "Techno": 0.3}})
        also_empty = Row({"model": {"House": 0.7, "Techno": 0.3}, "metadata": {}},
                         {"metadata": 0.9})
        self.assertEqual(self.fused([only]), self.fused([also_empty]))

    def test_agreeing_sources_reinforce(self):
        a = self.fused([Row({"model": {"House": 0.7, "Techno": 0.3}})])[0]["House"]
        both = self.fused([Row({"model": {"House": 0.7, "Techno": 0.3},
                                "metadata": {"House": 0.8, "Techno": 0.2}})])[0]["House"]
        self.assertGreater(both, a)          # not an average of the two

    def test_conflict_lowers_confidence(self):
        a = self.fused([Row({"model": {"House": 0.8, "Techno": 0.2}})])[0]["House"]
        c = self.fused([Row({"model": {"House": 0.8, "Techno": 0.2},
                             "metadata": {"Techno": 1.0}})])[0]["House"]
        self.assertLess(c, a)

    def test_match_quality_scales_metadata(self):
        weak = self.fused([Row({"metadata": {"Techno": 1.0}}, {"metadata": 0.2})])[0]["Techno"]
        good = self.fused([Row({"metadata": {"Techno": 1.0}}, {"metadata": 0.9})])[0]["Techno"]
        self.assertLess(weak, good)

    def test_partial_source_is_neutral_for_families_it_doesnt_know(self):
        # A classifier trained on House / Techno has no say about Trance.
        base = self.fused([Row({"model": {"Trance": 0.9, "House": 0.1}})])[0]
        lib = self.fused([Row({"model": {"Trance": 0.9, "House": 0.1},
                               "library": {"House": 0.5, "Techno": 0.5}},
                              partial=("library",))])[0]
        self.assertAlmostEqual(base["Trance"], lib["Trance"], places=6)

    def test_no_single_source_can_veto(self):
        p = self.fused([Row({"model": {"House": 1.0}, "metadata": {"Techno": 1.0}})])[0]
        self.assertGreater(p["Techno"], 0.05)

    def test_tempo_is_a_fixed_source(self):
        p = self.fused([Row({"model": {"House": 0.5, "Techno": 0.5},
                             "tempo": {"House": 1.0, "Techno": 0.3679, "Trance": 1.0}})])[0]
        self.assertGreater(p["House"], p["Techno"])
        # ...but it can't make a choice on its own (only lowers families).
        alone = self.fused([Row({"tempo": {"House": 1.0, "Techno": 0.3679, "Trance": 1.0}})])[0]
        self.assertLess(max(alone.values()), 0.5)


def synthetic(n, accuracy, seed=0, families=("House", "Techno"), share=0.7):
    """Rows whose model points at the right family with probability `accuracy`."""
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        truth = families[i % len(families)]
        said = truth if rng.random() < accuracy else rng.choice([f for f in families if f != truth])
        rows.append(Row({"model": {said: share, **{f: (1 - share) / (len(families) - 1)
                                                   for f in families if f != said}}}))
        labels.append(truth)
    return rows, labels


class FittingTests(unittest.TestCase):
    def test_confidence_is_calibrated(self):
        # The model is right 75% of the time with the same 0.7 share each time;
        # after fitting, its confidence should be about 0.75, not 0.7 or 0.99.
        rows, labels = synthetic(400, 0.75)
        f = fusion.build(rows, labels)
        self.assertTrue(f.fitted)
        p = f.predict(rows[:1])[0]
        self.assertAlmostEqual(max(p.values()), 0.75, delta=0.06)

    def test_unreliable_source_gets_less_weight(self):
        rows, labels = synthetic(200, 0.9, seed=1)
        noise = random.Random(2)
        for r in rows:
            said = noise.choice(["House", "Techno"])
            r.shares["metadata"] = {said: 1.0}
            r.strength["metadata"] = 1.0
        f = fusion.build(rows, labels)
        self.assertGreater(f.weight("model"), 3 * f.weight("metadata"))

    def test_correlated_sources_are_not_double_counted(self):
        # The same evidence twice (like a classifier built on the model's own
        # embeddings) shouldn't make the fitted confidence much higher.
        rows, labels = synthetic(300, 0.8, seed=3, share=0.9)   # overconfident source
        single = fusion.build(rows, labels).predict(rows[:1])[0]
        twice = []
        for r in rows:
            twice.append(Row({"model": r.shares["model"], "library": dict(r.shares["model"])}))
        double = fusion.build(twice, labels).predict(twice[:1])[0]
        self.assertAlmostEqual(max(double.values()), max(single.values()), delta=0.05)
        # Prior weights alone would double count.
        naive = fusion.Fusion(sorted(set(labels)), ["model", "library"],
                              fusion.label_prior(labels, sorted(set(labels))), [])
        self.assertGreater(max(naive.predict(twice[:1])[0].values()),
                           max(double.values()) + 0.05)

    def test_too_few_examples_keep_prior_weights(self):
        rows, labels = synthetic(fusion.MIN_FIT_EXAMPLES - 1, 1.0)
        self.assertFalse(fusion.build(rows, labels).fitted)
        rows, labels = synthetic(20, 1.0, families=("House",))
        self.assertFalse(fusion.build(rows, labels).fitted)       # one family: nothing to learn

    def test_family_specific_adjustments_need_enough_examples(self):
        rows, labels = synthetic(30, 0.8, families=("House", "House", "Techno"))
        f = fusion.build(rows, labels)
        self.assertEqual(f.big, ["House", "Techno"])   # 20 House, 10 Techno -> both >= 10
        rows, labels = synthetic(18, 0.8, families=("House", "House", "Techno"))
        self.assertEqual(fusion.build(rows, labels).big, ["House"])   # Techno only 6

    def test_deterministic(self):
        rows, labels = synthetic(60, 0.8)
        a = fusion.build(rows, labels).predict(rows[:3])
        b = fusion.build(rows, labels).predict(rows[:3])
        self.assertEqual(a, b)


class PriorShiftTests(unittest.TestCase):
    def test_label_prior_fits_but_does_not_vote(self):
        # 30 House tags, 3 Techno; the evidence for the new song is even.
        rows = [fusion.Row({"model": {"House": 0.9}}) for _ in range(30)] +                [fusion.Row({"model": {"Techno": 0.9}}) for _ in range(3)]
        f = fusion.build(rows, ["House"] * 30 + ["Techno"] * 3)
        even = fusion.Row({"model": {"House": 0.5, "Techno": 0.5}})
        with_prior = f.predict([even], prior=True)[0]
        without = f.predict([even])[0]
        self.assertGreater(with_prior["House"], 0.8)          # the tag mix would decide
        self.assertLess(without["House"], 0.6)                # it doesn't any more

    def test_family_reliability_needs_its_own_checks(self):
        # House: 1 wrong of 46 (upper bound 9.9%). Techno: 9 of 9 right, but 9 checks
        # can't rule out a 28% error rate. Pop: 1 of 2.
        guessed = ["House"] * 46 + ["Pop"] * 2 + ["Techno"] * 9
        conf = [0.99] * 57
        right = [True] * 45 + [False] + [False, True] + [True] * 9
        ok = fusion.family_reliable(guessed, conf, right, 0.5)
        self.assertEqual(ok, {"House": True, "Pop": False, "Techno": False})
        self.assertEqual(fusion.family_reliable(guessed, conf, right, None), {})
        self.assertNotIn("Techno", fusion.family_reliable(guessed, [0.4] * 57, right, 0.5))
        # One more mistake for House and it no longer qualifies.
        right[0] = False
        self.assertFalse(fusion.family_reliable(guessed, conf, right, 0.5)["House"])

    def test_error_upper_bound(self):
        """One-sided 95% Clopper-Pearson bounds, checked against the closed form."""
        for n in (1, 8, 29, 100):          # no errors: 1 - 0.05 ** (1 / n)
            self.assertAlmostEqual(fusion.error_upper_bound(0, n), 1 - 0.05 ** (1 / n), 6)
        self.assertGreater(fusion.error_upper_bound(0, 28), fusion.AUTO_MAX_ERROR)
        self.assertLessEqual(fusion.error_upper_bound(0, 29), fusion.AUTO_MAX_ERROR)
        self.assertGreater(fusion.error_upper_bound(1, 45), fusion.AUTO_MAX_ERROR)
        self.assertLessEqual(fusion.error_upper_bound(1, 46), fusion.AUTO_MAX_ERROR)
        self.assertEqual(fusion.error_upper_bound(0, 0), 1.0)
        self.assertEqual(fusion.error_upper_bound(3, 3), 1.0)
        # Always at least the observed error rate.
        self.assertGreater(fusion.error_upper_bound(5, 20), 0.25)


class GateBoundaryTests(unittest.TestCase):
    """The per-genre automatic-tagging gate at its edges, and failing closed."""

    def test_bound_is_the_exact_clopper_pearson_value(self):
        from math import comb

        def cdf(e, n, p):
            return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(e + 1))
        for e, n in [(0, 1), (0, 2), (1, 2), (0, 8), (1, 10), (2, 30), (1, 46), (5, 200)]:
            ub = fusion.error_upper_bound(e, n)
            # By definition P(X <= e | p = bound) = 5%, and the bound is above e / n.
            self.assertAlmostEqual(cdf(e, n, ub), 0.05, places=9, msg=(e, n))
            self.assertGreater(ub, e / n)
        self.assertAlmostEqual(fusion.error_upper_bound(0, 1), 0.95, 9)
        self.assertAlmostEqual(fusion.error_upper_bound(0, 2), 1 - 0.05 ** 0.5, 9)
        # More checks with the same error rate only ever tighten the bound.
        self.assertGreater(fusion.error_upper_bound(1, 20), fusion.error_upper_bound(2, 40))

    def test_malformed_input_never_looks_reliable(self):
        for args in [(-1, 10), (5, 3), (3, 3), (0, 0), (0, -4), ("x", 10), (None, 10),
                     (0, 10, 1.0), (0, 10, 0.0), (0, 10, float("nan"))]:
            self.assertEqual(fusion.error_upper_bound(*args), 1.0, args)

    def test_exactly_at_the_boundary(self):
        def gate(n, wrong=0):
            return fusion.family_reliable(["House"] * n, [0.9] * n,
                                          [False] * wrong + [True] * (n - wrong), 0.5)["House"]
        self.assertFalse(gate(28))
        self.assertTrue(gate(29))
        self.assertFalse(gate(45, 1))
        self.assertTrue(gate(46, 1))
        self.assertFalse(gate(200, 14))     # 7% observed, but the bound is above 10%

    def test_related_recordings_count_once(self):
        # 29 clean checks, but they're remixes of only 15 songs: not 29 pieces of evidence.
        groups = [f"song{i // 2}" for i in range(29)]
        ok = fusion.family_reliable(["House"] * 29, [0.9] * 29, [True] * 29, 0.5, groups)
        self.assertFalse(ok["House"])
        self.assertEqual(fusion.family_checks(["House"] * 29, [0.9] * 29, [True] * 29, 0.5,
                                              groups), {"House": (15, 0)})
        # One wrong version makes the whole group count as wrong.
        right = [True] * 29
        right[1] = False
        self.assertEqual(fusion.family_checks(["House"] * 29, [0.9] * 29, right, 0.5,
                                              groups)["House"], (15, 1))

    def test_missing_or_broken_checks_are_ignored(self):
        counts = fusion.family_checks(["House", "House", "", "House"],
                                      [float("nan"), None, 0.9, 0.9], [True] * 4, 0.5)
        self.assertEqual(counts, {"House": (1, 0)})
        self.assertEqual(fusion.family_reliable([], [], [], 0.5), {})
        self.assertEqual(fusion.family_reliable(["House"] * 40, [0.9] * 40, [True] * 40, None), {})

    def test_only_calibrated_predictions_that_passed_may_be_used_unapproved(self):
        P = genres.GenrePrediction
        self.assertTrue(genres.auto_allowed(P("confident", "House", method="calibrated")))
        for p in [None, P("confident", "House", method="legacy"),
                  P("suggested", "House", method="calibrated"),
                  P("uncertain", "House", method="calibrated"),
                  P("confident", "", method="calibrated")]:
            self.assertFalse(genres.auto_allowed(p), p)
        preds = {"a": P("confident", "House", method="calibrated"),
                 "b": P("confident", "Techno", method="legacy"),
                 "c": P("suggested", "Trance", method="calibrated")}
        self.assertEqual(genres.confident_labels(preds), {"a": "House"})


class ConfidenceDisplayTests(unittest.TestCase):
    def test_never_shown_as_certain(self):
        import organizer
        self.assertEqual(organizer.confidence_text(0.9999), ">99%")
        self.assertEqual(organizer.confidence_text(1.0), ">99%")
        self.assertEqual(organizer.confidence_text(0.001), "<1%")
        self.assertEqual(organizer.confidence_text(0.934), "93%")


class ThresholdTests(unittest.TestCase):
    def test_lowest_threshold_meeting_the_error_target(self):
        conf = [0.95] * 20 + [0.6] * 10
        right = [True] * 20 + [False] * 5 + [True] * 5
        self.assertEqual(fusion.choose_threshold(conf, right), 0.65)

    def test_none_when_never_accurate_enough(self):
        conf = [0.9] * 20
        right = [True, False] * 10
        self.assertIsNone(fusion.choose_threshold(conf, right))

    def test_needs_enough_checks(self):
        self.assertIsNone(fusion.choose_threshold([0.99] * 5, [True] * 5))
        self.assertEqual(fusion.choose_threshold([0.99] * 8, [True] * 8), 0.5)

    def test_error_estimate_is_conservative(self):
        self.assertAlmostEqual(fusion.error_estimate(0, 8), 0.1)
        self.assertGreater(fusion.error_estimate(0, 3), fusion.AUTO_MAX_ERROR)

    def test_folds_group_and_balance(self):
        keys = [f"g{i % 7}" for i in range(30)]
        f = fusion.folds(keys, 3)
        self.assertEqual(set(f.values()), {0, 1, 2})
        self.assertEqual(f, fusion.folds(list(reversed(keys)), 3))


class CalibratedGenreTests(HermeticLabels):
    """genres.predict_genres with the default (calibrated) method."""

    def test_clear_model_evidence_without_labels(self):
        tracks = [track("1", "", 174)]
        r = {"1": discogs([{"Electronic---Drum n Bass": 0.9}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r)["1"]
        # Nothing has been measured, so a clear guess is only a suggestion.
        self.assertEqual((p.method, p.family, p.status), ("calibrated", "Drum & Bass", "suggested"))
        self.assertEqual(p.threshold, fusion.UNFITTED_THRESHOLD)
        self.assertIn("too few tagged songs", p.reason)
        self.assertEqual(genres.guess_genres(tracks, {}, None, r), {})

    def test_moderate_evidence_is_predicted_but_not_assigned(self):
        tracks = [track("1", "", 126)]
        r = {"1": discogs([{"Electronic---Tech House": 0.5, "Electronic---Techno": 0.3}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r)["1"]
        self.assertEqual(p.family, "House")              # the prediction is kept...
        self.assertFalse(p.confident)                    # ...but goes to review
        self.assertIn("needed to tag automatically", p.reason)
        self.assertEqual(genres.guess_genres(tracks, {}, None, r), {})

    def test_bpm_alone_is_never_enough(self):
        p = genres.predict_genres([track("1", "", 174)], {}, None)["1"]
        self.assertFalse(p.confident)
        self.assertEqual(p.sources, ("tempo",))

    def test_missing_metadata_is_not_negative(self):
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---Techno": 0.8}] * 3)}
        a = genres.predict_genres(tracks, {}, None, r)["1"]
        b = genres.predict_genres(tracks, {}, None, r, metadata={"2": meta([("house", 1)])})["1"]
        self.assertEqual((a.family, a.score, a.status), (b.family, b.score, b.status))

    def test_conflicting_sources_go_to_review(self):
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---House": 0.9}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r, metadata={"1": meta([("techno", 4)])})["1"]
        self.assertFalse(p.confident)
        self.assertTrue(p.disagreement)
        self.assertIn("disagree", p.reason)
        self.assertEqual({f for f, _ in p.candidates[:2]}, {"House", "Techno"})

    def test_agreeing_metadata_raises_confidence(self):
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---Techno": 0.6, "Electronic---House": 0.3}] * 3)}
        alone = genres.predict_genres(tracks, {}, None, r)["1"]
        both = genres.predict_genres(tracks, {}, None, r, metadata={"1": meta([("techno", 3)])})["1"]
        self.assertGreater(both.score, alone.score)
        self.assertEqual(both.family, "Techno")

    def test_ambiguous_metadata_left_out_is_neutral(self):
        # metadata.best_match returns None for ambiguous matches; nothing reaches fusion.
        tracks = [track("1", "", 128)]
        r = {"1": discogs([{"Electronic---Techno": 0.8}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r, metadata={})["1"]
        self.assertNotIn("metadata", p.sources)

    def library(self, accuracy=1.0, n=12, seed=0, informative=True):
        """n House + n Techno tagged songs whose model output is right with `accuracy`.

        informative: the embeddings separate the families (so the classifier
        trained on tags can learn them); otherwise they're random.
        """
        rng = random.Random(seed)
        noise = np.random.default_rng(seed)
        tracks, r = [], {}
        for fam, style, axis in (("House", "Electronic---House", 0),
                                 ("Techno", "Electronic---Techno", 1)):
            other = "Electronic---Techno" if fam == "House" else "Electronic---House"
            for i in range(n):
                tid = f"{fam[:2]}{i}"
                tracks.append(track(tid, fam, 126, name=f"{fam} number {i} unique"))
                said = style if rng.random() < accuracy else other
                emb = direction(axis, 0.05 * i) if informative else noise.normal(size=16)
                r[tid] = discogs([{said: 0.6, "Electronic---Trance": 0.05}] * 2, emb)
        return tracks, r

    def test_fitted_on_tagged_songs(self):
        tracks, r = self.library(n=30)
        tracks.append(track("x", "", 126))
        r["x"] = discogs([{"Electronic---Techno": 0.6}] * 2, direction(1, 0.02))
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual((p.family, p.status), ("Techno", "confident"))
        self.assertIsNotNone(p.threshold)
        self.assertNotEqual(p.threshold, fusion.UNFITTED_THRESHOLD)

    def test_inaccurate_sources_mean_nothing_is_automatic(self):
        tracks, r = self.library(accuracy=0.55, seed=4, informative=False)
        tracks.append(track("x", "", 126))
        r["x"] = discogs([{"Electronic---Techno": 0.6}] * 2)
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertFalse(p.confident)
        self.assertIsNone(p.threshold)
        self.assertIn("aren't", p.reason.replace("don't", "aren't"))

    def test_family_with_too_few_checks_is_only_suggested(self):
        tracks, r = self.library(n=30)
        for i in range(3):                      # only 3 tagged Dubstep songs
            tid = f"Du{i}"
            tracks.append(track(tid, "Dubstep", 140, name=f"Dubstep number {i} unique"))
            r[tid] = discogs([{"Electronic---Dubstep": 0.9}] * 2, direction(3, 0.05 * i))
        tracks.append(track("x", "", 140))
        r["x"] = discogs([{"Electronic---Dubstep": 0.9}] * 2, direction(3, 0.02))
        tracks.append(track("y", "", 126))
        r["y"] = discogs([{"Electronic---Techno": 0.6}] * 2, direction(1, 0.02))
        out = genres.predict_genres(tracks, {}, None, r)
        self.assertEqual(out["x"].family, "Dubstep")
        self.assertEqual(out["x"].status, "suggested")
        self.assertIn("haven't yet proved reliable", out["x"].reason)
        self.assertIn("could be as high as", out["x"].reason)
        self.assertTrue(out["y"].confident)                 # Techno has 30 checks
        self.assertIn("were right", out["y"].reason)

    def test_twelve_clean_checks_are_not_enough(self):
        tracks, r = self.library(n=12)
        tracks.append(track("y", "", 126))
        r["y"] = discogs([{"Electronic---Techno": 0.6}] * 2, direction(1, 0.02))
        p = genres.predict_genres(tracks, {}, None, r)["y"]
        self.assertEqual((p.family, p.status), ("Techno", "suggested"))

    def test_unseen_family_is_not_forced_into_known_ones(self):
        # Only House and Techno are tagged; the track is clearly Trance.
        tracks, r = self.library()
        tracks.append(track("x", "", 138))
        r["x"] = discogs([{"Electronic---Trance": 0.9}] * 2, direction(5))
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual(p.family, "Trance")

    def test_genre_never_tagged_is_never_automatic(self):
        # House and Techno are well proven; Trance has no held-out checks at all.
        tracks, r = self.library(n=30)
        tracks.append(track("x", "", 138))
        r["x"] = discogs([{"Electronic---Trance": 0.95}] * 2, direction(5))
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual(p.family, "Trance")
        self.assertNotEqual(p.status, "confident")
        self.assertFalse(genres.auto_allowed(p))
        self.assertTrue("0 held-out checks" in p.reason or "disagree" in p.reason, p.reason)

    def test_subgenre_is_suggested_but_assigned_only_when_measured(self):
        tracks = [track("1", "", 126)]
        r = {"1": discogs([{"Electronic---Tech House": 0.95}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r)["1"]
        self.assertEqual((p.family, p.suggested_subgenre, p.subgenre), ("House", "Tech House", ""))
        self.assertEqual(p.label, "House")

    def test_subgenre_assigned_after_proving_itself(self):
        tracks, r = [], {}
        for i in range(30):
            for fam, sub, style, axis in (("House", "Tech House", "Electronic---Tech House", 0),
                                          ("Techno", "", "Electronic---Techno", 1)):
                tid = f"{fam[:2]}{i}"
                tracks.append(track(tid, sub or fam, 126, name=f"{tid} unique title"))
                r[tid] = discogs([{style: 0.9}] * 2, direction(axis, 0.05 * i))
        tracks.append(track("x", "", 126))
        r["x"] = discogs([{"Electronic---Tech House": 0.9}] * 2, direction(0, 0.01))
        p = genres.predict_genres(tracks, {}, None, r)["x"]
        self.assertEqual((p.status, p.subgenre, p.label), ("confident", "Tech House", "Tech House"))

    def test_discogs_metadata_is_never_trained_on(self):
        import metadata as md
        mb = md.MetadataResult("MusicBrainz", [("techno", 1)], 0.9, "recording", "a")
        dc = md.MetadataResult("Discogs", [("Techno", 1.0)], 0.7, "release", "b")
        self.assertIsNone(genres._trainable_metadata(dc))
        both = md.merge([mb, dc])
        self.assertEqual(genres._trainable_metadata(both).source, "MusicBrainz")
        self.assertIs(genres._trainable_metadata(mb), mb)

    def test_copies_of_tagged_songs_are_still_direct(self):
        tracks = [track("1", "Techno", 130, name="Same Song", length=300),
                  track("2", "", 130, name="Same Song", length=300)]
        p = genres.predict_genres(tracks, {}, None)["2"]
        self.assertEqual((p.family, p.sources, p.status), ("Techno", ("copy",), "suggested"))
        self.assertFalse(genres.auto_allowed(p))

    def test_legacy_method_still_available(self):
        tracks = [track("1", "", 174)]
        r = {"1": discogs([{"Electronic---Drum n Bass": 0.9}] * 3)}
        p = genres.predict_genres(tracks, {}, None, r, method="legacy")["1"]
        self.assertEqual((p.method, p.family), ("legacy", "Drum & Bass"))


class CalibratedEvaluationTests(HermeticLabels):
    def test_evaluation_is_nested(self):
        """The held-out song never influences the weights or threshold used for it."""
        tracks, r = CalibratedGenreTests.library(None, n=10)
        seen = []
        real = genres._Calibrated.__init__

        def spy(self, ctx):
            real(self, ctx)
            seen.append({e.group for e, _ in self.items})
        with mock.patch.object(genres._Calibrated, "__init__", spy):
            ev = genres.evaluate(tracks, {}, r)
        self.assertEqual(ev.recordings, 20)
        groups = genres._groups(tracks)
        for (probe, _, _), training in zip(ev.rows, seen):
            self.assertNotIn(groups[probe], training)


if __name__ == "__main__":
    unittest.main()
