"""Measure genre guessing on your own tagged songs (reads the library only).

Leave-one-recording-out: each tagged recording (all its copies together) is
hidden in turn and predicted from the rest. Only tags you set yourself count
as truth (see genres.training_labels).

    python evaluate_genres.py                 # straight from rekordbox
    python evaluate_genres.py library.xml     # from an exported XML
    python evaluate_genres.py --no-model      # without the audio model
"""

import sys

import genres
import organizer


def main(argv: list[str]) -> None:
    use_model = "--no-model" not in argv
    paths = [a for a in argv if not a.startswith("--")]
    if paths:
        _, tracks = organizer.load_library(paths[0])
    else:
        import rbdb
        db = rbdb.open_database()
        try:
            tracks, _ = rbdb.load_tracks(db)
        finally:
            db.close()
    log = print
    features = organizer.analyse_tracks(tracks, log)
    labels = genres.load_labels()
    results = organizer.model_genres(tracks, log) if use_model else {}

    for name, res in ((("rekordbox analysis only", {}),) +
                      ((("with the audio model", results),) if results else ())):
        ev = genres.evaluate(tracks, features, res, labels)
        print(f"\n{name}: {ev.summary()}")
        for (truth, guess), n in sorted(ev.confusion.items()):
            mark = "" if truth == guess else "   <- wrong"
            print(f"    {truth:>14} -> {guess:<14} {n}{mark}")


if __name__ == "__main__":
    main(sys.argv[1:])
