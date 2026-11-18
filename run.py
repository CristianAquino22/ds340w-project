#!/usr/bin/env python
"""Pipeline runner for the un-xPass tournament reproduction.

Each stage is a separate subcommand so stages can be re-run independently --
feature computation and SoccerMap training are slow, and you do not want to
redo them to change something downstream.

    python run.py download --competition euro2020 [--limit 5]
    python run.py split    --competition euro2020
    python run.py features --split train
    python run.py train    --component pass_success
    python run.py pool     --name euros_train --sources euro2020_train euro2024_train
    python run.py datasets
    python run.py rate     --dataset euro2020_validation
    python run.py report   --dataset euro2020_validation
"""
from __future__ import annotations

import argparse
import sys

from unxpass_project import compat  # noqa: F401  -- patches must load first
from unxpass_project import config, data, featurize, models, rating, splits


def cmd_download(args):
    ids = data.download(args.competition, limit=args.limit)
    print(f"[download] {args.competition}: {len(ids)} matches in database")


def cmd_split(args):
    ids = data.game_ids(args.competition)
    if not ids:
        sys.exit(f"No matches for {args.competition}. Run 'download' first.")
    sp = splits.split_by_match(ids)
    splits.write_split(sp, args.competition)
    print(
        f"[split] {args.competition}: {len(sp['train'])} train / "
        f"{len(sp['test'])} test matches (split by match, not by pass)"
    )


def cmd_features(args):
    if args.competition == config.UNSEEN_COMPETITION:
        ids = data.game_ids(args.competition)
        name = f"{args.competition}_unseen"
    else:
        sp = splits.read_split(args.competition)
        ids = sp[args.split]
        name = f"{args.competition}_{args.split}"
    if args.limit:
        ids = ids[: args.limit]
    featurize.build(name, ids)


def cmd_pool(args):
    featurize.pool(args.name, args.sources)


def cmd_datasets(args):
    """Summarise the three-way split, for the dataset-split deliverable."""
    rows = []
    for comp in config.TRAIN_COMPETITIONS:
        try:
            sp = splits.read_split(comp)
        except FileNotFoundError:
            continue
        for part in ("train", "test"):
            # Older runs named the held-out set "validation"; accept either.
            candidates = [f"{comp}_{part}"]
            if part == "test":
                candidates.append(f"{comp}_validation")
            n = None
            for cand in candidates:
                try:
                    n = featurize.count_passes(cand)
                    break
                except FileNotFoundError:
                    continue
            rows.append((part.upper(), comp, len(sp.get(part, [])), n))
    unseen = config.UNSEEN_COMPETITION
    try:
        n_games = len(data.game_ids(unseen))
    except Exception:
        n_games = 0
    try:
        n = featurize.count_passes(f"{unseen}_unseen")
    except FileNotFoundError:
        n = None
    rows.append(("UNSEEN", unseen, n_games, n))

    print("\n=== three-way split ===")
    print(f"{'set':<8} {'competition':<14} {'matches':>8} {'passes':>9}")
    for part, comp, g, n in rows:
        print(f"{part:<8} {comp:<14} {g:>8} {('-' if n is None else n):>9}")

    totals = {}
    for part, _, g, n in rows:
        t = totals.setdefault(part, [0, 0, False])
        t[0] += g
        t[1] += n or 0
        t[2] = t[2] or n is None
    print()
    grand_g = sum(t[0] for t in totals.values())
    grand_n = sum(t[1] for t in totals.values())
    for part in ("TRAIN", "TEST", "UNSEEN"):
        if part not in totals:
            continue
        g, n, partial = totals[part]
        share_g = g / grand_g * 100 if grand_g else 0
        share_n = n / grand_n * 100 if grand_n else 0
        note = "  (features not yet computed)" if partial else ""
        print(f"{part:<8} {g:>3} matches ({share_g:4.1f}%)   {n:>6} passes ({share_n:4.1f}%){note}")
    print("\nThe unseen set is a separate competition, not a slice of the")
    print("training competitions. No model has been fitted or evaluated on it.")


def cmd_train(args):
    models.train(
        args.component,
        args.dataset,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
    )


def cmd_rate(args):
    rating.rate(args.dataset)


def cmd_benchmark(args):
    from unxpass_project import benchmark

    benchmark.report(args.dataset, options_dataset=args.options_dataset)


def cmd_gamestate(args):
    from unxpass_project import analysis

    analysis.report(args.dataset)


def cmd_ablation(args):
    from unxpass_project import ablation

    ablation.report(args.train_dataset, args.test_dataset)


def cmd_report(args):
    import pandas as pd

    path = config.RESULTS / f"ratings_{args.dataset}.csv"
    df = pd.read_csv(path).set_index(["game_id", "action_id"])
    print(f"\n=== CDR summary: {args.dataset} ===")
    print(f"passes rated : {len(df)}")
    print(f"mean CDR     : {df['CDR'].mean():+.5f}")
    print(f"std CDR      : {df['CDR'].std():.5f}")
    print(f"share CDR > 0: {(df['CDR'] > 0).mean():.1%}")
    print("\nquantiles:")
    print(df["CDR"].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_string())
    top = rating.player_rankings(df, min_minutes=args.min_minutes)
    print(f"\n=== top players by CDR per 90 (min {args.min_minutes:.0f} minutes) ===")
    print(top.head(15).to_string())
    out = config.RESULTS / f"players_{args.dataset}.csv"
    top.to_csv(out)
    print(f"\nsaved -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch a competition from StatsBomb open data")
    d.add_argument("--competition", default=config.TRAIN_COMPETITION,
                   choices=list(config.COMPETITIONS))
    d.add_argument("--limit", type=int, default=None, help="only the first N matches")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("split", help="split a competition into train/test by match")
    s.add_argument("--competition", default=config.TRAIN_COMPETITION,
                   choices=list(config.COMPETITIONS))
    s.set_defaults(func=cmd_split)

    f = sub.add_parser("features", help="compute features and labels")
    f.add_argument("--competition", default=config.TRAIN_COMPETITION,
                   choices=list(config.COMPETITIONS))
    f.add_argument("--split", default="train", choices=["train", "test"])
    f.add_argument("--limit", type=int, default=None)
    f.set_defaults(func=cmd_features)

    pl = sub.add_parser("pool", help="combine cached datasets without recomputing")
    pl.add_argument("--name", required=True, help="name of the pooled dataset")
    pl.add_argument("--sources", nargs="+", required=True)
    pl.set_defaults(func=cmd_pool)

    ds = sub.add_parser("datasets", help="summarise the three-way split")
    ds.set_defaults(func=cmd_datasets)

    t = sub.add_parser("train", help="train one model component")
    t.add_argument("--component", required=True, choices=list(models.BUILDERS))
    t.add_argument("--dataset", default="euro2020_train")
    t.add_argument("--max-epochs", type=int, default=20)
    t.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "auto"])
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("rate", help="compute CDR for a dataset")
    r.add_argument("--dataset", default="euro2020_validation")
    r.set_defaults(func=cmd_rate)

    b = sub.add_parser("benchmark", help="compare accuracy with the parent paper")
    b.add_argument("--dataset", default="euro2020_validation")
    b.add_argument("--options-dataset", default=None,
                   help="dataset built with pass_options features, for the baseline")
    b.set_defaults(func=cmd_benchmark)

    g = sub.add_parser("gamestate", help="does game state affect CDR? (extension)")
    g.add_argument("--dataset", default="euro2020_validation")
    g.set_defaults(func=cmd_gamestate)

    ab = sub.add_parser("ablation", help="value model with vs without spatial features")
    ab.add_argument("--train-dataset", default="euro2020_train")
    ab.add_argument("--test-dataset", default="euro2020_validation")
    ab.set_defaults(func=cmd_ablation)

    rp = sub.add_parser("report", help="summarise CDR and rank players")
    rp.add_argument("--dataset", default="euro2020_validation")
    rp.add_argument("--min-minutes", type=float, default=180.0)
    rp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
