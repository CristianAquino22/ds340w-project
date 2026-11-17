#!/usr/bin/env python
"""Pipeline runner for the un-xPass tournament reproduction.

Each stage is a separate subcommand so stages can be re-run independently --
feature computation and SoccerMap training are slow, and you do not want to
redo them to change something downstream.

    python run.py download --competition euro2020 [--limit 5]
    python run.py split    --competition euro2020
    python run.py features --split train
    python run.py train    --component pass_success
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
        f"{len(sp['validation'])} validation matches (split by match, not by pass)"
    )


def cmd_features(args):
    if args.competition == config.TEST_COMPETITION:
        ids = data.game_ids(args.competition)
        name = f"{args.competition}_test"
    else:
        sp = splits.read_split(args.competition)
        ids = sp[args.split]
        name = f"{args.competition}_{args.split}"
    if args.limit:
        ids = ids[: args.limit]
    featurize.build(name, ids)


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

    s = sub.add_parser("split", help="split a competition into train/validation by match")
    s.add_argument("--competition", default=config.TRAIN_COMPETITION,
                   choices=list(config.COMPETITIONS))
    s.set_defaults(func=cmd_split)

    f = sub.add_parser("features", help="compute features and labels")
    f.add_argument("--competition", default=config.TRAIN_COMPETITION,
                   choices=list(config.COMPETITIONS))
    f.add_argument("--split", default="train", choices=["train", "validation"])
    f.add_argument("--limit", type=int, default=None)
    f.set_defaults(func=cmd_features)

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

    rp = sub.add_parser("report", help="summarise CDR and rank players")
    rp.add_argument("--dataset", default="euro2020_validation")
    rp.add_argument("--min-minutes", type=float, default=180.0)
    rp.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
