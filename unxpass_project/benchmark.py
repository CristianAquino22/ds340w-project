"""Stage 6 -- compare component performance against the parent paper.

The parent paper reports pass-selection accuracy for three models on the top-10
Premier League teams of 2021/22 (Robberechts et al., 2023, Table 1):

    closest teammate baseline   0.395
    XGBoost                     0.538
    SoccerMap                   0.513

Accuracy here is the share of passes whose actual receiver was the model's
most likely destination. SoccerMap's ``test()`` derives the receiver from the
freeze frame, so it needs no extra labels; the baseline and the XGBoost
variant need per-pass-option features, which the default feature set omits --
see :func:`unxpass_project.featurize.build_pass_options`.
"""
from __future__ import annotations

from typing import Dict, Optional

from unxpass_project import compat  # noqa: F401
from unxpass_project import config, models

# Robberechts, Van Roy & Davis (2023), Table 1.
PAPER_ACCURACY = {
    "baseline_closest_teammate": 0.395,
    "pass_selection_xgboost": 0.538,
    "pass_selection_soccermap": 0.513,
}


def soccermap_accuracy(dataset_name: str) -> Dict[str, float]:
    """Pass-selection accuracy of the trained SoccerMap component."""
    comp = models.load("pass_selection")
    return comp.test(models.dataset(dataset_name))


def baseline_accuracy(dataset_name: str) -> Optional[Dict[str, float]]:
    """Closest-teammate baseline. Needs a pass_options dataset.

    Returns None when that dataset has not been built.
    """
    from unxpass.components.pass_selection import ClosestPlayerBaseline

    path = config.DATASETS / dataset_name
    if not (path / "x_pass_options.parquet").exists():
        return None
    return ClosestPlayerBaseline().test(models.dataset(dataset_name))


def report(dataset_name: str, options_dataset: Optional[str] = None) -> None:
    """Print a side-by-side comparison with the paper's Table 1."""
    rows = []

    sm = soccermap_accuracy(dataset_name)
    rows.append(("SoccerMap (pass selection)", sm.get("acc"),
                 PAPER_ACCURACY["pass_selection_soccermap"]))

    if options_dataset:
        base = baseline_accuracy(options_dataset)
        if base is not None:
            rows.append(("Closest-teammate baseline", base.get("acc"),
                         PAPER_ACCURACY["baseline_closest_teammate"]))

    width = max(len(name) for name, _, _ in rows)
    print(f"\n=== pass selection accuracy vs parent paper ===")
    print(f"{'model'.ljust(width)}   ours    paper")
    for name, ours, paper in rows:
        ours_s = f"{ours:.3f}" if ours is not None else "  n/a"
        print(f"{name.ljust(width)}  {ours_s}   {paper:.3f}")

    extra = {k: v for k, v in sm.items() if k != "acc"}
    if extra:
        print("\nSoccerMap, other metrics on the same set:")
        for k, v in extra.items():
            print(f"  {k}: {v:.5f}")
    print(
        "\nPaper figures are for the top-10 Premier League teams of 2021/22;\n"
        "ours are for held-out tournament matches, so they are comparable in\n"
        "kind rather than exactly."
    )
