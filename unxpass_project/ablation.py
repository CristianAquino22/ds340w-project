"""Does off-ball spatial context improve the value model?

Hassani, Ramdani & Lotfi (2025) argue that action-value models built on ball
location alone carry a "deficit of realism": the same pitch cell is worth the
same whether the goal is open or three defenders block the angle. un-xPass
applies SoccerMap -- which does see the freeze frame -- only to pass
*selection*. Its value component is VAEP, a feature-based model, and only
three of its seventeen features come from the 360 freeze frame:

    packing_rate, defenders_in_3m_radius, defenders_in_5m_radius

This module trains the value model with and without those three and compares
held-out performance. If DxT's argument holds, dropping them should hurt.
"""
from __future__ import annotations

from typing import Dict

from unxpass_project import compat  # noqa: F401
from unxpass_project import models


def _test_half(comp, half: str, dataset) -> Dict:
    """Test one half of the VAEP model, tolerating degenerate label sets.

    If no goal (or no concession) occurs within ten actions anywhere in the
    test set, the labels are single-class and ROC-AUC and log-loss are
    undefined. That is a property of the sample, not a failure, so it is
    reported rather than raised.
    """
    model = getattr(comp, f"{half}_model")
    try:
        return model.test(dataset)
    except ValueError as exc:
        return {"error": str(exc).split(".")[0]}


def run(train_dataset: str, test_dataset: str) -> Dict[str, Dict]:
    """Train both variants and test each on the held-out set."""
    results: Dict[str, Dict] = {}
    ds = models.dataset(test_dataset)
    for name in ("pass_value", "pass_value_nonspatial"):
        comp = models.train(name, train_dataset)
        results[name] = {
            half: _test_half(comp, half, ds) for half in ("offensive", "defensive")
        }
    return results


def report(train_dataset: str, test_dataset: str) -> None:
    res = run(train_dataset, test_dataset)
    spatial = res["pass_value"]
    plain = res["pass_value_nonspatial"]

    print("\n=== value model: with vs without off-ball spatial features ===")
    print(f"train: {train_dataset}   test: {test_dataset}")
    print(f"dropped features: {', '.join(models.VAEP_SPATIAL_FEATURES)}\n")

    header = f"{'model':<28} {'half':<10} {'ROC-AUC':>9} {'Brier':>9} {'log-loss':>10}"
    print(header)
    print("-" * len(header))
    rows = [
        ("VAEP (with spatial)", spatial),
        ("VAEP (no spatial)", plain),
    ]
    for label, metrics in rows:
        for half in ("offensive", "defensive"):
            m = metrics.get(half, {})
            if "error" in m:
                print(f"{label:<28} {half:<10} {'n/a -- ' + m['error']:>30}")
                continue
            print(
                f"{label:<28} {half:<10} "
                f"{m.get('roc_auc', float('nan')):>9.4f} "
                f"{m.get('brier', float('nan')):>9.5f} "
                f"{m.get('log_loss', float('nan')):>10.5f}"
            )

    print()
    for half in ("offensive", "defensive"):
        a = spatial.get(half, {}).get("roc_auc")
        b = plain.get(half, {}).get("roc_auc")
        if a is None or b is None:
            continue
        delta = a - b
        verdict = "helps" if delta > 0 else "does not help"
        print(f"{half}: spatial features change ROC-AUC by {delta:+.4f} -- {verdict}")
    print(
        "\nThe spatial features are three of seventeen, and they describe "
        "\ndefender proximity rather than the full freeze frame, so this is a "
        "\nlower bound on what full spatial context could contribute."
    )
