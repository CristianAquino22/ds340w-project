"""Stage 4 -- the three model components behind CDR.

CDR needs three estimates (Robberechts et al., 2023, Section 2.3):

1. ``pass_selection``  P(D = l | S)        -- where a typical player passes
2. ``pass_success``    P(O = o | D = l, S) -- whether that pass completes
3. ``pass_value``      Q(S | D = l, O = o) -- what the resulting state is worth

Each is built by its own factory below, so a component can be swapped without
touching the others -- which is the point: the planned extension replaces the
value component with a spatially-aware one.
"""
from __future__ import annotations

import time
from functools import partial
from typing import Callable, Dict

from unxpass_project import compat  # noqa: F401
from unxpass_project import config

from unxpass.components import pass_selection, pass_success, pass_value
from unxpass.datasets import PassesDataset


# --- pass success feature set, from config/experiment/pass_success/threesixty.yaml
PASS_SUCCESS_FEATURES: Dict[str, list] = {
    "startpolar": ["start_dist_to_goal_a0", "start_angle_to_goal_a0"],
    "relative_startlocation": ["start_dist_goalline_a0", "start_dist_sideline_a0"],
    "endpolar": ["end_dist_to_goal_a0", "end_angle_to_goal_a0"],
    "relative_endlocation": ["end_dist_goalline_a0", "end_dist_sideline_a0"],
    "movement": ["movement_a0", "dx_a0", "dy_a0"],
    "angle": ["angle_a0"],
    "ball_height_onehot": [
        "ball_height_ground_a0",
        "ball_height_low_a0",
        "ball_height_high_a0",
    ],
    "player_possession_time": ["player_possession_time_a0"],
    "speed": ["speed_a01", "speed_a02"],
    "under_pressure": ["under_pressure_a0"],
    "dist_defender": [
        "dist_defender_start_a0",
        "dist_defender_end_a0",
        "dist_defender_action_a0",
    ],
    "nb_opp_in_path": ["nb_opp_in_path_a0"],
}

# --- VAEP feature set, from config/experiment/pass_value/vaep_xg_360.yaml
#
# The last three are derived from the StatsBomb 360 freeze frame: they are the
# only features in the value model that know where the off-ball players are.
# Splitting them out lets us test the claim Hassani et al. (2025) make about
# DxT -- that a value model ignoring off-ball positions is missing something.
VAEP_SPATIAL_FEATURES = [
    "packing_rate",
    "defenders_in_3m_radius",
    "defenders_in_5m_radius",
]

VAEP_BASE_FEATURES = [
    "actiontype_onehot", "result_onehot", "actiontype_result_onehot",
    "bodypart_onehot", "time", "startlocation", "endlocation", "startpolar",
    "endpolar", "movement", "team", "time_delta", "space_delta", "goalscore",
]

VAEP_FEATURES = VAEP_BASE_FEATURES + VAEP_SPATIAL_FEATURES

VAEP_XGB_PARAMS = dict(
    objective="reg:logistic",
    eval_metric="rmse",
    learning_rate=0.09824518286128303,
    max_depth=6,
    gamma=1.719417635880229e-06,
    reg_lambda=0.085802210677737,
    reg_alpha=1.042204226716464e-08,
    subsample=0.7856643781233339,
    colsample_bytree=0.29338345593959264,
)


def build_pass_selection(lr: float = 1e-4):
    """SoccerMap CNN over the pitch -- the surface CDR takes its argmax from."""
    return pass_selection.SoccerMapComponent(
        model=pass_selection.PytorchSoccerMapModel(lr=lr)
    )


def build_pass_success():
    """XGBoost on hand-crafted features, per the paper's 'threesixty' config."""
    from xgboost import XGBClassifier

    return pass_success.XGBoostComponent(
        model=XGBClassifier(objective="binary:logistic", eval_metric="auc"),
        features=PASS_SUCCESS_FEATURES,
    )


def build_pass_value(spatial: bool = True):
    """VAEP: separate offensive and defensive models, differenced at predict.

    Parameters
    ----------
    spatial
        Include the three freeze-frame features (`packing_rate`,
        `defenders_in_3m_radius`, `defenders_in_5m_radius`). Setting this to
        False gives a value model blind to off-ball positions, which is the
        ablation used to test DxT's argument.
    """
    from xgboost import XGBRegressor

    features = VAEP_FEATURES if spatial else VAEP_BASE_FEATURES
    return pass_value.VaepModel(
        offensive_model=pass_value.OffensiveVaepModel(
            model=XGBRegressor(**VAEP_XGB_PARAMS),
            features=features,
            label=["scores", "scores_xg"],
        ),
        defensive_model=pass_value.DefensiveVaepModel(
            model=XGBRegressor(**VAEP_XGB_PARAMS),
            features=features,
            label=["concedes", "concedes_xg"],
        ),
    )


def build_pass_value_nonspatial():
    return build_pass_value(spatial=False)


BUILDERS: Dict[str, Callable] = {
    "pass_selection": build_pass_selection,
    "pass_success": build_pass_success,
    "pass_value": build_pass_value,
    "pass_value_nonspatial": build_pass_value_nonspatial,
}


def dataset(name: str):
    return partial(PassesDataset, path=config.DATASETS / name)


def train(component_name: str, train_dataset: str, max_epochs: int = 20,
          accelerator: str = "cpu", batch_size: int = 32, num_workers: int = 0):
    """Train one component and save it under ``stores/models``."""
    import pickle

    import contextlib

    comp = BUILDERS[component_name]()
    ds = dataset(train_dataset)
    t0 = time.time()
    # The components call mlflow.*.log_model() internally, which needs an
    # active run. MLflow tracking is optional here -- models are pickled below
    # -- and its local-path handling is fragile on Windows, so if starting a
    # run fails for any reason we disable the internal logging and carry on.
    run_ctx = _mlflow_run(component_name)
    with run_ctx:
        if component_name == "pass_selection":
            comp.train(
                ds,
                trainer={"max_epochs": max_epochs, "accelerator": accelerator},
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            comp.train(ds)
    print(f"[train] {component_name}: {time.time() - t0:.0f}s")
    out = config.MODELS / f"{component_name}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(comp, fh)
    print(f"[train] saved -> {out}")
    return comp


def load(component_name: str):
    import pickle

    with open(config.MODELS / f"{component_name}.pkl", "rb") as fh:
        return pickle.load(fh)


def _mlflow_run(component_name: str):
    """An MLflow run if tracking works here, otherwise a no-op context.

    Returns a context manager either way. When MLflow cannot be initialised we
    also stub out the ``log_model`` helpers the components call, since those
    require an active run.
    """
    import contextlib

    try:
        import mlflow

        mlflow.set_experiment(experiment_name=f"unxpass_{component_name}")
        return mlflow.start_run()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[train] MLflow tracking unavailable ({type(exc).__name__}); "
              "continuing without it. Models are still saved to stores/models.")
        _disable_mlflow_logging()
        return contextlib.nullcontext()


def _disable_mlflow_logging() -> None:
    """Make the components' internal MLflow logging a no-op."""
    try:
        import mlflow
    except Exception:  # pragma: no cover
        return

    def _noop(*_args, **_kwargs):
        return None

    for mod in ("sklearn", "xgboost", "pytorch"):
        try:
            getattr(mlflow, mod).log_model = _noop
        except Exception:
            pass
    try:
        from unxpass.components import utils as comp_utils

        comp_utils.log_model = _noop
    except Exception:
        pass
