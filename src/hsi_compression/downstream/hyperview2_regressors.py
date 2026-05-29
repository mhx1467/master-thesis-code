from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegressorSpec:
    name: str
    dependency: str
    family: str
    description: str


DEFAULT_REGRESSORS = (
    "dummy_mean",
    "ridge",
    "pls",
    "knn",
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
)

BOOSTING_REGRESSORS = ("lightgbm", "catboost", "xgboost")

REGRESSOR_SPECS: Mapping[str, RegressorSpec] = {
    "dummy_mean": RegressorSpec(
        name="dummy_mean",
        dependency="sklearn",
        family="sanity_baseline",
        description="Train-mean baseline used to normalize the HYPERVIEW2 score.",
    ),
    "ridge": RegressorSpec(
        name="ridge",
        dependency="sklearn",
        family="linear",
        description="Scaled multi-target ridge regression with cross-validated alpha.",
    ),
    "pls": RegressorSpec(
        name="pls",
        dependency="sklearn",
        family="linear_latent",
        description="Partial least squares regression, common for chemometric spectra.",
    ),
    "knn": RegressorSpec(
        name="knn",
        dependency="sklearn",
        family="nonparametric",
        description="Scaled KNN regressor inspired by classical HYPERVIEW baselines.",
    ),
    "extra_trees": RegressorSpec(
        name="extra_trees",
        dependency="sklearn",
        family="tree_ensemble",
        description="Extremely randomized trees, close to the public HYPERVIEW2 winner style.",
    ),
    "random_forest": RegressorSpec(
        name="random_forest",
        dependency="sklearn",
        family="tree_ensemble",
        description="Random forest baseline used broadly in HSI soil-regression work.",
    ),
    "hist_gradient_boosting": RegressorSpec(
        name="hist_gradient_boosting",
        dependency="sklearn",
        family="boosting",
        description="Sklearn histogram gradient boosting wrapped for multi-output regression.",
    ),
    "lightgbm": RegressorSpec(
        name="lightgbm",
        dependency="lightgbm",
        family="boosting",
        description="Optional LightGBM multi-output wrapper for tabular spectral features.",
    ),
    "catboost": RegressorSpec(
        name="catboost",
        dependency="catboost",
        family="boosting",
        description="Optional CatBoost multi-output wrapper for tabular spectral features.",
    ),
    "xgboost": RegressorSpec(
        name="xgboost",
        dependency="xgboost",
        family="boosting",
        description="Optional XGBoost multi-output wrapper for tabular spectral features.",
    ),
}


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def available_regressor_names(include_unavailable: bool = False) -> list[str]:
    if include_unavailable:
        return list(REGRESSOR_SPECS)
    return [name for name, spec in REGRESSOR_SPECS.items() if _has_module(spec.dependency)]


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import (
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import RidgeCV
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "HYPERVIEW2 sklearn regressors require optional downstream dependencies. "
            "Install them with: pip install -e '.[downstream]'"
        ) from exc
    return {
        "DummyRegressor": DummyRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "MultiOutputRegressor": MultiOutputRegressor,
        "PLSRegression": PLSRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "RidgeCV": RidgeCV,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def _bounded_pls_components(
    requested: int,
    n_features: int | None,
    n_samples: int | None,
    n_targets: int | None,
) -> int:
    upper = requested
    if n_features is not None:
        upper = min(upper, max(1, int(n_features)))
    if n_samples is not None:
        upper = min(upper, max(1, int(n_samples) - 1))
    if n_targets is not None:
        upper = min(upper, max(1, int(n_targets)))
    return max(1, int(upper))


def build_hyperview2_regressor(
    name: str,
    *,
    random_state: int = 42,
    n_jobs: int | None = -1,
    n_features: int | None = None,
    n_samples: int | None = None,
    n_targets: int | None = None,
    **overrides: Any,
) -> Any:
    if name not in REGRESSOR_SPECS:
        available = ", ".join(REGRESSOR_SPECS)
        raise ValueError(f"Unknown HYPERVIEW2 regressor {name!r}. Available: {available}")

    sklearn = _require_sklearn()
    make_pipeline = sklearn["make_pipeline"]
    standard_scaler = sklearn["StandardScaler"]
    multi_output = sklearn["MultiOutputRegressor"]

    if name == "dummy_mean":
        return sklearn["DummyRegressor"](strategy=overrides.pop("strategy", "mean"), **overrides)

    if name == "ridge":
        alphas = overrides.pop("alphas", (0.01, 0.1, 1.0, 10.0, 100.0))
        model = sklearn["RidgeCV"](alphas=alphas, **overrides)
        return make_pipeline(standard_scaler(), model)

    if name == "pls":
        requested = int(overrides.pop("n_components", 16))
        components = _bounded_pls_components(requested, n_features, n_samples, n_targets)
        model = sklearn["PLSRegression"](n_components=components, scale=True, **overrides)
        return model

    if name == "knn":
        neighbors = int(overrides.pop("n_neighbors", 8))
        model = sklearn["KNeighborsRegressor"](
            n_neighbors=neighbors,
            weights=overrides.pop("weights", "distance"),
            n_jobs=n_jobs,
            **overrides,
        )
        return make_pipeline(standard_scaler(), model)

    if name == "extra_trees":
        return sklearn["ExtraTreesRegressor"](
            n_estimators=int(overrides.pop("n_estimators", 600)),
            max_features=overrides.pop("max_features", "sqrt"),
            min_samples_leaf=int(overrides.pop("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=n_jobs,
            **overrides,
        )

    if name == "random_forest":
        return sklearn["RandomForestRegressor"](
            n_estimators=int(overrides.pop("n_estimators", 500)),
            max_features=overrides.pop("max_features", "sqrt"),
            min_samples_leaf=int(overrides.pop("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=n_jobs,
            **overrides,
        )

    if name == "hist_gradient_boosting":
        base = sklearn["HistGradientBoostingRegressor"](
            max_iter=int(overrides.pop("max_iter", 400)),
            learning_rate=float(overrides.pop("learning_rate", 0.05)),
            l2_regularization=float(overrides.pop("l2_regularization", 0.0)),
            random_state=random_state,
            **overrides,
        )
        return multi_output(base, n_jobs=n_jobs)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "The lightgbm regressor requires: pip install -e '.[downstream-boosting]'"
            ) from exc
        base = LGBMRegressor(
            n_estimators=int(overrides.pop("n_estimators", 800)),
            learning_rate=float(overrides.pop("learning_rate", 0.03)),
            num_leaves=int(overrides.pop("num_leaves", 31)),
            random_state=random_state,
            n_jobs=n_jobs,
            **overrides,
        )
        return multi_output(base, n_jobs=n_jobs)

    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "The catboost regressor requires: pip install -e '.[downstream-boosting]'"
            ) from exc
        base = CatBoostRegressor(
            iterations=int(overrides.pop("iterations", 800)),
            learning_rate=float(overrides.pop("learning_rate", 0.03)),
            depth=int(overrides.pop("depth", 6)),
            loss_function=overrides.pop("loss_function", "RMSE"),
            random_seed=random_state,
            verbose=overrides.pop("verbose", False),
            allow_writing_files=overrides.pop("allow_writing_files", False),
            **overrides,
        )
        return multi_output(base, n_jobs=n_jobs)

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "The xgboost regressor requires: pip install -e '.[downstream-boosting]'"
            ) from exc
        base = XGBRegressor(
            n_estimators=int(overrides.pop("n_estimators", 800)),
            learning_rate=float(overrides.pop("learning_rate", 0.03)),
            max_depth=int(overrides.pop("max_depth", 6)),
            objective=overrides.pop("objective", "reg:squarederror"),
            tree_method=overrides.pop("tree_method", "hist"),
            random_state=random_state,
            n_jobs=n_jobs,
            **overrides,
        )
        return multi_output(base, n_jobs=n_jobs)

    raise AssertionError(f"Unhandled regressor: {name}")
