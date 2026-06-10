"""sd_toggle: Roofline-aware speculative decoding toggle calibration.

Provides offline calibration pipeline and toggle decision function
for determining when speculative decoding is beneficial.

Public API:
    load_config(path) -> SDToggleConfig
    save_config(config, path)
    should_enable_sd(config, B, S, gamma, L_accept) -> bool
    calibrate(csv_path, model_name, ...) -> SDToggleConfig
    compute_constants(model_name, tp) -> ModelConstants
"""
from .config import SDToggleConfig, load_config, save_config
from .predict import should_enable_sd, predict_decision, predict_batch
from .fit import calibrate
from .constants import compute_constants, ModelConstants
from .sweep import run_sweep

__all__ = [
    "SDToggleConfig",
    "load_config",
    "save_config",
    "should_enable_sd",
    "predict_decision",
    "predict_batch",
    "calibrate",
    "compute_constants",
    "ModelConstants",
    "run_sweep",
]
