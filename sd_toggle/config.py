"""Config schema for SD toggle calibration parameters.

One JSON file per (hardware, model, tp) combination.
Load/save with validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any


@dataclass
class HardwareConfig:
    """Hardware-specific parameters."""
    gpu: str                # e.g. "A100-SXM4-80GB"
    tp: int                 # tensor parallelism
    BW_eff: float           # effective memory bandwidth (bytes/s)
    BW_peak: float = 0.0    # peak memory bandwidth (bytes/s)
    F_peak: float = 0.0     # peak compute (FLOPS)
    c_comm: float = 0.0     # pre-measured NCCL all-reduce overhead per forward pass (sec);
                            # typically 0 for tp=1, summed over 2·L layers for tp>1


@dataclass
class ModelConfig:
    """Deterministic model constants."""
    name: str
    W_t: float              # target weight bytes
    W_d: float              # drafter weight bytes
    C_dense: float          # per-token dense compute (FLOPS)
    C_attn: float           # per-token-per-context attention compute (FLOPS)
    kappa_theoretical: int  # KV cache bytes per (B*S)
    rho: float = 0.0        # W_d / W_t
    gqa: int = 1            # grouped query attention factor


@dataclass
class PerGammaCalibration:
    """Overhead calibration parameters.

    Used both as per-gamma overheads (legacy) and as a container for
    shared γ-independent overheads returned by get_gamma_params().
    """
    c_D: float              # drafter batch overhead (microseconds per batch element)
    c_V: float              # verify batch overhead (microseconds per batch element)
    c_T: float = 0.0        # target batch overhead (microseconds per batch element)
    R2: float = 0.0         # R-squared goodness of fit


@dataclass
class CalibrationConfig:
    """Calibration parameters — all gamma-independent (6 fitted params).

    Stage 1 fits: BW_eff (in HardwareConfig), kappa_eff, c_T
    Stage 2 fits: eta_d, c_D, c_V  (shared across all gammas)
    beta=0 fixed (tried, marginal gain, dropped).
    """
    eta_d: float                            # drafter overhead multiplier (gamma-independent)
    kappa_eff: float                        # effective KV cache coefficient (gamma-independent)
    F_eff: float                            # effective compute throughput (FLOPS)
    c_T: float = 0.0                        # target decode overhead (μs/batch) — shared, from Stage 1
    c_D: float = 0.0                        # drafter overhead (μs/batch) — shared, from Stage 2
    c_V: float = 0.0                        # verify overhead (μs/batch) — shared, from Stage 2
    beta: float = 0.0                       # verify memory traffic scaling (unused, kept for compat)
    per_gamma: Dict[int, PerGammaCalibration] = field(default_factory=dict)
    # Legacy: per-gamma eta_d/kappa_eff (kept for backward compat, empty in new pipeline)
    per_gamma_full: Optional[Dict[int, Dict[str, float]]] = None


@dataclass
class SDToggleConfig:
    """Complete SD toggle configuration for a (hardware, model) combination."""
    hardware: HardwareConfig
    model: ModelConfig
    calibration: CalibrationConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_gamma_params(self, gamma: int) -> PerGammaCalibration:
        """Get per-gamma calibration.

        Falls back to shared top-level c_D/c_V when per_gamma is empty (new pipeline),
        or nearest gamma for legacy configs.
        """
        if gamma in self.calibration.per_gamma:
            return self.calibration.per_gamma[gamma]
        # Fall back to shared c_D/c_V (new pipeline: per_gamma is empty)
        if self.calibration.c_D > 0 or self.calibration.c_V > 0:
            return PerGammaCalibration(
                c_D=self.calibration.c_D,
                c_V=self.calibration.c_V,
                c_T=self.calibration.c_T,
            )
        # Legacy: nearest gamma
        available = sorted(self.calibration.per_gamma.keys())
        if not available:
            raise ValueError("No calibration data available")
        nearest = min(available, key=lambda g: abs(g - gamma))
        import warnings
        warnings.warn(
            f"gamma={gamma} not calibrated, using nearest gamma={nearest}",
            stacklevel=2,
        )
        return self.calibration.per_gamma[nearest]


def save_config(config: SDToggleConfig, path: str | Path) -> None:
    """Save SDToggleConfig to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "hardware": {
            "gpu": config.hardware.gpu,
            "tp": config.hardware.tp,
            "BW_peak": config.hardware.BW_peak,
            "F_peak": config.hardware.F_peak,
            "BW_eff": config.hardware.BW_eff,
            "c_comm": config.hardware.c_comm,
        },
        "model": {
            "name": config.model.name,
            "W_t": config.model.W_t,
            "W_d": config.model.W_d,
            "C_dense": config.model.C_dense,
            "C_attn": config.model.C_attn,
            "kappa_theoretical": config.model.kappa_theoretical,
            "rho": config.model.rho,
            "gqa": config.model.gqa,
        },
        "calibration": {
            "eta_d": config.calibration.eta_d,
            "kappa_eff": config.calibration.kappa_eff,
            "F_eff": config.calibration.F_eff,
            "c_T": config.calibration.c_T,
            "c_D": config.calibration.c_D,
            "c_V": config.calibration.c_V,
            "beta": config.calibration.beta,
            "per_gamma": {
                str(g): {
                    "c_T": cal.c_T,
                    "c_D": cal.c_D,
                    "c_V": cal.c_V,
                    "R2": cal.R2,
                }
                for g, cal in sorted(config.calibration.per_gamma.items())
            },
        },
        "metadata": config.metadata,
    }

    if config.calibration.per_gamma_full:
        data["calibration"]["per_gamma_full"] = {
            str(g): v
            for g, v in sorted(config.calibration.per_gamma_full.items())
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: str | Path) -> SDToggleConfig:
    """Load SDToggleConfig from JSON file.

    Args:
        path: Path to config JSON file

    Returns:
        Validated SDToggleConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is missing required fields or has invalid values
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    _validate_config_data(data)

    hw = data["hardware"]
    hardware = HardwareConfig(
        gpu=hw["gpu"],
        tp=hw["tp"],
        BW_eff=hw["BW_eff"],
        BW_peak=hw.get("BW_peak", 0.0),
        F_peak=hw.get("F_peak", 0.0),
        c_comm=hw.get("c_comm", 0.0),
    )

    md = data["model"]
    model = ModelConfig(
        name=md["name"],
        W_t=md["W_t"],
        W_d=md["W_d"],
        C_dense=md["C_dense"],
        C_attn=md["C_attn"],
        kappa_theoretical=md["kappa_theoretical"],
        rho=md.get("rho", 0.0),
        gqa=md.get("gqa", 1),
    )

    cal = data["calibration"]
    per_gamma = {}
    for g_str, gc in cal.get("per_gamma", {}).items():
        per_gamma[int(g_str)] = PerGammaCalibration(
            c_D=gc["c_D"],
            c_V=gc["c_V"],
            c_T=gc.get("c_T", 0.0),
            R2=gc.get("R2", 0.0),
        )

    per_gamma_full = None
    if "per_gamma_full" in cal:
        per_gamma_full = {int(g): v for g, v in cal["per_gamma_full"].items()}

    calibration = CalibrationConfig(
        eta_d=cal["eta_d"],
        kappa_eff=cal["kappa_eff"],
        F_eff=cal["F_eff"],
        c_T=cal.get("c_T", 0.0),
        c_D=cal.get("c_D", 0.0),
        c_V=cal.get("c_V", 0.0),
        beta=cal.get("beta", 0.0),
        per_gamma=per_gamma,
        per_gamma_full=per_gamma_full,
    )

    return SDToggleConfig(
        hardware=hardware,
        model=model,
        calibration=calibration,
        metadata=data.get("metadata", {}),
    )


def _validate_config_data(data: dict) -> None:
    """Validate config JSON structure and value ranges."""
    for section in ("hardware", "model", "calibration"):
        if section not in data:
            raise ValueError(f"Missing required section: {section}")

    hw = data["hardware"]
    for key in ("gpu", "tp", "BW_eff"):
        if key not in hw:
            raise ValueError(f"Missing hardware.{key}")
    if hw["BW_eff"] <= 0:
        raise ValueError(f"BW_eff must be positive, got {hw['BW_eff']}")

    md = data["model"]
    for key in ("name", "W_t", "W_d", "C_dense", "C_attn", "kappa_theoretical"):
        if key not in md:
            raise ValueError(f"Missing model.{key}")

    cal = data["calibration"]
    for key in ("eta_d", "kappa_eff", "F_eff"):
        if key not in cal:
            raise ValueError(f"Missing calibration.{key}")
    if cal["eta_d"] < 1.0:
        raise ValueError(f"eta_d should be >= 1.0, got {cal['eta_d']}")
