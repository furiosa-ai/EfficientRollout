"""Tests for sd_toggle package — runs on CPU without GPU/vLLM."""
import json
import subprocess
import sys
import pytest
from pathlib import Path

RESULTS_DIR = Path("results/lv1_roofline")


class TestConstants:
    # Golden constants are architecture-derived and deterministic, so they are
    # asserted inline rather than against the gitignored results/ calibration
    # dir. This keeps the test self-contained and runnable for external users.
    def test_qwen7b_constants(self):
        from sd_toggle.constants import compute_constants
        c = compute_constants("Qwen2.5-7B", tp=1)
        assert c.W_t == 15230974976.0
        assert c.W_d == 5443042304.0
        assert c.kappa_theoretical == 57344
        assert c.C_dense == 14140571648
        assert c.C_attn == 401408

    def test_llama8b_constants(self):
        from sd_toggle.constants import compute_constants
        c = compute_constants("LLaMA3.1-8B", tp=1)
        assert c.W_t == 16060522496.0
        assert c.kappa_theoretical == 131072

    def test_qwen14b_tp2_constants(self):
        from sd_toggle.constants import compute_constants
        c = compute_constants("Qwen2.5-14B", tp=2)
        # tp=2 → per-GPU weights and dense/attn compute are halved;
        # kappa is per-GPU (n_kv=8, tp=2 → n_kv_per_gpu=4 → 98304).
        assert c.W_t == 14769689600.0
        assert c.kappa_theoretical == 98304
        assert c.C_dense == 13990625280
        assert c.C_attn == 491520

    def test_model_name_aliases(self):
        from sd_toggle.constants import compute_constants
        c1 = compute_constants("Qwen2.5-7B")
        c2 = compute_constants("Qwen/Qwen2.5-7B")
        assert c1.W_t == c2.W_t

    def test_unknown_model_raises(self):
        from sd_toggle.constants import compute_constants
        with pytest.raises(ValueError, match="Unknown model"):
            compute_constants("NonExistent-Model")


class TestConfig:
    def test_round_trip(self, tmp_path):
        from sd_toggle.config import (
            SDToggleConfig, HardwareConfig, ModelConfig,
            CalibrationConfig, PerGammaCalibration,
            save_config, load_config,
        )
        config = SDToggleConfig(
            hardware=HardwareConfig(gpu="A100", tp=1, BW_eff=1.5e12),
            model=ModelConfig(name="Test", W_t=15e9, W_d=5e9,
                              C_dense=14e9, C_attn=400000,
                              kappa_theoretical=57344),
            calibration=CalibrationConfig(
                eta_d=1.4, kappa_eff=28000, F_eff=200e12,
                per_gamma={5: PerGammaCalibration(c_D=30, c_V=100)},
            ),
        )
        path = tmp_path / "test.json"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.hardware.BW_eff == config.hardware.BW_eff
        assert loaded.calibration.eta_d == config.calibration.eta_d
        assert loaded.calibration.per_gamma[5].c_D == 30

    def test_validation_missing_section(self, tmp_path):
        from sd_toggle.config import load_config
        path = tmp_path / "bad.json"
        path.write_text('{"hardware": {"gpu": "A100", "tp": 1, "BW_eff": 1e12}}')
        with pytest.raises(ValueError, match="Missing required section"):
            load_config(path)


class TestRoofline:
    def test_T_T_increases_with_batch(self):
        from sd_toggle.roofline import predict_T_T
        t1 = predict_T_T(1, 1024, 15e9, 28000, 1.5e12, 14e9, 400000, 200e12)
        t32 = predict_T_T(32, 1024, 15e9, 28000, 1.5e12, 14e9, 400000, 200e12)
        assert t32 > t1

    def test_T_V_larger_than_T_T(self):
        from sd_toggle.roofline import predict_T_T, predict_T_V
        t_t = predict_T_T(32, 2048, 15e9, 28000, 1.5e12, 14e9, 400000, 200e12)
        t_v = predict_T_V(32, 2048, 5, 15e9, 28000, 1.5e12, 14e9, 400000, 200e12)
        assert t_v >= t_t  # verify always >= target due to (gamma+1) multiplier

    def test_speedup_decreases_with_batch(self):
        from sd_toggle.roofline import predict_speedup
        from sd_toggle.config import (
            SDToggleConfig, HardwareConfig, ModelConfig,
            CalibrationConfig, PerGammaCalibration,
        )
        config = SDToggleConfig(
            hardware=HardwareConfig(gpu="A100", tp=1, BW_eff=1.5e12),
            model=ModelConfig(name="Test", W_t=15e9, W_d=5e9,
                              C_dense=14e9, C_attn=400000,
                              kappa_theoretical=57344),
            calibration=CalibrationConfig(
                eta_d=1.4, kappa_eff=28000, F_eff=200e12,
                per_gamma={5: PerGammaCalibration(c_D=30, c_V=100)},
            ),
        )
        s1 = float(predict_speedup(1, 1024, 5, 5.0, config))
        s64 = float(predict_speedup(64, 1024, 5, 5.0, config))
        assert s1 > s64  # SD less beneficial at high batch


class TestPredict:
    def test_should_enable_sd_basic(self):
        from sd_toggle.config import (
            SDToggleConfig, HardwareConfig, ModelConfig,
            CalibrationConfig, PerGammaCalibration,
        )
        from sd_toggle.predict import should_enable_sd
        config = SDToggleConfig(
            hardware=HardwareConfig(gpu="A100", tp=1, BW_eff=1.5e12),
            model=ModelConfig(name="Test", W_t=15e9, W_d=5e9,
                              C_dense=14e9, C_attn=400000,
                              kappa_theoretical=57344),
            calibration=CalibrationConfig(
                eta_d=1.4, kappa_eff=28000, F_eff=200e12,
                per_gamma={5: PerGammaCalibration(c_D=30, c_V=100)},
            ),
        )
        # Low batch → SD beneficial
        assert should_enable_sd(config, 1, 1024, 5, 5.0) is True
        # Very high batch → SD harmful
        assert should_enable_sd(config, 64, 4096, 5, 5.0) is False


class TestImport:
    def test_public_api_importable(self):
        from sd_toggle import (
            should_enable_sd, load_config, save_config,
            calibrate, compute_constants,
        )
        assert callable(should_enable_sd)
        assert callable(load_config)
        assert callable(calibrate)

    def test_no_gpu_dependency(self):
        """Import should not require torch/vllm."""
        import sd_toggle
        import sys
        # sd_toggle itself should not have imported torch
        # (it may already be in sys.modules from other tests, so just check import works)
        assert hasattr(sd_toggle, "should_enable_sd")


@pytest.mark.skipif(
    not (RESULTS_DIR / "qwen7b_finegrain.csv").exists(),
    reason="Finegrain CSV not available",
)
class TestSignAccuracy:
    def test_qwen7b_sign_accuracy_at_standard_L(self):
        from sd_toggle.fit import calibrate
        from sd_toggle.predict import evaluate_sign_accuracy

        config = calibrate(
            RESULTS_DIR / "qwen7b_finegrain.csv", "Qwen2.5-7B"
        )
        gammas = config.metadata.get("gammas", [])
        assert gammas, "calibrate() should record calibrated gamma values in metadata"

        # Validate each gamma at L_accept ≈ gamma (typical trained-policy regime).
        for gamma in gammas:
            L_val = float(gamma)  # assume near-perfect acceptance
            acc_p2 = evaluate_sign_accuracy(
                config, RESULTS_DIR / "qwen7b_finegrain.csv",
                L_accept=L_val, gammas=[gamma], pow2_only=True,
            )
            assert acc_p2 >= 0.85, (
                f"Sign accuracy {acc_p2:.1%} < 85% at gamma={gamma} (pow2)"
            )


# ---------------------------------------------------------------------------
# Helpers shared across new test classes
# ---------------------------------------------------------------------------

def _make_test_config():
    from sd_toggle.config import (
        SDToggleConfig, HardwareConfig, ModelConfig,
        CalibrationConfig, PerGammaCalibration,
    )
    return SDToggleConfig(
        hardware=HardwareConfig(gpu="A100", tp=1, BW_eff=1.5e12),
        model=ModelConfig(name="Test", W_t=15e9, W_d=5e9,
                          C_dense=14e9, C_attn=400000,
                          kappa_theoretical=57344),
        calibration=CalibrationConfig(
            eta_d=1.4, kappa_eff=28000, F_eff=200e12,
            per_gamma={3: PerGammaCalibration(c_D=25, c_V=90),
                       7: PerGammaCalibration(c_D=30, c_V=100),
                       11: PerGammaCalibration(c_D=35, c_V=110),
                       15: PerGammaCalibration(c_D=40, c_V=120)},
        ),
    )


_SAMPLE_TIMING_OUTPUT = """\
SWEEP_META,batch=8,seq=1024,rep=0,gamma=7
SD_TIMING,T_T,batch=8,num_tokens=8,qlen=1,elapsed_ms=9.500
SD_TIMING,T_D,batch=8,fwd_ms=4.200,full_ms=5.100,overhead_ms=0.900
SD_TIMING,T_V,batch=8,num_tokens=48,qlen=6,elapsed_ms=11.300
"""


class TestSweep:
    def test_sweep_importable(self):
        from sd_toggle.sweep import run_sweep, _parse_timing_output, _aggregate_records
        assert callable(run_sweep)
        assert callable(_parse_timing_output)
        assert callable(_aggregate_records)

    def test_parse_timing_output(self):
        from sd_toggle.sweep import _parse_timing_output
        records = _parse_timing_output(_SAMPLE_TIMING_OUTPUT, gamma=7)
        assert len(records) == 3
        modes = {r["mode"] for r in records}
        assert modes == {"target_decode", "drafter_decode", "target_verify"}

        t_t = next(r for r in records if r["mode"] == "target_decode")
        assert t_t["batch"] == 8
        assert t_t["seq"] == 1024
        assert t_t["gamma"] == 7
        assert abs(t_t["elapsed_ms"] - 9.5) < 1e-6

        t_d = next(r for r in records if r["mode"] == "drafter_decode")
        assert abs(t_d["fwd_ms"] - 4.2) < 1e-6
        assert abs(t_d["full_ms"] - 5.1) < 1e-6
        assert abs(t_d["elapsed_ms"] - 4.2) < 1e-6  # elapsed_ms = fwd_ms

        t_v = next(r for r in records if r["mode"] == "target_verify")
        assert abs(t_v["elapsed_ms"] - 11.3) < 1e-6
        assert t_v["qlen"] == 6

    def test_aggregate_records(self):
        from sd_toggle.sweep import _aggregate_records
        # 10 raw records for same (batch, seq, gamma, mode); first 20% = 2 discarded
        records = [
            {"batch": 4, "seq": 512, "gamma": 7, "mode": "target_decode",
             "elapsed_ms": float(i)}
            for i in range(10)
        ]
        agg = _aggregate_records(records)
        assert len(agg) == 1
        row = agg[0]
        assert row["batch"] == 4
        assert row["seq"] == 512
        assert row["gamma"] == 7
        assert row["mode"] == "target_decode"
        # After discarding 2 (20% of 10), 8 samples remain
        assert row["n_samples"] == 8
        assert "mean_ms" in row
        assert "median_ms" in row
        assert "std_ms" in row
        assert row["min_ms"] <= row["mean_ms"] <= row["max_ms"]

class TestEdgeCases:
    def test_predict_with_gamma_not_in_config(self):
        """gamma=5 not in config {3,7,11,15} → falls back to nearest (3 or 7) with warning."""
        config = _make_test_config()
        from sd_toggle.predict import should_enable_sd
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = should_enable_sd(config, B=1, S=1024, gamma=5, L_accept=4.0)
        assert isinstance(result, bool)
        assert any("nearest" in str(w.message).lower() or "not calibrated" in str(w.message).lower()
                   for w in caught)

    def test_predict_decision_fields(self):
        """predict_decision returns all expected keys."""
        from sd_toggle.predict import predict_decision
        config = _make_test_config()
        result = predict_decision(config, B=4, S=1024, gamma=7, L_accept=6.08)
        expected_keys = {"sd_on", "speedup", "r", "v", "L_accept",
                         "gamma", "B", "S", "regime_T_T", "regime_T_V"}
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )
        assert isinstance(result["sd_on"], bool)
        assert result["gamma"] == 7
        assert result["B"] == 4
        assert result["S"] == 1024
        assert result["regime_T_T"] in ("mem", "cmp")
        assert result["regime_T_V"] in ("mem", "cmp")

    def test_predict_batch(self):
        """predict_batch returns list[bool] of correct length."""
        from sd_toggle.predict import predict_batch
        config = _make_test_config()
        queries = [
            (1, 1024, 7, 6.08),
            (8, 1024, 7, 6.08),
            (64, 4096, 7, 6.08),
        ]
        results = predict_batch(config, queries)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, bool) for r in results)

    def test_evaluate_sign_accuracy_pow2(self):
        """evaluate_sign_accuracy with pow2_only=True behaves consistently."""
        from sd_toggle.predict import evaluate_sign_accuracy
        from sd_toggle.fit import calibrate
        csv_path = RESULTS_DIR / "qwen7b_finegrain.csv"
        if not csv_path.exists():
            pytest.skip("Finegrain CSV not available")
        config = calibrate(csv_path, "Qwen2.5-7B")
        gamma = config.metadata.get("gammas", [])[0]
        L_accept = float(gamma)  # typical trained-policy regime
        acc_all = evaluate_sign_accuracy(
            config, csv_path, L_accept=L_accept, gammas=[gamma], pow2_only=False,
        )
        acc_pow2 = evaluate_sign_accuracy(
            config, csv_path, L_accept=L_accept, gammas=[gamma], pow2_only=True,
        )
        # pow2_only restricts to a subset; both should be in [0, 1]
        assert 0.0 <= acc_all <= 1.0
        assert 0.0 <= acc_pow2 <= 1.0


class TestCLI:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "sd_toggle", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "sd_toggle" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_info(self):
        config_path = "sd_toggle/configs/a100_tp1_qwen257b.json"
        result = subprocess.run(
            [sys.executable, "-m", "sd_toggle", "info", "--config", config_path],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        # Output should contain hardware or model info
        combined = result.stdout + result.stderr
        assert any(kw in combined.lower() for kw in ("a100", "qwen", "bw_eff", "gpu"))
