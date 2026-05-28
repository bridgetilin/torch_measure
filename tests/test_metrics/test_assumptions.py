# Copyright (c) 2026 AIMS Foundations. MIT License.

import numpy as np
import pandas as pd
import pytest

from torch_measure.metrics.assumptions import normality_check, tukey_nonadditivity_test


def _additive_design(
    n_p: int = 30,
    n_i: int = 10,
    n_r: int = 2,
    sigma_p: float = 1.0,
    sigma_i: float = 0.7,
    sigma_e: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Strictly additive Y = a_p + b_i + e_pir (no interaction)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma_p, size=n_p)
    b = rng.normal(0.0, sigma_i, size=n_i)
    e = rng.normal(0.0, sigma_e, size=(n_p, n_i, n_r))
    y = a[:, None, None] + b[None, :, None] + e
    rows = [(f"s{p}", f"i{i}", r, float(y[p, i, r])) for p in range(n_p) for i in range(n_i) for r in range(n_r)]
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"])


def _multiplicative_design(
    n_p: int = 40,
    n_i: int = 12,
    n_r: int = 2,
    sigma_e: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Strongly non-additive Y_pi = a_p * b_i + small noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=n_p)
    b = rng.normal(0.0, 1.0, size=n_i)
    e = rng.normal(0.0, sigma_e, size=(n_p, n_i, n_r))
    y = (a[:, None] * b[None, :])[:, :, None] + e
    rows = [(f"s{p}", f"i{i}", r, float(y[p, i, r])) for p in range(n_p) for i in range(n_i) for r in range(n_r)]
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"])


def _skewed_residuals_design(seed: int = 0) -> pd.DataFrame:
    """Additive structure but log-normal (heavily right-skewed) within-cell noise."""
    rng = np.random.default_rng(seed)
    n_p, n_i, n_r = 40, 12, 4
    a = rng.normal(0.0, 1.0, size=n_p)
    b = rng.normal(0.0, 0.7, size=n_i)
    e = rng.lognormal(mean=0.0, sigma=1.0, size=(n_p, n_i, n_r))
    y = a[:, None, None] + b[None, :, None] + e
    rows = [(f"s{p}", f"i{i}", r, float(y[p, i, r])) for p in range(n_p) for i in range(n_i) for r in range(n_r)]
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"])


class TestNormalityCheck:
    def test_output_structure(self):
        out = normality_check(_additive_design())
        for k in ("test", "target", "statistic", "p_value", "n", "qq_theoretical", "qq_sample"):
            assert k in out
        assert out["test"] == "shapiro_wilk"
        assert out["qq_theoretical"].shape == out["qq_sample"].shape
        assert np.all(np.diff(out["qq_sample"]) >= 0)  # sorted

    def test_normal_residuals_not_rejected(self):
        out = normality_check(_additive_design(seed=1))
        assert out["p_value"] > 0.01

    def test_skewed_residuals_rejected(self):
        out = normality_check(_skewed_residuals_design(seed=0))
        assert out["p_value"] < 0.01

    def test_target_subject(self):
        out = normality_check(_additive_design(n_p=40), target="subject")
        assert out["target"] == "subject"
        assert out["n"] == 40

    def test_target_item(self):
        out = normality_check(_additive_design(n_i=15), target="item")
        assert out["n"] == 15

    def test_target_subject_item_available_without_reps(self):
        # n_r=1 design — residual would error, but subject_item is fine.
        df = _additive_design(n_p=10, n_i=6, n_r=1)
        out = normality_check(df, target="subject_item")
        assert out["n"] == 10 * 6

    def test_target_residual_requires_reps(self):
        df = _additive_design(n_p=10, n_i=6, n_r=1)
        with pytest.raises(ValueError, match="requires replications"):
            normality_check(df, target="residual")

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            normality_check(_additive_design(), target="bogus")

    def test_missing_columns_raises(self):
        df = _additive_design().drop(columns=["item_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            normality_check(df)


class TestTukeyNonadditivity:
    def test_output_structure(self):
        out = tukey_nonadditivity_test(_additive_design())
        for k in ("test", "statistic", "p_value", "df_num", "df_den", "ss_nonadditivity", "ss_remainder"):
            assert k in out
        assert out["test"] == "tukey_nonadditivity"
        assert out["df_num"] == 1
        assert out["df_den"] >= 1

    def test_additive_not_rejected(self):
        out = tukey_nonadditivity_test(_additive_design(seed=2))
        assert out["p_value"] > 0.01

    def test_multiplicative_rejected(self):
        out = tukey_nonadditivity_test(_multiplicative_design(seed=0))
        assert out["p_value"] < 0.01

    def test_works_without_replications(self):
        # Single-rep design: classical use case where interaction is otherwise
        # confounded with residual.
        out = tukey_nonadditivity_test(_multiplicative_design(n_r=1, seed=3))
        assert out["p_value"] < 0.05

    def test_unbalanced_raises(self):
        df = _additive_design(n_p=8, n_i=5, n_r=2)
        df = df[~((df["subject_id"] == "s0") & (df["item_id"] == "i0"))]
        with pytest.raises(ValueError, match="Unbalanced design"):
            tukey_nonadditivity_test(df)
