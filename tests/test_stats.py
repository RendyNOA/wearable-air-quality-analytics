import math

import numpy as np

from src.aq_analytics.io import LOCAL_LABEL, SYNTHETIC_LABEL, figure_label
from src.aq_analytics.stats import block_bootstrap_diff_ci, cv_percent, descriptive, relative_difference


def test_descriptive_and_cv():
    s = descriptive([10, 10, 10, 10])
    assert s["n"] == 4 and s["mean"] == 10 and s["cv_pct"] == 0
    assert math.isnan(cv_percent([0, 0, 0]))


def test_empty_input_has_zero_count():
    # n must be 0 (not NaN, which is truthy) so callers can skip empty groups.
    s = descriptive([])
    assert s["n"] == 0 and math.isnan(s["mean"])


def test_relative_difference():
    assert relative_difference(12, 10) == 20
    assert math.isnan(relative_difference(12, 0))


def test_block_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 1, 200)
    b = rng.normal(12, 1, 200)
    lo, hi = block_bootstrap_diff_ci(a, b, block=5, n_boot=500)
    assert lo < 2 < hi and lo > 1


def test_figure_label():
    assert figure_label("data/demo") == SYNTHETIC_LABEL
    assert figure_label("data/raw") == LOCAL_LABEL
