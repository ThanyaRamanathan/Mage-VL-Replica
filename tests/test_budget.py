import numpy as np
import pytest

from surgical_tokens.budget import allocate_exact
from surgical_tokens.selectors import uniform_mask


def test_exact_budget_and_mandatory():
    scores = np.arange(24).reshape(2, 3, 4)
    mandatory = np.zeros_like(scores, dtype=bool)
    mandatory[0, 0] = True
    mask = allocate_exact(scores, 7, mandatory)
    assert mask.sum() == 7
    assert mask[0, 0].all()


def test_rejects_impossible_anchor_budget():
    with pytest.raises(ValueError, match="mandatory"):
        allocate_exact(np.zeros(8), 2, np.array([1, 1, 1, 0, 0, 0, 0, 0], bool))


def test_uniform_is_exact_and_deterministic():
    a = uniform_mask(12, 4, 5, 73)
    b = uniform_mask(12, 4, 5, 73)
    assert a.sum() == 73
    np.testing.assert_array_equal(a, b)
