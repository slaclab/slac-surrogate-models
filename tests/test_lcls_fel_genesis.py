"""Benchmark / regression tests for the LCLS FEL Genesis model.

These tests ensure the model continues to produce the same outputs over time.
Run with:
    pytest tests/test_lcls_fel_genesis.py -v
"""

import numpy as np
import pytest
import torch

from models.lcls_fel_genesis import load_model

INPUT_VARIABLES = [
    "alphax",
    "alphay",
    "betax",
    "betay",
    "emitx",
    "emity",
    "chirp",
    "taper",
    "ustart",
    "beam_length",
]

OUTPUT_VARIABLES = ["energy", "bandwidth", "peak_intensity", "peak_energy"]

# ---------------------------------------------------------------------------
# Reference input — row 0 of the upstream test set (info/test_model_inputs2.pt)
# ---------------------------------------------------------------------------
EXAMPLE_INPUT = {
    "alphax": -1.013674020767212,
    "alphay": 1.1685378551483154,
    "betax": 11.200857162475586,
    "betay": 12.651693344116211,
    "emitx": 2.0490000451900414e-07,
    "emity": 2.5599999275982555e-07,
    "chirp": -1.5900346040725708,
    "taper": 0.0174599327147007,
    "ustart": 11.0,
    "beam_length": 1.7761200069799088e-05,
}

# Reference output from the verified model.
# NOTE: these are *model* predictions, not Genesis ground truth. The model
# tracks Genesis closely for peak_energy but only loosely for bandwidth and
# peak_intensity, so pinning ground truth would not be a useful regression
# check. The corresponding Genesis values for this point are
# energy=7283.5, bandwidth=39.1, peak_intensity=3414.5, peak_energy=8982.1.
EXPECTED_OUTPUTS = {
    "energy": 7335.614044581565,
    "bandwidth": 42.50653795535956,
    "peak_intensity": 3504.8991219001546,
    "peak_energy": 8982.551963594136,
}

# Predictions for the model's own default input values (all inputs omitted)
EXPECTED_DEFAULT_OUTPUTS = {
    "energy": 407.90961503558077,
    "bandwidth": 40.76443884716915,
    "peak_intensity": 71.28865888283549,
    "peak_energy": 8991.475360753991,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model():
    """Load the model once for the entire test module."""
    return load_model()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    """Verify that the model loads and exposes the expected variables."""

    def test_load_model(self, model):
        assert model is not None

    def test_input_names(self, model):
        assert model.input_names == INPUT_VARIABLES

    def test_output_names(self, model):
        assert model.output_names == OUTPUT_VARIABLES

    def test_output_units(self, model):
        units = {v.name: v.unit for v in model.output_variables}
        assert units == {
            "energy": "uJ",
            "bandwidth": "eV",
            "peak_intensity": "au",
            "peak_energy": "eV",
        }

    def test_input_value_ranges(self, model):
        ranges = {v.name: tuple(v.value_range) for v in model.input_variables}
        assert ranges["betax"] == (5.0, 30.0)
        assert ranges["chirp"] == (-20.0, 0.0)
        assert ranges["taper"] == (0.0, 0.2)
        assert ranges["ustart"] == (10.0, 16.0)
        assert ranges["beam_length"] == (1.5e-05, 1.8e-05)


class TestEvaluate:
    """Verify model.evaluate() produces expected outputs."""

    def test_evaluate_returns_all_targets(self, model):
        result = model.evaluate(EXAMPLE_INPUT)
        for target in EXPECTED_OUTPUTS:
            assert target in result

    def test_output_matches_reference(self, model):
        result = model.evaluate(EXAMPLE_INPUT)
        for target, expected in EXPECTED_OUTPUTS.items():
            actual = result[target]
            if isinstance(actual, torch.Tensor):
                actual = actual.item()
            np.testing.assert_allclose(
                actual, expected, rtol=1e-4,
                err_msg=f"Mismatch for {target}",
            )

    def test_defaults_match_reference(self, model):
        """Omitted inputs should fall back to their configured defaults."""
        result = model.evaluate({})
        for target, expected in EXPECTED_DEFAULT_OUTPUTS.items():
            actual = result[target]
            if isinstance(actual, torch.Tensor):
                actual = actual.item()
            np.testing.assert_allclose(
                actual, expected, rtol=1e-4,
                err_msg=f"Mismatch for {target}",
            )

    def test_batched_evaluate(self, model):
        """A batch of N points should give N results per output."""
        batch = {
            name: torch.tensor([value, value], dtype=torch.double)
            for name, value in EXAMPLE_INPUT.items()
        }
        result = model.evaluate(batch)
        for target, expected in EXPECTED_OUTPUTS.items():
            actual = result[target].squeeze()
            assert actual.shape == (2,)
            np.testing.assert_allclose(
                actual.numpy(), [expected, expected], rtol=1e-4,
                err_msg=f"Mismatch for {target}",
            )
