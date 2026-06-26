"""Benchmark / regression tests for the FACET-II beam-evolution model.

These tests ensure the model continues to produce the same outputs over time.
Run with:
    pytest tests/test_facet2_inj_twiss.py -v
"""

import numpy as np
import pytest
import torch

from models.facet2_inj_twiss import load_model
from models.facet2_inj_twiss.pv_mapping import (
    machine_input_names,
    machine_to_sim_array,
    sim_to_machine_array,
)

# ---------------------------------------------------------------------------
# Reference input (simulator-parameter space) — from dataset-test.csv row 1000
# ---------------------------------------------------------------------------
EXAMPLE_SIM_INPUT = {
    "CQ10121:b1_gradient": -0.010443165898323059,
    "GUNF:rf_field_scale": 50737172.0,
    "GUNF:theta0_deg": -78.0258560180664,
    "L0AF_phase:theta0_deg": -1.976251244544983,
    "L0AF_scale:rf_field_scale": 50867140.0,
    "L0BF_phase:theta0_deg": 2.7504806518554688,
    "L0BF_scale:rf_field_scale": 61794876.0,
    "QA10361": 3.399810552597046,
    "QA10371": -3.8894050121307373,
    "QE10425": 4.975119113922119,
    "QE10441": -6.094447612762451,
    "QE10511": 2.5492796897888184,
    "QE10525": -4.390674591064453,
    "SOL10111:solenoid_field_scale": 0.25849562883377075,
    "SQ10122:b1_gradient": -0.011345886625349522,
    "distgen:VCC": 611.0,
    "distgen:t_dist:sigma_t:value": 0.999123752117157,
    "distgen:total_charge:value": 947.1427001953125,
    "impact_VCC_Cal": 5.353600045054918e-06,
    "s": 3.5409998893737793,
}

EXAMPLE_MACHINE_INPUT = {
    "QUAD:IN10:121:BCTRL": 0.021930648013949394,
    "KLYS:LI10:21:AMPL": 40.073787689208984,
    "KLYS:LI10:21:PHAS": 74.2741470336914,
    "KLYS:LI10:31:PHAS": 23.52374839782715,
    "L0AF_scale:rf_field_scale": -11512872.0,
    "KLYS:LI10:41:PHAS": 140.25048828125,
    "L0BF_scale:rf_field_scale": 1908768.0,
    "QUAD:IN10:361:BCTRL": -3.671795606613159,
    "QUAD:IN10:371:BCTRL": 4.200557708740234,
    "QUAD:IN10:425:BCTRL": -5.373128890991211,
    "QUAD:IN10:441:BCTRL": 6.582003593444824,
    "QUAD:IN10:511:BCTRL": -2.7532222270965576,
    "QUAD:IN10:525:BCTRL": 4.741928577423096,
    "SOLN:IN10:121:BCTRL": 0.41359302401542664,
    "QUAD:IN10:122:BCTRL": 0.02382636070251465,
    "distgen:VCC": 611.0,
    "distgen:t_dist:sigma_t:value": -0.17087620496749878,
    "TORO:IN10:591:TMIT_PC": 947.1427001953125,
    "impact_VCC_Cal": -1.6663998394506052e-06,
    "s": 3.5409998893737793,
}

# Reference output from the verified model (model-output-100h)
EXPECTED_OUTPUTS = {
    "sigma_x": 0.000657239,
    "sigma_y": 0.00060078769,
    "sigma_z": 0.000508948,
    "norm_emit_x": 4.716183e-06,
    "norm_emit_y": 4.1137992e-06,
    "emit_geom_x": 3.1203967e-08,
    "emit_geom_y": 2.0755238e-08,
    "beta_x": 15.036438,
    "alpha_x": -1.6871569,
    "beta_y": 14.322559,
    "alpha_y": -1.4267795,
    "mean_kinetic_energy": 86220616,
}

# Sim-space input variable names (19 control params + s)
SIM_INPUT_VARIABLES = [
    "CQ10121:b1_gradient",
    "GUNF:rf_field_scale",
    "GUNF:theta0_deg",
    "L0AF_phase:theta0_deg",
    "L0AF_scale:rf_field_scale",
    "L0BF_phase:theta0_deg",
    "L0BF_scale:rf_field_scale",
    "QA10361",
    "QA10371",
    "QE10425",
    "QE10441",
    "QE10511",
    "QE10525",
    "SOL10111:solenoid_field_scale",
    "SQ10122:b1_gradient",
    "distgen:VCC",
    "distgen:t_dist:sigma_t:value",
    "distgen:total_charge:value",
    "impact_VCC_Cal",
    "s",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def machine_model():
    """Load the machine-PV model once for the entire test module."""
    return load_model("machine")


@pytest.fixture(scope="module")
def sim_model():
    """Load the sim model once for the entire test module."""
    return load_model("sim")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    """Verify that both model variants load without errors."""

    def test_load_machine_model(self, machine_model):
        assert machine_model is not None

    def test_load_sim_model(self, sim_model):
        assert sim_model is not None

    def test_machine_model_has_output_variables(self, machine_model):
        var_names = [v.name for v in machine_model.output_variables]
        for target in EXPECTED_OUTPUTS:
            assert target in var_names

    def test_sim_model_has_output_variables(self, sim_model):
        var_names = [v.name for v in sim_model.output_variables]
        for target in EXPECTED_OUTPUTS:
            assert target in var_names

    def test_machine_model_has_20_inputs(self, machine_model):
        assert len(machine_model.input_names) == 20

    def test_sim_model_has_20_inputs(self, sim_model):
        assert len(sim_model.input_names) == 20


class TestEvaluate:
    """Verify model.evaluate() produces expected outputs."""

    def test_sim_evaluate_returns_all_targets(self, sim_model):
        result = sim_model.evaluate(EXAMPLE_SIM_INPUT)
        for target in EXPECTED_OUTPUTS:
            assert target in result

    def test_sim_output_matches_reference(self, sim_model):
        result = sim_model.evaluate(EXAMPLE_SIM_INPUT)
        for target, expected in EXPECTED_OUTPUTS.items():
            actual = result[target]
            if isinstance(actual, torch.Tensor):
                actual = actual.item()
            np.testing.assert_allclose(
                actual, expected, rtol=1e-3,
                err_msg=f"Mismatch for {target}",
            )

    def test_machine_evaluate_returns_all_targets(self, machine_model):
        result = machine_model.evaluate(EXAMPLE_MACHINE_INPUT)
        for target in EXPECTED_OUTPUTS:
            assert target in result

    def test_machine_output_reasonable(self, machine_model):
        """Machine model should produce outputs in the same ballpark.

        Due to float32 roundtrip through PV→sim conversion, we allow
        slightly looser tolerance than the sim model.
        """
        result = machine_model.evaluate(EXAMPLE_MACHINE_INPUT)
        for target, expected in EXPECTED_OUTPUTS.items():
            actual = result[target]
            if isinstance(actual, torch.Tensor):
                actual = actual.item()
            np.testing.assert_allclose(
                actual, expected, rtol=5e-2,
                err_msg=f"Mismatch for {target}",
            )


class TestPVMapping:
    """Verify PV ↔ sim parameter conversions are consistent."""

    def test_roundtrip_sim_to_pv_to_sim(self):
        sim_cols = SIM_INPUT_VARIABLES[:-1]  # exclude 's'
        sim_vals = np.array(
            [EXAMPLE_SIM_INPUT[c] for c in sim_cols], dtype=np.float32
        ).reshape(1, -1)
        pv_vals = sim_to_machine_array(sim_vals, sim_cols)
        sim_roundtrip = machine_to_sim_array(pv_vals, sim_cols)
        np.testing.assert_allclose(sim_vals, sim_roundtrip, rtol=1e-5)

    def test_pv_names_match_expected(self):
        sim_cols = SIM_INPUT_VARIABLES[:-1]  # exclude 's'
        pv_names = machine_input_names(sim_cols)
        assert len(pv_names) == 19
        # Check a few known mappings
        assert "QUAD:IN10:121:BCTRL" in pv_names
        assert "SOLN:IN10:121:BCTRL" in pv_names
        assert "KLYS:LI10:21:AMPL" in pv_names
