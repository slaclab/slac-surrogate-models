"""Model loading utilities for the FACET-II beam-evolution surrogate model."""

import sys
from pathlib import Path
from lume_torch.models import TorchModel, TorchModule

from models.facet2_inj_twiss import lume_model_utils
from models.facet2_inj_twiss import training as _training_module

# Register module aliases so torch.load can unpickle classes serialized as
# lume_model_utils.PVToSimWithS / lume_model_utils.OutputDenormTransform
# and train.BeamEvolutionModel.
sys.modules.setdefault("lume_model_utils", lume_model_utils)
sys.modules.setdefault("infer_beam_evolution", lume_model_utils)
sys.modules.setdefault("train", _training_module)


_MODEL_CONFIGS = {
    "machine": "lumetorchyaml-machine/beam_evolution_machine.yaml",
    "sim": "lumetorchyaml-sim/beam_evolution_simulator.yaml",
}


def get_resource_path(filename):
    """Get the absolute path to a resource file."""
    package_dir = Path(__file__).parent
    resource_path = package_dir / "resources" / filename

    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")

    return str(resource_path)


def load_model(input_space="machine"):
    """
    Load the FACET-II beam-evolution surrogate model.

    Parameters
    ----------
    input_space : str, optional
        Which input space the model expects. ``"machine"`` (default) accepts
        machine PV values + s; ``"sim"`` accepts simulator parameters + s.

    Returns
    -------
    TorchModel
        Loaded model instance ready for inference.

    Example
    -------
        >>> from models.facet2_inj_twiss import load_model
        >>> model = load_model()             # machine PVs + s
        >>> model = load_model("sim")        # sim parameters + s

    """
    if input_space not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown input_space {input_space!r}; expected one of {list(_MODEL_CONFIGS)}"
        )
    config_path = get_resource_path(_MODEL_CONFIGS[input_space])
    return TorchModel(config_path)
