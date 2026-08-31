"""Model loading utilities"""

from pathlib import Path
from lume_torch.models import TorchModel


def get_resource_path(filename):
    """Get the absolute path to a resource file."""
    package_dir = Path(__file__).parent
    resource_path = package_dir / "resources" / filename

    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")

    return str(resource_path)


def load_model():
    """
    Load the LCLS FEL Genesis TorchModel.

    Predicts Genesis FEL output quantities (energy, bandwidth, peak
    intensity, peak photon energy) from electron-beam Twiss parameters,
    chirp, undulator taper and start, and beam length.

    Returns:
        TorchModel: Loaded model instance ready for inference.

    Example:
        >>> from models.lcls_fel_genesis import load_model
        >>> model = load_model()
        >>> result = model.evaluate({"taper": 0.05})
    """
    config_path = get_resource_path("model_config.yaml")

    # Load the model using the config
    model = TorchModel(config_path)

    return model
