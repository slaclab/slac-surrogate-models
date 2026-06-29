"""FACET-II Injector (241) Model Package."""

import sys
from importlib import import_module

import models.facet2_inj_241.transforms as _transforms
from models.facet2_inj_241.loader import load_model

_LEGACY_TRAIN_EXPORTS = {
	"CovarianceAwareLoss",
	"CovarianceSurrogateModel",
	"build_model",
	"chol_vectors_to_covariance",
}


def __getattr__(name):
	if name in _LEGACY_TRAIN_EXPORTS:
		training_module = import_module("models.facet2_inj_241.training")
		return getattr(training_module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register legacy module aliases so torch.load can unpickle classes serialized
# under the old single-repo package name.
sys.modules.setdefault("facet2_inj_ml_model", sys.modules[__name__])
sys.modules.setdefault("facet2_inj_ml_model.transforms", _transforms)

__version__ = "0.1.0"
__all__ = ["load_model", "BeamOutputModel", "pv_mapping", *_LEGACY_TRAIN_EXPORTS]
