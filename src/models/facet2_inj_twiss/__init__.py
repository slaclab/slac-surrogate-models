"""FACET-II Injector Beam-Evolution Model Package (Twiss Parameters)."""

from importlib import import_module

from models.facet2_inj_twiss.loader import load_model

_LEGACY_TRAIN_EXPORTS = {
    "BeamEvolutionModel",
    "build_model",
}


def __getattr__(name):
    if name in _LEGACY_TRAIN_EXPORTS:
        training_module = import_module("models.facet2_inj_twiss.training")
        return getattr(training_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__all__ = ["load_model", "pv_mapping", *_LEGACY_TRAIN_EXPORTS]
