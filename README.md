# SLAC Surrogate Models

Monorepo of SLAC accelerator surrogate models built with [LUME-Torch](https://github.com/lume-science/lume-torch). Each model is packaged as an optional extra so you can install only the models you need.

## Available Models

| Extra name | Model | Inputs | Outputs | Description |
|---|---|---|---|---|
| `lcls_cu_inj` | LCLS Cu Injector | Machine PVs | Beam properties at OTR2 | Trained on IMPACT-T data for the LCLS Cu injector |
| `lcls_fel` | LCLS FEL | Machine PVs | FEL pulse intensity | ML surrogate of the LCLS FEL |
| `facet2_inj_241` | FACET-II Injector (241) | 7 PVs / sim params | 6×6 covariance matrix | Beam covariance at screen 241 |
| `facet2_inj_571` | FACET-II Injector (571) | 19 PVs / sim params | 6×6 covariance matrix + optional means | Beam covariance (+ phase-space means) at screen 571 |
| `facet2_inj_twiss` | FACET-II Twiss | 19 PVs / sim params + s | 12 beam-evolution targets | Twiss parameters & beam sizes vs beamline position |

## Installation

Requires Python ≥ 3.11.

### From a local clone

```bash
git clone <repo-url>
cd slac-surrogate-models

# Install with all models
pip install -e ".[all]"

# Install specific models only
pip install -e ".[facet2_inj_241,facet2_inj_571]"

# Install a single model
pip install -e ".[lcls_cu_inj]"

# Install with dev/test dependencies
pip install -e ".[all,dev]"
```

### From a remote repository (once published)

```bash
pip install "slac-surrogate-models[all]"
pip install "slac-surrogate-models[lcls_cu_inj]"
```

The base package always installs `lume-torch`, `torch`, and `numpy`. Model-specific extras pull in additional dependencies only when needed:

| Extra | Additional dependencies |
|---|---|
| `lcls_cu_inj` | — |
| `lcls_fel` | — |
| `facet2_inj_241` | `distgen`, `pandas`, `scipy` |
| `facet2_inj_571` | `distgen`, `pandas`, `scipy` |
| `facet2_inj_twiss` | `pandas` |
| `test` | `pytest`, `openPMD-beamphysics`, `botorch`, `scipy` |
| `dev` | all models + test deps + `black` |

## Quick Start

Every model exposes a `load_model()` function that returns a ready-to-use `lume_torch.models.TorchModel`:

```python
from models.lcls_cu_inj import load_model
model = load_model()
result = model.evaluate({"QUAD:IN20:425:BCTRL": -1})
```

```python
from models.lcls_fel import load_model
model = load_model()
result = model.evaluate({"ACCL:LI25:1:ADES": 6260.0})
```

When not specified, input variables use their default values as defined in each model's `model_config.yaml`.

## Model Details

### LCLS Cu Injector (`lcls_cu_inj`)

```python
from models.lcls_cu_inj import load_model
model = load_model()
```

Trained on IMPACT-T simulation data to predict beam properties at OTR2 using injector PVs.

### LCLS FEL (`lcls_fel`)

```python
from models.lcls_fel import load_model
model = load_model()
```

ML surrogate of the LCLS FEL pulse intensity.

### FACET-II Injector at Screen 241 (`facet2_inj_241`)

```python
from models.facet2_inj_241 import load_model

model = load_model()           # machine-PV inputs (7 inputs)
model = load_model("sim")      # simulator-parameter inputs
```

Predicts a 6×6 beam covariance matrix from 7 inputs (Cholesky-factor architecture). Includes a `BeamOutputModel` wrapper for generating openPMD particle distributions:

```python
from models.facet2_inj_241.beam_output_model import BeamOutputModel

beam_model = BeamOutputModel(load_model(), n_particles=10000)
beam_model.set({"QUAD:IN10:121:BCTRL": 0.032, ...})
particles = beam_model.final_particles  # openPMD ParticleGroup
```

**Machine-PV inputs:** `QUAD:IN10:121:BCTRL`, `KLYS:LI10:21:AMPL`, `KLYS:LI10:21:PHAS`, `SOLN:IN10:121:BCTRL`, `QUAD:IN10:122:BCTRL`, `distgen:t_dist:sigma_t:value`, `TORO:IN10:591:TMIT_PC`

### FACET-II Injector at Screen 571 (`facet2_inj_571`)

```python
from models.facet2_inj_571 import load_model

model = load_model()                        # cov-only, machine PVs (19 inputs)
model = load_model("sim")                   # cov-only, sim parameters
model_full = load_model(full=True)          # cov + 6 phase-space means
model_full = load_model("sim", full=True)   # sim inputs, full output
```

Predicts beam covariance at screen 571 from 19 inputs. The `full=True` variant also predicts 6 phase-space mean values (mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz). Also supports `BeamOutputModel` for particle generation.

### FACET-II Beam Evolution / Twiss Parameters (`facet2_inj_twiss`)

```python
from models.facet2_inj_twiss import load_model

model = load_model()           # machine-PV inputs (19 PVs + s)
model = load_model("sim")      # simulator-parameter inputs (19 params + s)
```

Predicts 12 beam-evolution targets as a continuous function of beamline position `s` (cathode to L0AFEND, s ≈ 0.001–4.13 m):

| Output | Description |
|---|---|
| `sigma_x`, `sigma_y`, `sigma_z` | Beam sizes (m) |
| `norm_emit_x`, `norm_emit_y` | Normalized emittances (m·rad) |
| `emit_geom_x`, `emit_geom_y` | Geometric emittances (m·rad) |
| `beta_x`, `beta_y` | Twiss beta (m) |
| `alpha_x`, `alpha_y` | Twiss alpha |
| `mean_kinetic_energy` | Mean kinetic energy (eV) |

## Running Tests

```bash
pip install -e ".[all,test]"
pytest tests/ -v
```
