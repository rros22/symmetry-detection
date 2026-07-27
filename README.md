# symmetry-detection

Data-driven detection of point symmetries in ordinary differential equations (ODEs).

Given trajectory data sampled from an ODE `du/dx = f(x, u)`, this package constructs a
homogeneous linear system from a chosen basis-function family (monomial or Chebyshev) and
recovers the infinitesimal generator of the equation's point symmetry (if one exists) via
the null space of that system.

## Installation

```bash
git clone https://github.com/rros22/symmetry_detection.git
cd symmetry_detection
pip install -e ".[dev]"
```

## Usage

The package ships with a handful of built-in demo ODEs (`bernoulli`, `rational`, `riccati`,
`scaling`, `abel`) with known point symmetries, useful for trying out the method without
writing your own ODE first.

Generate a synthetic trajectory dataset:

```bash
symdet-generate --ode bernoulli
```

Run symmetry detection on a built-in demo ODE:

```bash
symdet-detect --ode bernoulli --basis_type monomial --param_range 0 3
```

For visualizations of the generated manifold and the detected symmetry field, see the
scripts in [`examples/`](examples/):

```bash
python examples/plot_ode_manifold.py --ode bernoulli
python examples/run_odes_demo.py --ode bernoulli
```

For an interactive version where you can tune the basis type/elements and re-run just
the detection plots without regenerating trajectories, see
[`notebooks/ode_symmetry_explorer.ipynb`](notebooks/ode_symmetry_explorer.ipynb):

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/ode_symmetry_explorer.ipynb
```

## Project layout

```
src/symmetry_detection/   Installable package: ODE catalog, basis functions, detection engine
examples/                 Runnable demo/plotting scripts (not part of the installed package)
notebooks/                Interactive Jupyter notebooks built on the same package API
tests/                    Automated tests (pytest)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
