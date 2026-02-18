<!-- GitHub Copilot Instructions for gwlearn -->

Purpose
-------
A short, practical guide to help AI coding agents be immediately productive in this repository.

Quick setup
-----------
- Python version: use the project's `pyproject.toml` (requires >=3.11).
- Virtualenv available at `pysal-env`. On Windows PowerShell run:
  - `.
pysal-env\Scripts\Activate.ps1`
- Run tests from repository root:
  - `pytest -q`
  - Run a single test file: `pytest gwlearn/tests/test_search.py -q`

What this project is (big picture)
----------------------------------
- `gwlearn` implements geographically-weighted ML models built on top of scikit-learn.
- Core pieces:
  - `gwlearn/base.py`: core `_BaseModel` logic, kernels, weight construction, batching, and model storage.
  - `gwlearn/linear_model.py`: model wrappers (GWLinearRegression, GWLogisticRegression, etc.).
  - `gwlearn/search.py`: `BandwidthSearch` utility that searches for optimal bandwidths.
  - `gwlearn/tests/`: test-suite and fixtures used as compact usage examples.

Key conventions & patterns (concrete)
-----------------------------------
- Geometry-first APIs: most public `.fit(...)` methods accept `geometry` as a `geopandas.GeoSeries` of Point geometries. Validate using `Base._validate_geometry`.
- Bandwidth semantics:
  - `fixed=True` -> `bandwidth` is a float distance (meters or data units).
  - `fixed=False` -> `bandwidth` is an integer neighbor count (adaptive KNN).
- Kernel functions are centralized in `gwlearn/base.py` via `_kernel_functions` (supported names: `bisquare`, `tricube`, `triangular`, `parabolic`, `cosine`, `boxcar`). Use those exact strings when passing `kernel=`.
- Model forwarding: many utilities accept a `model` class and arbitrary model kwargs; these are forwarded via `_model_kwargs` (see `BandwidthSearch` and `_BaseModel._fit_global_model`). Pass scikit-learn-compatible args (e.g., `random_state`, `max_iter`) through the search wrapper.
- `keep_models` supports three modes:
  - `False` (default): do not keep local models
  - `True`: keep models in memory
  - `Path`: save per-local-model joblib files to disk
- Parallelism/storage: uses `joblib.Parallel` and `dump/load` for temp model storage. Be careful with `n_jobs` and `temp_folder` when running on CI or limited environments.

Developer workflows
-------------------
- Run full test-suite: `pytest -q`.
- Run a focused test: `pytest gwlearn/tests/test_search.py::test_interval_search_basic -q`.
- Linting/formatting: project uses `ruff` (configured in `pyproject.toml`).
- Build docs: see `docs/Makefile` and `docs/make.bat` (Sphinx + notebooks under `docs/source`).

Integration points & dependencies
---------------------------------
- Relies heavily on:
  - `geopandas` for geometry and spatial indexes (`.sindex`).
  - `libpysal.graph` for adjacency/weight construction.
  - `scikit-learn` for estimator APIs and compatibility.
  - `joblib` for parallelism and persistence.
- Tests/fixtures reference `geodatasets` for sample data (see test fixtures in `gwlearn/tests/conftest.py`).

Testing notes (specifics discovered in tests)
-------------------------------------------
- Tests sometimes use very large numeric bandwidths (e.g., 100000) to avoid expensive neighbor searches in CI — do not assume those values are physically meaningful; they are shortcuts for fast tests.
- Many tests check that the wrapper forwards model kwargs correctly; use `search = BandwidthSearch(..., max_iter=200)` to reproduce that behavior.
- Verbosity behavior: `BandwidthSearch(verbose=True)` prints progress lines; tests capture stdout to assert messages.

Where to look for examples & behavior
-------------------------------------
- Example usage in `README.md` (quick snippet for `GWLinearRegression`).
- Real behavior patterns in `gwlearn/base.py` and `gwlearn/search.py` (weight construction, kernel use, scoring logic).
- Tests: `gwlearn/tests/test_search.py` shows search API usage and common parameter combinations.

Do not assume
------------
- Geometry is optional for some calls; when present it must be a `geopandas.GeoSeries` of Points.
- Bandwidth units are context-dependent — check `fixed` flag before changing code that treats `bandwidth` as numeric distance.

Next steps for the agent
------------------------
- When modifying estimation logic, run the test file `gwlearn/tests/test_search.py` and `gwlearn/tests/test_linear_model.py`.
- When adding new public API parameters, mirror handling in `_BaseModel.__init__` and ensure forwarding to `_model_kwargs`.

If anything here is unclear, tell me which area you want expanded (examples, tests, CLI commands).
