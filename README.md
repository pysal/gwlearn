## GWPCA Prototype (Work in Progress)

This branch introduces a prototype implementation of **Geographically Weighted Principal Component Analysis (GWPCA)** for spatially varying dimensionality reduction.

### Features

* Local PCA using spatial kernel weighting
* Supports:
  * Fixed bandwidth
  * Adaptive (k-nearest neighbors) bandwidth
* Efficient neighbor search using `cKDTree`
* Compatible with `scikit-learn` style (`fit`, `transform`)

---

### Examples

Run the following to see GWPCA in action:

```bash
python examples/gwpca_demo.py
python examples/gwpca_visual_demo.py
```

* `gwpca_demo.py` → basic usage
* `gwpca_visual_demo.py` → shows spatial non-stationarity

---

### Tests

Unit tests are included:

```bash
pytest
```

All tests pass successfully, ensuring basic correctness and stability.

---

### Validation

The implementation includes experimental validation:

* Comparison with global PCA
* Spatial analysis using Moran’s I
* Monte Carlo simulation

(See `experiments/gwpca_validation.py`)

---

### Note

This is an initial prototype and subject to change. Feedback on API design, performance, and integration with `gwlearn` is welcome.
