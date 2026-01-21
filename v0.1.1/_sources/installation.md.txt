# Installation

You can install gwlearn from PyPI or from conda-forge using the tool of your choice:

```sh
pip install gwlearn
```

Or from conda-forge:

```sh
conda install gwlearn -c conda-forge
```

## Installing development version

You can either clone the repository:

```sh
git clone https://github.com/pysal/gwlearn.git
cd gwlearn
pip install .
```

Or install directly from Github:

```sh
pip install git+https://github.com/pysal/gwlearn.git
```

The package depends on:

```yaml
geopandas>=1.0.0
joblib>=1.4.0
libpysal>=4.12
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
pandas>=2.1.0
```