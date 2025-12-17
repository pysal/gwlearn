# gwlearn

Geographically weighted modeling based on `scikit-learn`.

The aim of the package is to provide implementations of spatially-explicit modelling.

## Status

Current development status is early beta. API of the package can change without a
warning. Use with caution.

## Features

`gwlearn` provides a framework for prototyping geographically weighted extensions of
regression and classification models based on `scikit-learn` and `libpysal.graph` and a
subset of models implemented on top of this framework. For example, you can run
geographically weighted linear regression in a following manner.

```py
import geopandas as gpd
from geodatasets import get_path

from gwlearn.linear_model import GWLinearRegression


gdf = gpd.read_file(get_path('geoda.guerry'))

adaptive = GWLinearRegression(
    geometry=gdf.representative_point(),
    bandwidth=25,
    fixed=False,
    kernel='tricube'
)
adaptive.fit(
    gdf[['Crm_prp', 'Litercy', 'Donatns', 'Lottery']],
    gdf["Suicids"],
)
```

For details, see the [documentation](https://pysal.org/gwlearn).

## Installation

The package is currently not released, so you will need to install it from source.

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

## Bug reports

To search for or report bugs, please see the
[Github issue tracker](https://github.com/pysal/gwlearn/issues).

## Get in touch

If you have a question regarding `gwlearn`, feel free to open an issue or join a chat on
[Discord](https://discord.gg/he6Y8D2ap3).

## License

The package is licensed under BSD 3-Clause License (Copyright (c) 2025, Martin
Fleischmann & PySAL Developers)