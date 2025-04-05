import contextlib
from importlib.metadata import PackageNotFoundError, version

from .bandwidth_search import BandwidthSearch
from .base import BaseClassifier
from .ensemble import GWRandomForestClassifier, GWGradientBoostingClassifier
from .linear_model import GWLogisticRegression

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("gwlearn")
