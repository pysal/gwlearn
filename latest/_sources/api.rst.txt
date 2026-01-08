.. _reference:

API reference
=============

The API reference provides an overview of all public functions in ``gwlearn``.

Base classes
------------

.. currentmodule:: gwlearn.base
.. autosummary::
   :toctree: generated/

   BaseClassifier
   BaseRegressor

Regression modelling
--------------------

Linear models:

.. currentmodule:: gwlearn.linear_model
.. autosummary::
   :toctree: generated/

   GWLinearRegression

Ensembles:

.. currentmodule:: gwlearn.ensemble
.. autosummary::
   :toctree: generated/

   GWRandomForestRegressor
   GWGradientBoostingRegressor


Classification modelling
------------------------

Linear models:

.. currentmodule:: gwlearn.linear_model
.. autosummary::
   :toctree: generated/

   GWLogisticRegression

Ensembles:

.. currentmodule:: gwlearn.ensemble
.. autosummary::
   :toctree: generated/

   GWRandomForestClassifier
   GWGradientBoostingClassifier

Bandwidth search
----------------

.. currentmodule:: gwlearn.search
.. autosummary::
   :toctree: generated/

   BandwidthSearch