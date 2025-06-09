.. _reference:

API reference
=============

The API reference provides an overview of all public functions in ``gwlearn``.

Regression modelling
--------------------

Linear models:

.. currentmodule:: gwlearn.linear_model
.. autosummary::
   :toctree: generated/

   GWLinearRegression

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

Base classes
------------

.. currentmodule:: gwlearn.base
.. autosummary::
   :toctree: generated/

   BaseClassifier
   BaseRegressor