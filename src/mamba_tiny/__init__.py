"""Vendored from mamba-tiny (https://github.com/PeaBrane/mamba-tiny, Apache 2.0 -
see LICENSE in this directory), itself derived from mamba-minimal
(https://github.com/johnma2006/mamba-minimal).

Local modifications made for this project (mamba-recall-scaling-laws): multi-head ``head_pattern`` support
(MQA/MKA/MVA/MHA), the simplified linear variant (``skip_non_linearities`` /
``skip_gating`` config flags), analytic MQAR weight initialization
(``weights.py``), and recorder hooks for mechanistic interpretability.
"""
