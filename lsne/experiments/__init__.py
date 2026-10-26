"""Experiment stages A-L (docs/DESIGN.md section 8, docs/STAGES.md).

Every stage module exposes

    run(cfg: lsne.config.RunConfig, mb: lsne.model.ModelBundle, run_dir: lsne.io.RunDir, **kw)
        -> dict[str, pandas.DataFrame]

and saves every returned DataFrame through :func:`lsne.io.save_df` plus a ``manifest.json``.
Optional keyword arguments shared by all stages (see :mod:`lsne.experiments.common`):

* ``ctx``     - a mutable dict used to share expensive objects (variants, the original
                State-2143 variants, the direction bank, stage G's realization clouds,
                stage I's domain axes and worst head) between stages of one CLI run;
* ``limits``  - a :class:`~lsne.experiments.common.Limits` instance (smoke / limited runs);
* ``smoke``   - shortcut for ``limits=Limits.smoke()``.

Stages G-L record failures of their substages in the manifest (``substages``) and in
``substage_errors.txt`` instead of aborting the stage.

``STAGES`` maps the stage letter to its module; ``STAGE_ORDER`` is the order used by
``python -m lsne.cli run --stage all``.
"""
from __future__ import annotations

from . import (common, stage_a_stimuli, stage_b_baseline, stage_c_grid, stage_d_noneq,
               stage_e_tn, stage_f_scale, stage_g_geometry, stage_h_lambda, stage_i_compass,
               stage_j_rdx, stage_k_ttt, stage_l_transplant)

STAGES = {
    "A": stage_a_stimuli,
    "B": stage_b_baseline,
    "C": stage_c_grid,
    "D": stage_d_noneq,
    "E": stage_e_tn,
    "F": stage_f_scale,
    "G": stage_g_geometry,
    "H": stage_h_lambda,
    "I": stage_i_compass,
    "J": stage_j_rdx,
    "K": stage_k_ttt,
    "L": stage_l_transplant,
}
STAGE_ORDER = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
STAGE_TITLES = {
    "A": "stimuli",
    "B": "baseline and directions",
    "C": "switching grid and counterfactual attribution",
    "D": "non-equilibrium diagnostics",
    "E": "tensor-network linear response",
    "F": "scale and checkpoint sweep",
    "G": "local geometry and trajectory",
    "H": "lambda dynamics",
    "I": "domain compass and heads",
    "J": "RDX, Pile difficulty and sharing",
    "K": "test-time training",
    "L": "cross-model transplant",
}

__all__ = ["STAGES", "STAGE_ORDER", "STAGE_TITLES", "common", "stage_a_stimuli", "stage_b_baseline",
           "stage_c_grid", "stage_d_noneq", "stage_e_tn", "stage_f_scale", "stage_g_geometry", "stage_h_lambda",
           "stage_i_compass", "stage_j_rdx", "stage_k_ttt", "stage_l_transplant"]
