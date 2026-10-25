"""Experiment stages A-F (docs/DESIGN.md section 8).

Every stage module exposes

    run(cfg: lsne.config.RunConfig, mb: lsne.model.ModelBundle, run_dir: lsne.io.RunDir, **kw)
        -> dict[str, pandas.DataFrame]

and saves every returned DataFrame through :func:`lsne.io.save_df` plus a ``manifest.json``.
Optional keyword arguments shared by all stages (see :mod:`lsne.experiments.common`):

* ``ctx``     - a mutable dict used to share expensive objects (variants, the original
                State-2143 variants, the direction bank) between stages of one CLI run;
* ``limits``  - a :class:`~lsne.experiments.common.Limits` instance (smoke / limited runs);
* ``smoke``   - shortcut for ``limits=Limits.smoke()``.

``STAGES`` maps the stage letter to its module; ``STAGE_ORDER`` is the order used by
``python -m lsne.cli run --stage all``.
"""
from __future__ import annotations

from . import (common, stage_a_stimuli, stage_b_baseline, stage_c_grid, stage_d_noneq,
               stage_e_tn, stage_f_scale)

STAGES = {
    "A": stage_a_stimuli,
    "B": stage_b_baseline,
    "C": stage_c_grid,
    "D": stage_d_noneq,
    "E": stage_e_tn,
    "F": stage_f_scale,
}
STAGE_ORDER = ("A", "B", "C", "D", "E", "F")
STAGE_TITLES = {
    "A": "stimuli",
    "B": "baseline and directions",
    "C": "switching grid and counterfactual attribution",
    "D": "non-equilibrium diagnostics",
    "E": "tensor-network linear response",
    "F": "scale and checkpoint sweep",
}

__all__ = ["STAGES", "STAGE_ORDER", "STAGE_TITLES", "common", "stage_a_stimuli", "stage_b_baseline",
           "stage_c_grid", "stage_d_noneq", "stage_e_tn", "stage_f_scale"]
