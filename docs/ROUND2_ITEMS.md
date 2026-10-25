# Round 2 work items (from the round-1 verification)

Each group lists the files its owner may edit. Items reference docs/DESIGN.md,
docs/PREREGISTRATION.md and docs/SPEC.md.

## G1 model + config
Files: lsne/model.py, lsne/io.py, pyproject.toml, tests/test_model.py, tests/test_io.py
1. entropy_gradient: return `float(H.detach())`.
2. Steering during generation needs a Soll/Ist audit (pre-registration §4: rel_err > 1e-4
   invalidates a run): on the first dosed call record intended = eps·RMS(h)·dir/RMS(dir) at
   the dosed positions vs written = h_new − h in fp32; expose as Injector.audit and return it
   from generate_steered under key `audit` with `valid = rel_err <= 1e-4`. Test fp32 (valid)
   and a bfloat16 copy of pythia-14m (rel_err reported).
3. pyproject.toml: drop `-q` from pytest addopts; add `pyarrow` to the `notebooks` extra.
4. io.save_df: log once when parquet is skipped because pyarrow is missing.

## G2 stimuli + lang_id + stage A
Files: lsne/stimuli.py, lsne/lang_id.py, lsne/experiments/stage_a_stimuli.py,
tests/test_stimuli.py, tests/test_lang_id.py
1. annotate: a whitespace-only token that precedes a byte-split word belongs to the
   segment of the next non-space character when that character starts a new segment
   (the space belongs to the following word); boundary and dist_to_last_switch follow.
   Keep class `ws` for whitespace tokens that are not a word's leading space. Add a test
   with a German byte-split word at a C1 boundary.
2. lang_id.switch_count is a lower bound (one-word segments that are names/N are not
   counted): document it; extend test_switch_count_tracks_K to C2/C3/seg with the measured
   tolerance; assert exact K for C1.
3. stage A families_table: fill target_id from the K=0 variant; smoke-level assertion.

## G3 generation + stats
Files: lsne/generation.py, lsne/stats.py, tests/test_generation.py, tests/test_stats.py
1. Ramp: include zero-dose plateaus at both ends (0, eps_max/n, …, eps_max, …, eps_max/n, 0)
   as DESIGN D3 states; up/down and area use all levels; document.
2. Plateaus without E/D-labelled tokens contribute NaN (not 0); NaN-aware trapezoid area;
   record n_levels_with_data.
3. Relaxation: fit tau on the unwindowed per-token D-indicator (window=1) or convolve the
   model with the same box kernel; windowed series only for plots; document the floor.
4. stats.cluster_bootstrap: add stat_level='cluster' (aggregate per cluster first) and make
   it the documented estimator for the confirmatory contrasts (family means); keep 'row'.
5. paired_contrast: return n_dropped_a / n_dropped_b; warn when the clusters of a pair differ.

## G4 pile_fetch + sae + directions + common
Files: lsne/pile_fetch.py, lsne/sae.py, lsne/directions.py, lsne/experiments/common.py,
tests/test_pile_fetch.py (new), tests/test_sae.py, tests/test_directions.py
1. fetch_sequence: stream=True; require 206 (or 200 with matching Content-Length) before
   reading; parse Content-Range when present; close and raise otherwise.
2. tests/test_pile_fetch.py: shard-straddling byte ranges (index = shard_bytes // 4098),
   non-straddling case, network-marked anchor fetch with hash check, unigram sampling n=2.
3. sae: raise ValueError for unsupported sparsify configs (signed, non-topk activation,
   skip_connection, transcode, W_skip present).
4. directions.language_axis / collect_activations: `include_tail=False` so only paired
   segment-final tokens form the E/D classes (DESIGN §3); common.direction_bank uses it.
5. common: default n_unigram = 2000 (DESIGN §3); smoke keeps a small number.

## G5 tn_lattice tests
Files: tests/test_tn_lattice.py (lsne/tn_lattice.py only if a test exposes a bug)
1. Frozen-attention JVP reference: monkeypatch the eager attention function of
   transformers' GPT-NeoX to detach softmax weights; assert the factorized lattice (r = d)
   matches that reference within a stated tolerance while the full JVP differs.
2. Compare TNCO and opt_einsum contractions against an independent sequential reference.
3. Assert both backends' log10 costs are within one order of magnitude of opt_einsum 'dp'.

## G6 family wording
Files: data/stimuli/families/F05B.json, F09A.json, F16B.json, F06B.json, F08B.json,
data/stimuli/review.csv (adjust rows only)
1. F05B unit 7: align 'line up' / 'stauen sich' (use 'back up' or 'warten in einer Reihe').
2. F09A unit 5: replace clause-final 'noch heute' by an adjunct natural in final position
   in both languages.
3. F16B unit 5: replace 'verleiht ihren Saal' (e.g. 'stellt ihren Saal zur Verfügung').
4. F06B, F08B: refresh stale `notes` fields.
Validate with the family validator and the stimuli/lang_id tests; update review.csv.

## Formulations C and D
For every family (DEV01–DEV04, F01–F20) two further independent paraphrases C and D in
both languages, same meaning units, same target, same top-level keys; schema rules apply;
needed for the realization clouds of DESIGN §4.4 (n ≥ 4 formulations per meaning).

## Integration of stages G–L
Stage runners stage_g_geometry, stage_h_lambda, stage_i_compass, stage_j_rdx,
stage_k_ttt, stage_l_transplant following the conventions of stages A–F; CLI letters;
jobs/shards.yaml; smoke test covering all stages; docs/STAGES.md with a
pre-registration-to-output map (endpoint → csv → columns). Stage L only if
lsne/transplant.py exists and its tests pass; otherwise a stub that writes a manifest.

## Notebooks 06–11 and docs
Generated by scripts/build_notebooks.py with the conventions of 00–05; README stage table
A–L and inputs the user supplies (data/ttt, data/atlas, data/s4); RUNBOOK shard plan;
tests/test_notebooks.py including a slow headless execution of notebooks 00 and 06 with
SMOKE=True on pythia-14m via nbclient.
