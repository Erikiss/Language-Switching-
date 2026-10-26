# Stages A–L: what each stage writes, and where every pre-registered endpoint lives

Stand 25.10.2026. Companion of `docs/DESIGN.md` §8 (the staged plan) and
`docs/PREREGISTRATION.md` §2–3 (the endpoints). Every stage is a module
`lsne/experiments/stage_<letter>_*.py` with

```python
run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pandas.DataFrame]
```

that writes `<out>/<stage>/<model>@<revision>/<table>.csv` (+ `.parquet` when pyarrow is
installed) and a `manifest.json` (git sha, model, limits, runtime, per-stage metadata, and
for G–L a `prereg` map and a `substages` report). `python -m lsne.cli run --stage all`
runs A B C D E F G H I J K L in that order on one loaded model and prints the per-stage
durations; `python -m lsne.cli merge` joins the csv files of several runs.

Conventions shared by all stages (`lsne/experiments/common.py`):

* **Sites.** "Dossier site" = block input of the last block at the last context position
  (`L11.block_input` for pythia-160m, `L5.block_input` for pythia-14m); "depth site" =
  block input of block `n_layers // 3`; "layer grid" = the pre-registered
  `attribution_layer_grid` ({2, 5, 8, 11} for 12 blocks, {1, 2, 4, 5} for 6); residual
  "snapshots" = block inputs 0..L−1 plus the output of the last block (L+1 entries).
* **Family domain groups** (`family_domain`): culture/politics F13–F16, law F17–F18,
  protein misfolding F19–F20, everything else (F01–F12, DEV01–DEV04, the original context)
  = everyday / stable (PREREGISTRATION contrast 13).
* **Bootstraps.** `*_boot` tables are family cluster bootstraps (`lsne.stats.cluster_bootstrap`,
  cluster = family, `stat_level='cluster'` = family means, `Limits.n_boot` resamples) with
  `mean, ci_low, ci_high, se, n_clusters, n_rows`.
* **Substages.** Stages G–L run every substage under `common.Substages`: a failure is
  recorded in `manifest.json → substages.errors` and `substage_errors.txt`, the table gets
  its empty default and the stage goes on. The smoke test requires `n_failed == 0`.
* **Placeholders are flagged, never silent.** Rows computed with a synthetic random SAE
  (smoke runs without a published SAE) carry `sae_source = "synthetic_random"`; TTT rows on
  the built-in public-domain text carry `is_placeholder = True`. Such values are function
  tests, not findings.
* **Limits.** `Limits` (common.py) holds every size knob; `Limits.smoke()` is the CPU preset
  used by `--smoke` (the `SMOKE_GL` dict for G–L). Full-run defaults follow DESIGN /
  PREREGISTRATION (32 random directions / bases, 2 000 Pile sequences, λ_rel sweep
  0…30, 100 path steps, …).

Inputs the user supplies (all optional; the stages run without them and say so in the
manifest): `data/ttt/dao_fork` (+ `neutral`, `nofork`) for stage K, `data/atlas/*.txt`
for stage I, `data/s4/axes.npz` for the ARI-orthogonal `ent` direction (`--axis-npz`),
and a real source model for stage L (`--transplant-source model@revision`).

---

## Pre-registration → output map

| # | Endpoint (PREREGISTRATION §2–3) | Stage | Table | Row / columns that carry the statistic |
|---|---|---|---|---|
| 1° | λ(K, ε, dir) growth exponent, H1: λ(K>0, lang_dom) − λ(K=0, lang_dom) > 0, > rand | D | `damage`, `damage_phase` | `inject="depth"`, `dir_family`, `eps`, `lam`; phase means per (K, C, dir_family, eps) |
| 1 | T(K): K=8 vs K=0 at C1 | C | `contrasts`, `contrasts_boot` | `quantity="T"`, `K=8`, `C="C1"` |
| 2 | K × C: T(8, C3) − T(8, C1) | C | `contrasts_boot` | the two `quantity="T"` rows (K=8, C3 / C1) |
| 3 | R_ws+punct+boundary − R_random | C | `attribution_contrasts`, `contrasts_boot` | `class="delim"`, `quantity="R_excess"` per layer |
| 4 | A_hyst(lang_dom) − A_hyst(restart null) | D | `hysteresis` | `direction="lang_dom"`, `area_minus_null`, `area_z` |
| 5 | Non-monotonicity index K=8 vs K=0 | D | `lens_summary` | `nmi_last`, `nmi_tail` per variant |
| 6 | B(x) with `ent` vs 32 random directions | C | `grid_summary`, `contrasts_boot` | `dir_family="ent"` vs `"rand"` (`quantity="I"`) |
| 7 | **G0 gate**: d(A_E, A_D) < d(A_E, B_E) < d(other family) in ≥ 50 % of layers (Mahalanobis, mean pool, k=5) | G | `pair_ordering`, `g0_gate` | per snapshot: `category`, `mean_mahal`, `core_order_ok`; gate row: `frac_ok`, `pass_all`, `pass_middle`, `degenerate` (r_eff ≥ N−1 → uninformative; `pass_all`/`pass_middle` are then empty, never True); the gate rank is capped at (N−1)//4 (`r`, requested `r_requested`) |
| 8 | **G1 contraction**: Δ log-Vol(K=8, C1) − Δ log-Vol(K=0) (mean pool, k=5) and ΔPR (last token) | G | `contraction`, `contraction_boot` | `preregistered=True` rows: `d_log_volume` (readout `mean`), `d_participation_ratio` (readout `last`), n-matched (see below), per snapshot; raw values in `*_raw` |
| 9 | **T3 steering validity**: cos(steering displacement at ε=0.05, mean(K=8) − mean(K=0)) ≥ 0.3 in the layer of maximal discrete displacement, vs 32 random | G | `steering`, `steering_boot` | `inject_site`, `snapshot`, `eps=0.05`, `cos_steer_vs_K`, `cos_rand_mean/sd`, `at_max_disp_layer`; boot row `quantity="cos_at_max_disp_layer"`, `min_effect_met` |
| 10 | **λ-dynamics**: Spearman ρ(λ, ·) for SAE FVU (ρ<0), steps to crossing, Mahalanobis to the target cloud (ρ<0); features with \|ρ\| ≥ 0.7 | H | `lambda_corr`, `lambda_boot`, `lambda_features(_summary)` | `discriminant="target_logit"`, `preregistered=True`, `quantity ∈ {sae_fvu, steps_to_cross_censored, mahal_to_ref_mean}` (uncrossed paths at n_steps+1; raw `steps_to_cross` secondary), `expected_sign`, `sign_as_expected`; features: `candidate` (\|ρ\| ≥ 0.7), `candidate_all` |
| 11 | B(x)(λ) for State 2143: sign at λ_rel = 3 vs λ = 0 | H | `lambda_original` | `context="original_2143"`, `B_lambda0`, `B_lambda3`, `same_sign` |
| 12 | Domain profile of `ent` at h₂₀₇ (z vs 32 random): is `law` top? | I | `compass`, `compass_summary` | `state_source="original_2143"`, `direction="ent"`, `lang="E"`: per-domain `z`, `rank`; summary `law_z`, `law_rank`, `law_is_top` |
| 13 | Direction × domain: B(dom_culture_politics after counter-steering dom_law) − B(lang_dom) on culture/politics families, ≥ 0.05 nats | I | `direction_domain`, `direction_domain_T`, `direction_domain_boot` | `direction ∈ {dom_culture_politics_cs, dom_culture_politics, dom_law, lang_dom, rand}`, `family_domain`; boot `quantity="B_cs_minus_lang_dom"`, `family_domain="culture_politics"`, `K=8`, `preregistered`, `min_effect_met`; `T` per formulation |
| 14 | Head arm: B at the worst head vs residual stream, same relative dose | I | `head_effects`, `worst_head`, `head_steering`, `head_steering_boot` | `selected=True` head chosen on `selection_families` only, steering tested on the disjoint `evaluation_families` (manifest `worst_head`; `--worst-head L:H` freezes a head from a development run); `arm ∈ {head, residual}`; boot `quantity="B_head_minus_residual"` |
| 15 | **RDX**: fraction of ‖Δ‖² (culture/politics ↔ misfolding) explained by the domain axes vs 32 random bases, z ≥ 2 in ≥ half of the layers | J | `rdx`, `rdx_summary`, `rdx_stability` | `contrast="culture_politics_vs_misfolding"`, `basis="domain_axes"`, `readout="last"`, `preregistered=True`: `explained`, `z`, `z_ge_2`; summary `frac_z_ge_2`, `min_effect_met`; stability over family pairs |
| 16 | Empirical difficulty ranking: mean token NLL per Pile subfield (N ≥ 2000); rank of `protein_misfolding` | J | `pile_difficulty`, `pile_ranking`, `pile_rank_check` | `rank`, `in_hardest_third`, `hardest`, `counterpole` (the replacement rule applied) |
| 17 | **TTT**: (i) memorisation speed vs matched controls, (ii) Δ sharing / Δ activity vs the three control arms, (iii) Δ T(K=8), Δ B(x) | K | `ttt_memorisation(_boot)`, `ttt_before_after`, `ttt_effects` | (i) `arm="injection-<control>"`, `quantity="steps_to_threshold"`; (ii) `measure ∈ {sharing_jaccard, activity_*}`; (iii) `measure ∈ {T_K8, B_lang_dom}`; `delta`, `delta_injection_minus_control`, `exceeds_all_controls` |
| 18 | **Transplant**: alignment R² hold-out − permuted ≥ 0.2; cos(transferred, native axis) ≥ 0.3 in the best layer; same sign of the fD shift for residual substitution and KV planting as the native axis | L | `transfer_test`, `transfer_summary`, `transplant`, `transplant_summary`, `verdict` | per target layer `r2_gap`, `r2_gap_ok`, `cos_transferred_vs_native` vs `cos_null_*`, `cos_ok`, `sign_agreement`, `validated`; summary = best validated layer; per mode `sign_fD_vs_identity`, `sign_native`, `sign_agrees_with_native`; `verdict.validated` = transfer criteria AND sign agreement for `residual` and `kv`; the transplant runs at the best validated grid layer; sources `self`/`rotated_self` are plumbing checks (`counts_for_prereg=False`, no verdict); full runs default to `EleutherAI/pythia-70m@step143000` |

° = the primary endpoint (§2). The manifests of G–L repeat this map under `prereg`.

---

## Stage G — local geometry and trajectory (`stage_g_geometry.py`)

DESIGN §4.4 (Meaning Contracts under language switching) and §4.5 (trajectory).
Realization clouds per `(family, K, C)`: every formulation of the family × up to
`g_max_masks` admissible unit masks with exactly K switches (`geometry.enumerate_unit_masks`;
K=2 and K=8 have exactly 8, K=4 a seeded subset of 56) realised at level C
(`stimuli.segment_masks_from_unit_mask`); K=0 = the formulations. Every realization passes
the K=0 consistency check of stage A.

| Table | Content | Key columns |
|---|---|---|
| `clouds` | one row per realization | `group, family, formulation, K, C, unit_mask, n_tokens, n_tokens_D, excluded` |
| `geometry` | `geometry.geometry_table` at every snapshot × readout (mean, last) × deflation k (`g_deflate_k`, default 0 and 5) | `site, readout, k, family_id, K, C, n, log_volume, participation_ratio, disp_euclid, disp_mahal, ref_group, r` |
| `pair_ordering` | G0 categories per snapshot, prefix-mean pooled states (the shared English tail excluded from the pooling) of every formulation's all-E and all-D rendering, k=5 | `site, layer, category, n_pairs, mean_mahal, mean_cosine, core_order_ok, r_eff, n_states, degenerate` |
| `g0_gate` | the gate | `n_layers, n_ok, frac_ok, pass_all, pass_middle, degenerate, n_states, n_families` |
| `depth_trajectory` | T1 (`trajectory.depth_trajectory`) | `layer, log_volume, delta_log_volume, breathing_index, breathing_amplitude, layer_of_max_contraction, net_change` |
| `contraction` | G1 per family × mixed condition × snapshot × readout × k: `log_volume(K,C) − log_volume(K=0)` and the PR difference, raw (`*_raw`: the geometry-table values) **and n-matched** | `d_log_volume, d_participation_ratio, n_mixed, n_ref, n_subsample, preregistered` |
| `contraction_boot` | family bootstrap of the n-matched contrasts | `quantity, K, C, site, readout, k, preregistered, mean, ci_*` |
| `position_traj` | T2 (`trajectory.position_trajectory`) per group at the dossier, middle and first snapshots | `region ∈ {tail, prefix_bin, from_switch}, index, rel_to_last_switch, log_volume, delta_log_volume` |
| `steering` | T3 (`trajectory.steering_tracks_switching`) per family, `lang_dom` injected at the dossier site (stage-B bank) and at the depth site (axis fitted there), doses `g_steer_eps` | `inject_site, snapshot, eps, cos_steer_vs_K, cos_rand_mean, cos_rand_sd, cos_excess, norm_K, disp_mahal_K, at_max_disp_layer` |
| `steering_boot` | family bootstrap per (inject site, snapshot, eps) + `cos_at_max_disp_layer` | `quantity, preregistered, min_effect_met, mean, ci_*` |
| `coupling` | T5 (`trajectory.couple_with_dynamics`): Spearman over layers between Δ log-Vol and λ_ℓ (from stage D `damage_profile`, depth injection, lang_dom: ln D(ℓ+1) − ln D(ℓ)) / lens KL (stage D `lens`) | `family_id, K, C, readout, k, partner ∈ {lambda, kl}, rho, pvalue, n` (empty without a stage-D run for the model) |

**n-matching.** The log-volume sums `min(n, d)` eigenvalues, so clouds of different n are
not comparable (in the smoke run the raw contrast is −89 nats against an n-matched −0.6).
The confirmatory contrast re-samples the mixed cloud to the K=0 cloud's size
(`g_n_subsample` seeded draws, averaged). Both numbers are reported.

Also written: nothing beyond the tables; the clouds are shared with stages H and K via `ctx`.

## Stage H — λ dynamics (`stage_h_lambda.py`)

DESIGN §4.6 (Biggio et al., Algorithm 1 in activation space). Start = the dossier-site
residual at the last position of a K=0 variant (`h_max_starts` per family); target
density = KDE over the last-position states of the K=8/C1 realization cloud of all
selected families (Mahalanobis-whitened, rank `h_cov_rank`, `k_nearest = h_knn`, median
bandwidth); discriminants `h_discriminants` — `target_logit` (b, pre-registered; boundary
= B ≥ `h_cross_nats`), `probe` (a, the stage-B language probe; boundary = midpoint between
the start and the target-cloud mean), `entropy` (b′), `sae_feature` (c, only with an SAE).
λ = λ_rel · `lambda_scale(range g, n_eff, bw)`, λ_rel ∈ `h_lambdas_rel`; paths of
`h_n_steps` unit steps (`h_step_rel`·‖h₀‖) in the l2 ball `h_dmax_rel`·‖h₀‖.

| Table | Content |
|---|---|
| `lambda_paths` | per start × discriminant × λ_rel: path geometry (`trajectory_geometry`), data support of the endpoint (`manifold_stats`: `mahal_to_ref_mean, knn_mean_dist, local_log_volume, sae_fvu, n_shared_features_with_ref`), next-token effect of inserting the endpoint (`endpoint_effects`: `logp_target, entropy, p_freq, B`) and `kl_from_base`; `crossed`, `reason` |
| `lambda_steps` | g, density, F, distance per step of every path |
| `lambda_corr` | Spearman ρ of every path quantity with λ per start × discriminant (`preregistered`, `expected_sign`) |
| `lambda_boot` | family bootstrap of ρ per (discriminant, quantity); `sign_as_expected` |
| `lambda_features` / `_summary` | `feature_lambda_correlation` of the SAE latents at the endpoints; `candidate` = \|ρ\| ≥ 0.7, `active_only_lambda0/_pos` |
| `lambda_original` | the same sweep from the State-2143 context (target 22905; target cloud = original K>0 states + the standard K=8/C1 cloud), `B_lambda0`, `B_lambda3`, `same_sign` |

Full-run cost: the model-own discriminants need one forward+backward per step
(≈ 2 × 80 starts × 7 λ × 100 steps ≈ 110 k passes on pythia-160m); lower `h_n_steps` or
`h_max_starts` on a small GPU.

## Stage I — domain compass, domain axes, heads (`stage_i_compass.py`)

DESIGN §4.7. Domain lexicon `data/lexicon/domains.json` (11 domains incl.
`biomed_protein` and `protein_misfolding`; only single-token words with a leading space
enter the projections, dropped words are counted in the manifest).

| Table | Content |
|---|---|
| `compass` | `atlas.domain_profile` of h₂₀₇ itself (`direction="state"`) and of ε=0.05 steps along `ent`, `lang_dom`, `freq` and the domain axes (z vs `i_n_random_profile` random directions), both lexicon languages; `rank`, `is_top`; SAE decoder profiles appended with `--use-sae` |
| `compass_summary` | per direction: `top_domain`, `law_z`, `law_rank`, `law_is_top` (contrast 12 is documentation, not a test) |
| `atlas_overlap` | `atlas.atlas_domain_overlap` of the client's extraction under `data/atlas` (`--atlas-dir`); empty without files |
| `domain_axes` | `dom_law`, `dom_culture_politics` (culture_values + politics), `dom_protein_misfolding`, each vs `everyday` at the dossier site (`atlas.domain_axis`, denoised difference of means); cosines to the bank directions; vectors in `directions_domain.npz` |
| `direction_domain` | per K=0 / K=8-C1 variant × direction × dose `i_doses`: B(x) for `dom_culture_politics_cs` (counter-steer: `dom_law` component projected out, then the `dom_culture_politics` dose, written with `Injector(mode="replace")`), `dom_culture_politics`, `dom_law`, `lang_dom`, `rand` (mean over the bank's random directions); `family_domain`, `cos_after_remove`, audit |
| `direction_domain_T` | T(K=8, C1) per formulation with its family group |
| `direction_domain_boot` | family bootstrap of B per (family_domain, direction, K, eps) and of `B_cs_minus_lang_dom` (contrast 13; `min_effect_met` = ≥ 0.05 nats) |
| `head_effects` | mean-ablation effect of every head of the layer grid on log p(target) (`heads.head_ablation_effects`) |
| `head_direct` | direct logit effect of every head at the last position |
| `worst_head` | `heads.worst_head` ranking: weak = culture/politics families, stable = everyday; `selection="fallback_all_families_vs_none"` when no culture/politics family is selected (smoke); `selected` |
| `head_steering` | contrast 14: B when the direction is applied through the worst head (`heads.steer_at_head`, dose relative to the head's own RMS) vs added at the block output of the same layer with the same relative dose; `arm`, `rms_delta`, `cos_realised_direction`, `preimage_rel_err` |
| `head_steering_boot` | family bootstrap of `B_head − B_residual` per (direction, eps) |

## Stage J — RDX, Pile difficulty, sharing (`stage_j_rdx.py`)

DESIGN §4.8 a–c.

| Table | Content |
|---|---|
| `pile_difficulty` | `pile_domains.sample_sequence_difficulty`: `j_n_pile` Pile sequences by range request (cached under `$LSNE_CACHE/pile_nll`), subfield by keyword profile (`PILE_MARKERS`), mean token NLL (first `j_pile_max_len` tokens), `subfield_lexicon` (lexicon-domain classification); needs network (`LSNE_OFFLINE` → recorded failure) |
| `pile_ranking` | `difficulty_ranking` (hardest first; bootstrap CI over sequences) |
| `pile_rank_check` | `misfolding_rank_check` + `counterpole` (contrast 16 replacement rule) |
| `rdx` | `rdx.rdx_table` for the contrasts `culture_politics_vs_misfolding`, `law_vs_culture_politics` (K=0 variants of the family groups; absent when a group is not selected), `E_vs_D` (all-E vs all-D rendering of every formulation), `K0_vs_K8` (K=0 vs K=8/C1) at the grid sites (`j_max_sites` keeps the last ones), readouts last / mean; bases `domain_axes` (fitted at the site), `sae` (decoder rows of the bank's language-selective features at the SAE site, or the most active latents of a synthetic placeholder), `heads` (head contributions of the block at the site), `pca` (top `j_n_pca` components of the pooled contrast states); z vs `j_n_random_bases` random bases; `unit="pooled"` rows plus per-family / per-family-pair rows |
| `rdx_summary` | per (contrast, basis, readout): `frac_z_ge_2`, `min_effect_met` (contrast 15: ≥ half of the sites) |
| `rdx_stability` | `rdx.stability` across units |
| `sharing` | `sharing.sharing_table` at the dossier site: Jaccard sharing index between the conditions (family domain groups of the K=0 variants, K0, K8C1), network activity (`frac_active, mean_mass, participation_ratio, n_active_mean, frac_features_active`), head-profile cosines over the layer grid; `sae_source` |

## Stage K — test-time training (`stage_k_ttt.py`)

DESIGN §4.8 d. Corpus `k_corpus` / `--ttt-corpus` (default `data/ttt/dao_fork`; the
public-domain placeholder with `is_placeholder=True` when absent); controls `neutral`
(`data/ttt/neutral`), `nofork` (`data/ttt/nofork`), `permuted` (token-permuted injection
corpus). Training = `ttt.test_time_train` on a deep copy (AdamW, grad clipping 1.0, FP32,
`k_train_steps` × `k_train_batch`, `k_train_lr`); the frozen model is never mutated.

| Table | Content |
|---|---|
| `ttt_corpora` | per arm: source, `is_placeholder`, n_docs, n_snippets, n_tokens |
| `ttt_memorisation` | contrast 17 (i): `ttt.ttt_memorisation_table` per checkpoint (`k_checkpoints` / `--ttt-checkpoints step4000,step98000,step143000`) × arm × snippet: `nll_start, steps_to_threshold` (NLL ≤ `k_threshold` = 0.5 nats), `final_nll, auc_drop` |
| `ttt_memorisation_boot` | bootstrap over snippets per (checkpoint, arm) and of the paired `injection-<control>` differences |
| `ttt_training` | loss and gradient norm per step and arm |
| `ttt_before_after` | per phase (before / after) and arm: `logp_target` per variant, `T_K8` per formulation, `B_lang_dom` (dose `k_eps` along the pre-TTT axis), `sharing_jaccard` (K0 vs K8C1), `activity_*` per condition, `head_delta` of stage I's worst head, `log_volume` / `participation_ratio` of the K=0 and K=8/C1 clouds (stage G) |
| `ttt_effects` | per arm × measure × condition: `delta` (after − before), `delta_injection_minus_control`, `exceeds_all_controls` (an effect counts only when the injection arm exceeds all three controls, DESIGN §4.8 d 3) |

## Stage L — cross-model transplant (`stage_l_transplant.py`)

DESIGN §4.9, `docs/BLOCKGPT_FEASIBILITY.md`. Source = `l_source` / `--transplant-source`:
`rotated_self` (default: the target's residuals through a seeded isometry 3·H·Q + 0.5,
`transplant.RotatedSourceModel` — the alignment must undo Q, so the check is not
tautological), `self` (plumbing only) or `model@revision` (an HF causal LM such as
`EleutherAI/pythia-70m@step143000`). A BlockGPT-like trace model plugs into the same
`SourceModel` interface. Paired items = the English / German unit sentences of the
selected families (`common.family_text_pairs`, `l_n_texts` per language) + the tails.

| Table | Content |
|---|---|
| `transfer_test` | `transplant.functional_transfer_test` per grid layer (source layer = same index): `cos_transferred_vs_native` vs `cos_null_vs_native` / `cos_null_max` / `z_cos_vs_null` (`l_n_null` permuted-pair draws), `r2_holdout, r2_permuted, r2_gap`, German-fraction shifts `delta_fD_transferred / _native`, `sign_agreement`, `readout_alive`, `n_ED_*`; flags `r2_gap_ok` (≥ 0.2), `cos_ok` (≥ 0.3), `validated` |
| `transfer_summary` | the best layer (largest cosine among the validated layers, else overall), `n_validated_layers` |
| `layer_matching` | `transplant.layer_matching` R² over source × target grid layers |
| `alignments` | one `transplant.layer_alignments` map per target layer 0..L (what plants the KV prefix at every layer) |
| `transplant` | `transplant.transplant_experiment` per prompt (K=0 contexts with their targets) × mode (residual / kv / both) × control (none / random / identity / permuted_alignment): `entropy, logp_target, fD, written_norm, kv_layers` |
| `transplant_summary` | per mode: transplant minus each control (fD, logp), `sign_fD_vs_identity`, `sign_native`, `sign_agrees_with_native` (contrast 18, last clause) |

If `lsne.transplant` is not importable the stage writes a manifest with `stub=True` and no
tables.

---

## Running

```
python -m lsne.cli run --stage all --model EleutherAI/pythia-160m --revision step98000 --out runs \
    [--use-sae] [--ttt-corpus data/ttt/dao_fork] [--ttt-checkpoints step4000,step98000,step143000] \
    [--transplant-source EleutherAI/pythia-70m@step143000] [--atlas-dir data/atlas]
python -m lsne.cli run --stage all --model EleutherAI/pythia-14m --revision step98000 --device cpu \
    --families DEV01 --out /tmp/lsne_smoke --smoke        # CPU smoke run, A-L
```

`jobs/shards.yaml` gives the A100 plan (shards 1–3 run A–E and G–L; shard 7 runs stage K
over the dossier checkpoints). Smoke-run durations on the 4-core CPU box (pythia-14m, DEV01,
unloaded; the test suite's budget is 1 200 s): A 0.2 s, B 0.4 s, C 7.5 s, D 12.3 s, E 1.5 s,
F 16.1 s, G 2.4 s, H 2.8 s, I 1.5 s, J 0.8 s (≈ 5 s on the first run: three Pile range
requests), K 6.1 s, L 3.9 s — 56 s of stage time, 63 s wall.
